#### Importando Bibliotecas
import json
import time
from pathlib import Path

import modal

### ============= ### ###  Hiperparametros  ### ###  ============= ###

# Dominio fisico (deve casar com 1_preprocess.py)
X_MIN, X_MAX = 0.0, 1.0
Z_MIN, Z_MAX = 0.0, 1.0
DISC_CX, DISC_CZ = 0.5, 0.5
DISC_R = 0.2
MU_OUT, MU_DISC = 1.0, 3.0
SIGMOID_ALPHA = 500.0

# Otimizador / scheduler
LR = 1e-3
SCHED_GAMMA = 0.5

# Fase Adam
ADAM_ITERS = 10000

# Fase L-BFGS (refinamento, executado em sequencia)
LBFGS_OUTER = 300
LBFGS_INNER = 50

# MixFunn: annealing da temperatura
T_INIT_MIX = 5.0
T_FINAL_MIX = 0.05

# Loop de treino
EPOCHS_LOG = 500

# Seed (precisa ser determinista entre 1_preprocess e 2_train)
SEED = 21

### ============= ### ###  Modal App  ### ###  ============= ###

PARENT = Path(__file__).resolve().parent
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "numpy", "scipy")
    .add_local_file(str(PARENT / "mixfunn.py"), "/root/mixfunn.py", copy=True)
)
app = modal.App("nnphysics-exp07", image=image)
volume = modal.Volume.from_name("tcc", create_if_missing=True)
VOLUME_PATH = "/data"
PREPROCESS_DIR = "/data/preprocess/exp_07"
CHECKPOINT_DIR = "/data/checkpoints/exp_07"


### ============= ### ###  Treino de um config  ### ###  ============= ###

@app.function(
    gpu="T4",
    timeout=120 * 60,
    volumes={VOLUME_PATH: volume},
    max_containers=4,
)
def train_one(
    label: str,
    kind: str,
    n_layers: int,
    width: int,
    seed: int,
    lr: float,
    adam_iters: int,
    lbfgs_outer: int,
    lbfgs_inner: int,
    sof: bool = False,
) -> dict:
    import sys
    sys.path.insert(0, "/root")
    import numpy as np
    import torch as tc
    from torch import nn

    tc.set_default_dtype(tc.float64)
    tc.manual_seed(seed)
    np.random.seed(seed)
    device = tc.device("cuda" if tc.cuda.is_available() else "cpu")

    ### ============= ### ###  Carregando referencia FD do volume  ### ###  ============= ###
    pre = Path(PREPROCESS_DIR)
    ref_data = np.load(pre / "referencia.npz")
    Xg = ref_data["x"]; Zg = ref_data["z"]
    Hx_ref = ref_data["Hx_ref"]; Hz_ref = ref_data["Hz_ref"]
    treino_np = np.load(pre / "treino.npz")["treino"]
    NGRID = Xg.shape[0]

    Xg_t = tc.tensor(Xg, dtype=tc.float64, device=device)
    Zg_t = tc.tensor(Zg, dtype=tc.float64, device=device)
    Hx_ref_t = tc.tensor(Hx_ref, dtype=tc.float64, device=device)
    Hz_ref_t = tc.tensor(Hz_ref, dtype=tc.float64, device=device)

    ### ============= ### ###  Definindo a rede  ### ###  ============= ###
    if kind == "pinn":
        # PINN: MLP com Tanh + inicializacao Xavier (duas saidas H_x, H_z)
        layers = [2] + [width] * n_layers + [2]
        mods = []
        for i in range(len(layers) - 1):
            mods.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                mods.append(nn.Tanh())
        net = nn.Sequential(*mods).to(device)
        for m in net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    else:
        # MixFunn com base atomica reduzida a 4 funcoes: sin, cos, identidade, quadrado.
        # A reducao e feita por monkey-patch das listas globais BASE_FUNCTIONS / Q
        # antes de instanciar a rede. O modulo mixfunn.py permanece intacto.
        import mixfunn as _mf

        class Square(nn.Module):
            def forward(self, x: tc.Tensor) -> tc.Tensor:
                return x * x

        _mf.BASE_FUNCTIONS = [_mf.Sin(), _mf.Cos(), _mf.Id(), Square()]
        _mf.BASE_FUNCTION_NAMES = ["sin", "cos", "id", "square"]
        _mf.Q = 4

        from mixfunn import Mix2Funn
        net = Mix2Funn(
            n_in=2, n_out=2,
            n_layers=n_layers, n_hidden=width,
            use_softmax=True,
            T_init=T_INIT_MIX, T_final=T_FINAL_MIX,
            n_anneal_epochs=adam_iters,
            second_order_function=sof,
            dropout=0.0, init_alpha_std=0.1,
        ).to(device)

    n_params = int(sum(p.numel() for p in net.parameters() if p.requires_grad))

    ### ============= ### ###  BC hard via lift  ### ###  ============= ###
    # H(x) = f(x) * d_0(x) + V(x), com V = (0, 1), d_0 = 1 - exp(-10 * dist_bordo).
    # Garante exatamente H = (0, 1) em todo o bordo do quadrado unitario.
    def d0_fn(x, z):
        d = tc.minimum(tc.minimum(x, 1.0 - x), tc.minimum(z, 1.0 - z))
        return 1.0 - tc.exp(-10.0 * d)

    def forward_hard(xz):
        out = net(xz)
        f_x = out[:, 0:1]
        f_z = out[:, 1:2]
        d0 = d0_fn(xz[:, 0:1], xz[:, 1:2])
        Hx = f_x * d0
        Hz = f_z * d0 + 1.0
        return Hx, Hz

    ### ============= ### ###  Permeabilidade mu_r(x, z)  ### ###  ============= ###
    def mu_fn(x, z):
        r = tc.sqrt((x - DISC_CX) ** 2 + (z - DISC_CZ) ** 2)
        s = tc.sigmoid(SIGMOID_ALPHA * (DISC_R - r))
        return MU_OUT + (MU_DISC - MU_OUT) * s

    ### ============= ### ###  Pontos interiores (LHS do volume)  ### ###  ============= ###
    x_pool = tc.tensor(treino_np[:, 0:1], dtype=tc.float64, device=device)
    z_pool = tc.tensor(treino_np[:, 1:2], dtype=tc.float64, device=device)

    def grad(out, inp):
        return tc.autograd.grad(
            out, inp, grad_outputs=tc.ones_like(out), create_graph=True
        )[0]

    def pde_loss():
        # Curl em y: f_1 = d_z(H_x) - d_x(H_z)
        # Divergencia de B: f_2 = d_x(mu H_x) + d_z(mu H_z)
        # L = mean(f_1^2) + mean(f_2^2)
        x_int = x_pool.detach().clone().requires_grad_(True)
        z_int = z_pool.detach().clone().requires_grad_(True)
        xz_int = tc.cat([x_int, z_int], dim=1)
        Hx, Hz = forward_hard(xz_int)

        Hx_x = grad(Hx, x_int); Hx_z = grad(Hx, z_int)
        Hz_x = grad(Hz, x_int); Hz_z = grad(Hz, z_int)

        mu = mu_fn(x_int, z_int)
        r = tc.sqrt((x_int - DISC_CX) ** 2 + (z_int - DISC_CZ) ** 2).clamp_min(1e-12)
        s = tc.sigmoid(SIGMOID_ALPHA * (DISC_R - r))
        dmu_dr = -(MU_DISC - MU_OUT) * SIGMOID_ALPHA * s * (1.0 - s)
        dmu_dx = dmu_dr * (x_int - DISC_CX) / r
        dmu_dz = dmu_dr * (z_int - DISC_CZ) / r

        res_curl = Hx_z - Hz_x
        res_div = mu * (Hx_x + Hz_z) + Hx * dmu_dx + Hz * dmu_dz
        L_curl = tc.mean(res_curl ** 2)
        L_div = tc.mean(res_div ** 2)
        return L_curl + L_div, L_curl, L_div

    def eval_l2():
        net.eval()
        with tc.no_grad():
            xz_v = tc.stack([Xg_t.reshape(-1), Zg_t.reshape(-1)], dim=1)
            Hx_v, Hz_v = forward_hard(xz_v)
            Hx_v = Hx_v.reshape(NGRID, NGRID); Hz_v = Hz_v.reshape(NGRID, NGRID)
            num_x = ((Hx_v - Hx_ref_t) ** 2).sum()
            den_x = (Hx_ref_t ** 2).sum().clamp_min(1e-20)
            num_z = ((Hz_v - Hz_ref_t) ** 2).sum()
            den_z = (Hz_ref_t ** 2).sum().clamp_min(1e-20)
            l2x = float(tc.sqrt(num_x / den_x))
            l2z = float(tc.sqrt(num_z / den_z))
        net.train()
        return l2x, l2z

    loss_curl_hist, loss_div_hist, loss_total_hist = [], [], []
    l2_curve, epochs_log = [], []
    t0 = time.perf_counter()

    ### ============= ### ###  Fase Adam  ### ###  ============= ###
    print(f"[{label}] phase=adam adam_iters={adam_iters} lr={lr}", flush=True)
    opt = tc.optim.Adam(net.parameters(), lr=lr)
    sched = tc.optim.lr_scheduler.StepLR(
        opt, step_size=max(1, adam_iters // 4), gamma=SCHED_GAMMA
    )
    t_adam_start = time.perf_counter()

    for ep in range(adam_iters + 1):
        if kind == "mix":
            net.update_temperature_from_epoch(ep)

        loss, L_curl, L_div = pde_loss()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()

        if ep % EPOCHS_LOG == 0:
            l2x, l2z = eval_l2()
            l2_curve.append([l2x, l2z])
            loss_curl_hist.append(float(L_curl.item()))
            loss_div_hist.append(float(L_div.item()))
            loss_total_hist.append(float(loss.item()))
            epochs_log.append(ep)
            if ep % (EPOCHS_LOG * 4) == 0:
                print(f"[{label}] adam ep={ep} L={loss.item():.3e} "
                      f"L_curl={L_curl.item():.3e} L_div={L_div.item():.3e} "
                      f"L2x={l2x:.3e} L2z={l2z:.3e}", flush=True)

    wall_adam = time.perf_counter() - t_adam_start

    ### ============= ### ###  Fase L-BFGS  ### ###  ============= ###
    print(f"[{label}] phase=lbfgs outer={lbfgs_outer} inner={lbfgs_inner}", flush=True)
    t_lbfgs_start = time.perf_counter()

    opt_lbfgs = tc.optim.LBFGS(
        net.parameters(),
        max_iter=lbfgs_inner,
        history_size=100,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-10,
        tolerance_change=1e-14,
    )

    state = {"L": None, "Lc": None, "Ld": None}

    def closure():
        opt_lbfgs.zero_grad()
        loss, L_curl, L_div = pde_loss()
        loss.backward()
        state["L"] = loss; state["Lc"] = L_curl; state["Ld"] = L_div
        return loss

    best = {"l2x": float("inf"), "l2z": float("inf"), "l2_tot": float("inf"),
            "state": None}

    for k in range(lbfgs_outer):
        try:
            opt_lbfgs.step(closure)
        except RuntimeError as e:
            print(f"[{label}] LBFGS RuntimeError at k={k}: {e}", flush=True)
            break
        if k % max(1, lbfgs_outer // 30) == 0 or k == lbfgs_outer - 1:
            l2x, l2z = eval_l2()
            ep_log = adam_iters + (k + 1) * lbfgs_inner
            l2_curve.append([l2x, l2z])
            loss_curl_hist.append(float(state["Lc"].item()))
            loss_div_hist.append(float(state["Ld"].item()))
            loss_total_hist.append(float(state["L"].item()))
            epochs_log.append(ep_log)
            tot = (l2x ** 2 + l2z ** 2) ** 0.5
            if tot < best["l2_tot"]:
                best["l2x"] = l2x; best["l2z"] = l2z; best["l2_tot"] = tot
                best["state"] = {k_: v.detach().clone() for k_, v in net.state_dict().items()}
            print(f"[{label}] lbfgs k={k} L={state['L'].item():.3e} "
                  f"L_curl={state['Lc'].item():.3e} L_div={state['Ld'].item():.3e} "
                  f"L2x={l2x:.3e} L2z={l2z:.3e} (best L2_tot={best['l2_tot']:.3e})",
                  flush=True)

    wall_lbfgs = time.perf_counter() - t_lbfgs_start
    wall = time.perf_counter() - t0

    # Restaura melhor estado encontrado durante L-BFGS
    if best["state"] is not None:
        net.load_state_dict(best["state"])

    ### ============= ### ###  Avaliacao final  ### ###  ============= ###
    net.eval()
    with tc.no_grad():
        xz_v = tc.stack([Xg_t.reshape(-1), Zg_t.reshape(-1)], dim=1)
        Hx_pred_t, Hz_pred_t = forward_hard(xz_v)
        Hx_pred = Hx_pred_t.reshape(NGRID, NGRID).cpu().numpy()
        Hz_pred = Hz_pred_t.reshape(NGRID, NGRID).cpu().numpy()
        Hx_ref_arr = Hx_ref_t.cpu().numpy()
        Hz_ref_arr = Hz_ref_t.cpu().numpy()
        num_x = ((Hx_pred - Hx_ref_arr) ** 2).sum()
        den_x = (Hx_ref_arr ** 2).sum() + 1e-20
        num_z = ((Hz_pred - Hz_ref_arr) ** 2).sum()
        den_z = (Hz_ref_arr ** 2).sum() + 1e-20
        l2_hx = float((num_x / den_x) ** 0.5)
        l2_hz = float((num_z / den_z) ** 0.5)
        l2_tot = float((l2_hx ** 2 + l2_hz ** 2) ** 0.5)

    print(f"[{label}] DONE kind={kind} {n_layers}x{width} "
          f"L2_Hx={l2_hx:.3e} L2_Hz={l2_hz:.3e} L2_tot={l2_tot:.3e} "
          f"wall_total={wall:.1f}s wall_adam={wall_adam:.1f}s "
          f"wall_lbfgs={wall_lbfgs:.1f}s n_params={n_params}", flush=True)

    ### ============= ### ###  Salvando checkpoint + metricas  ### ###  ============= ###
    out_record = {
        "label": label, "kind": kind, "n_layers": n_layers, "width": width,
        "mode": "nsup_lift", "seed": seed, "lr": lr,
        "adam_iters": adam_iters, "lbfgs_outer": lbfgs_outer,
        "lbfgs_inner": lbfgs_inner, "sof": sof,
        "n_params": n_params,
        "l2_hx": l2_hx, "l2_hz": l2_hz, "l2_tot": l2_tot,
        "wall_clock": wall, "wall_adam": wall_adam, "wall_lbfgs": wall_lbfgs,
        "epochs_log": epochs_log,
        "loss_curl": loss_curl_hist,
        "loss_div": loss_div_hist,
        "loss_total": loss_total_hist,
        "l2_curve": l2_curve,
    }

    try:
        out_dir = Path(CHECKPOINT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{label}.json").write_text(json.dumps(out_record))
        np.savez(
            out_dir / f"{label}.npz",
            x=Xg, z=Zg,
            Hx_ref=Hx_ref_arr, Hz_ref=Hz_ref_arr,
            Hx_pred=Hx_pred, Hz_pred=Hz_pred,
        )
        tc.save(net.state_dict(), out_dir / f"{label}.pt")
        volume.commit()
    except Exception as e:
        print(f"[{label}] WARN commit: {e}", flush=True)

    return out_record


### ============= ### ###  Local entrypoint  ### ###  ============= ###

@app.local_entrypoint()
def main():
    # Duas configuracoes nao-supervisionadas (lift hard-BC), unificadas Adam -> L-BFGS:
    #   - PINN 8 x 96
    #   - MixFunn-sof 1 x 2 (sof=True, base atomica reduzida a 4 funcoes)
    cfg = [
        ("pinn_8x96_nsup", "pinn", 8, 96, SEED, LR,
         ADAM_ITERS, LBFGS_OUTER, LBFGS_INNER, False),
        ("mix_1x2_sof_nsup", "mix", 1, 2, SEED, LR,
         ADAM_ITERS, LBFGS_OUTER, LBFGS_INNER, True),
    ]
    print(f"[main] launching {len(cfg)} jobs Modal T4 (nnphysics-exp07)...",
          flush=True)
    t0 = time.perf_counter()
    results = list(train_one.starmap(cfg))
    wall = time.perf_counter() - t0
    print(f"[main] {len(results)} done in {wall:.1f}s", flush=True)
    for r in results:
        print(f"  {r['label']:30s} L2_Hx={r['l2_hx']:.3e} "
              f"L2_Hz={r['l2_hz']:.3e} wall={r['wall_clock']:.0f}s",
              flush=True)
    summary = {r["label"]: {"l2_hx": r["l2_hx"], "l2_hz": r["l2_hz"]}
               for r in results}
    print(json.dumps(summary, indent=2))
