#### Importando Bibliotecas
import json
import math
import time
from pathlib import Path

import modal

### ============= ### ###  Hiperparametros  ### ###  ============= ###

# Numero de Reynolds e parametro analitico
RE = 40.0
LAM = RE / 2.0 - math.sqrt((RE / 2.0) ** 2 + 4.0 * math.pi ** 2)

# Dominio fisico (deve casar com 1_preprocess.py)
X_MIN, X_MAX = -0.5, 1.0
Y_MIN, Y_MAX = -0.5, 1.5
X_INFER_MAX = 2.5

# Otimizador / scheduler
LR_PINN = 1e-3
LR_MIX = 1e-2
SCHED_STEP = 3000
SCHED_GAMMA = 0.5

# MixFunn: annealing da temperatura
T_INIT_MIX = 5.0
T_FINAL_MIX = 0.05

# Loop de treino
EPOCHS_LOG = 100
ITER_T1 = 15000   # Tabela 1 (sup vs nsup, 3 seeds)
ITER_T2 = 10000   # Tabela 2 (sweep arquitetural)
SEEDS_T1 = [21, 22, 23]
SEED_T2 = 21

# Coons BC
S_BC = 5.0

### ============= ### ###  Modal App  ### ###  ============= ###

PARENT = Path(__file__).resolve().parent
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "numpy")
    .add_local_file(str(PARENT / "mixfunn.py"), "/root/mixfunn.py", copy=True)
)
app = modal.App("nnphysics-exp01", image=image)
volume = modal.Volume.from_name("tcc", create_if_missing=True)
VOLUME_PATH = "/data"
PREPROCESS_DIR = "/data/preprocess/exp_01"
CHECKPOINT_DIR = "/data/checkpoints/exp_01"


### ============= ### ###  Treino de um config  ### ###  ============= ###

@app.function(
    gpu="T4",
    timeout=120 * 60,
    volumes={VOLUME_PATH: volume},
    max_containers=35,
)
def train_one(
    label: str,
    kind: str,
    n_layers: int,
    width: int,
    mode: str,
    seed: int,
    lr: float,
    iterations: int,
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

    ### ============= ### ###  Solucao analitica  ### ###  ============= ###
    def kovasznay(x, y):
        u = 1.0 - tc.exp(LAM * x) * tc.cos(2.0 * math.pi * y)
        v = (LAM / (2.0 * math.pi)) * tc.exp(LAM * x) * tc.sin(2.0 * math.pi * y)
        p = 0.5 * (1.0 - tc.exp(2.0 * LAM * x))
        return u, v, p

    ### ============= ### ###  Definindo a rede  ### ###  ============= ###
    if kind == "pinn":
        # PINN: MLP com Tanh + inicializacao Xavier
        layers = [2] + [width] * n_layers + [3]
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
        # MixFunn: vem de mixfunn.py (Mix2Funn)
        from mixfunn import Mix2Funn
        net = Mix2Funn(
            n_in=2, n_out=3,
            n_layers=n_layers, n_hidden=width,
            use_softmax=True,
            T_init=T_INIT_MIX, T_final=T_FINAL_MIX,
            n_anneal_epochs=iterations,
            second_order_function=sof,
            dropout=0.0, init_alpha_std=0.1,
        ).to(device)

    n_params = int(sum(p.numel() for p in net.parameters() if p.requires_grad))

    ### ============= ### ###  BC hard via interpolacao Coons  ### ###  ============= ###
    def dist_fn(xy):
        x = xy[:, 0:1]; y = xy[:, 1:2]
        return (tc.tanh(S_BC * (x - X_MIN))
                * tc.tanh(S_BC * (X_MAX - x))
                * tc.tanh(S_BC * (y - Y_MIN))
                * tc.tanh(S_BC * (Y_MAX - y)))

    def u_bc_fn(xy):
        # Interpolacao transfinita de Coons dos 4 bordos da solucao analitica
        x = xy[:, 0:1]; y = xy[:, 1:2]
        xi = (x - X_MIN) / (X_MAX - X_MIN)
        eta = (y - Y_MIN) / (Y_MAX - Y_MIN)
        xm_v = tc.full_like(x, X_MIN); xM_v = tc.full_like(x, X_MAX)
        ym_v = tc.full_like(y, Y_MIN); yM_v = tc.full_like(y, Y_MAX)
        uw, vw, pw = kovasznay(xm_v, y)
        ue, ve, pe = kovasznay(xM_v, y)
        us, vs, ps = kovasznay(x, ym_v)
        un, vn, pn = kovasznay(x, yM_v)
        usw, vsw, psw = kovasznay(xm_v, ym_v)
        use_, vse, pse = kovasznay(xM_v, ym_v)
        unw, vnw, pnw = kovasznay(xm_v, yM_v)
        une, vne, pne = kovasznay(xM_v, yM_v)

        def coons(b_w, b_e, b_s, b_n, c_sw, c_se, c_nw, c_ne):
            horiz = (1.0 - xi) * b_w + xi * b_e
            vert = (1.0 - eta) * b_s + eta * b_n
            corner = ((1.0 - xi) * (1.0 - eta) * c_sw
                      + xi * (1.0 - eta) * c_se
                      + (1.0 - xi) * eta * c_nw
                      + xi * eta * c_ne)
            return horiz + vert - corner

        return (coons(uw, ue, us, un, usw, use_, unw, une),
                coons(vw, ve, vs, vn, vsw, vse, vnw, vne),
                coons(pw, pe, ps, pn, psw, pse, pnw, pne))

    def forward_hard(xy):
        out = net(xy)
        d = dist_fn(xy)
        u_bc, v_bc, p_bc = u_bc_fn(xy)
        u = u_bc + d * out[:, 0:1]
        v = v_bc + d * out[:, 1:2]
        p = p_bc + d * out[:, 2:3]
        return u, v, p

    ### ============= ### ###  Carregando pontos do volume  ### ###  ============= ###
    pre = Path(PREPROCESS_DIR)
    pre_treino = np.load(pre / "treino.npz")
    treino_np = pre_treino["treino"]
    supervisionado_np = pre_treino["supervisionado"]
    val_data = np.load(pre / "validacao.npz")
    eval_data = np.load(pre / "teste.npz")
    extrap_data = np.load(pre / "extrap.npz")

    # Pontos interiores para o residuo da EDP
    x_int_pool = tc.tensor(treino_np[:, 0:1], device=device)
    y_int_pool = tc.tensor(treino_np[:, 1:2], device=device)
    N_INT = x_int_pool.shape[0]

    # Pontos supervisionados (analiticos)
    x_sup = tc.tensor(supervisionado_np[:, 0:1], device=device)
    y_sup = tc.tensor(supervisionado_np[:, 1:2], device=device)
    u_sup = tc.tensor(supervisionado_np[:, 2:3], device=device)
    v_sup = tc.tensor(supervisionado_np[:, 3:4], device=device)
    p_sup = tc.tensor(supervisionado_np[:, 4:5], device=device)

    # Grade de validacao (curva L2 ao longo do treino)
    xv = tc.tensor(val_data["x_lin"], device=device)
    yv = tc.tensor(val_data["y_lin"], device=device)
    XV, YV = tc.meshgrid(xv, yv, indexing="xy")
    xy_v = tc.stack([XV.reshape(-1), YV.reshape(-1)], dim=1)
    u_ref_v = tc.tensor(val_data["u_ref"].reshape(-1, 1), device=device)
    v_ref_v = tc.tensor(val_data["v_ref"].reshape(-1, 1), device=device)
    p_ref_v = tc.tensor(val_data["p_ref"].reshape(-1, 1), device=device)

    ### ============= ### ###  Otimizador  ### ###  ============= ###
    opt = tc.optim.Adam(net.parameters(), lr=lr)
    sched = tc.optim.lr_scheduler.StepLR(opt, step_size=SCHED_STEP, gamma=SCHED_GAMMA)

    loss_momento, loss_massa, loss_total, l2_val_curve, epochs_log = [], [], [], [], []
    t0 = time.perf_counter()

    ### ============= ### ###  Loop de treino  ### ###  ============= ###
    for ep in range(iterations + 1):
        if kind == "mix":
            net.update_temperature_from_epoch(ep)

        # Pontos interiores com gradiente habilitado a cada epoca
        x_int = x_int_pool.detach().clone().requires_grad_(True)
        y_int = y_int_pool.detach().clone().requires_grad_(True)

        if mode == "sup":
            # Treino supervisionado: MSE contra solucao analitica
            xy = tc.cat([x_sup, y_sup], dim=1)
            u_pred, v_pred, p_pred = forward_hard(xy)
            p_pred_n = p_pred - p_pred.mean()
            p_sup_n = p_sup - p_sup.mean()
            loss = (tc.mean((u_pred - u_sup) ** 2)
                    + tc.mean((v_pred - v_sup) ** 2)
                    + tc.mean((p_pred_n - p_sup_n) ** 2))
            l_mom = l_mas = float(loss.item())
        else:
            # Treino nao supervisionado: residuo Navier-Stokes
            xy_int = tc.cat([x_int, y_int], dim=1)
            u, v, p = forward_hard(xy_int)

            def grad(out, inp):
                return tc.autograd.grad(
                    out, inp, grad_outputs=tc.ones_like(out), create_graph=True
                )[0]

            u_x = grad(u, x_int); u_y = grad(u, y_int)
            v_x = grad(v, x_int); v_y = grad(v, y_int)
            p_x = grad(p, x_int); p_y = grad(p, y_int)
            u_xx = grad(u_x, x_int); u_yy = grad(u_y, y_int)
            v_xx = grad(v_x, x_int); v_yy = grad(v_y, y_int)

            # Equacoes de Navier-Stokes incompressivel estacionario
            res_u = u * u_x + v * u_y + p_x - (1.0 / RE) * (u_xx + u_yy)
            res_v = u * v_x + v * v_y + p_y - (1.0 / RE) * (v_xx + v_yy)
            res_div = u_x + v_y
            loss_EDP = tc.mean(res_u ** 2) + tc.mean(res_v ** 2)
            loss_div = tc.mean(res_div ** 2)
            loss = loss_EDP + loss_div
            l_mom = float(loss_EDP.item())
            l_mas = float(loss_div.item())

        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

        if ep % EPOCHS_LOG == 0:
            with tc.no_grad():
                u_v, v_v, p_v = forward_hard(xy_v)
                p_v_n = p_v - p_v.mean()
                p_r_n = p_ref_v - p_ref_v.mean()
                num = ((u_v - u_ref_v) ** 2 + (v_v - v_ref_v) ** 2
                       + (p_v_n - p_r_n) ** 2).sum()
                den = (u_ref_v ** 2 + v_ref_v ** 2 + p_r_n ** 2).sum()
                l2 = float(tc.sqrt(num / den))
            l2_val_curve.append(l2)
            loss_momento.append(l_mom)
            loss_massa.append(l_mas)
            loss_total.append(float(loss.item()))
            epochs_log.append(ep)

    wall = time.perf_counter() - t0
    net.eval()

    ### ============= ### ###  Avaliacao final (grade 80x100)  ### ###  ============= ###
    xe = tc.tensor(eval_data["x_lin"], device=device)
    ye = tc.tensor(eval_data["y_lin"], device=device)
    XE, YE = tc.meshgrid(xe, ye, indexing="xy")
    xy_e = tc.stack([XE.reshape(-1), YE.reshape(-1)], dim=1)
    with tc.no_grad():
        u_e, v_e, p_e = forward_hard(xy_e)
        u_r = tc.tensor(eval_data["u_ref"].reshape(-1, 1), device=device)
        v_r = tc.tensor(eval_data["v_ref"].reshape(-1, 1), device=device)
        p_r = tc.tensor(eval_data["p_ref"].reshape(-1, 1), device=device)
        p_e_n = p_e - p_e.mean(); p_r_n = p_r - p_r.mean()
        num_total = ((u_e - u_r) ** 2 + (v_e - v_r) ** 2 + (p_e_n - p_r_n) ** 2).sum()
        den_total = (u_r ** 2 + v_r ** 2 + p_r_n ** 2).sum()
        l2_val = float(tc.sqrt(num_total / den_total))

    ### ============= ### ###  Inferencia em x estendido  ### ###  ============= ###
    nx_xe = int(extrap_data["x_lin"].shape[0])
    ny_xe = int(extrap_data["y_lin"].shape[0])
    xex = tc.tensor(extrap_data["x_lin"], device=device)
    yex = tc.tensor(extrap_data["y_lin"], device=device)
    XEX, YEX = tc.meshgrid(xex, yex, indexing="xy")
    xy_ex = tc.stack([XEX.reshape(-1), YEX.reshape(-1)], dim=1)
    with tc.no_grad():
        u_ex, _, _ = forward_hard(xy_ex)
    u_an_ex = extrap_data["u_analytic"]

    extrap_x_grid = {
        "x_lin": xex.cpu().numpy().tolist(),
        "y_lin": yex.cpu().numpy().tolist(),
        "u_pred": u_ex.cpu().numpy().reshape(ny_xe, nx_xe).tolist(),
        "u_analytic": u_an_ex.tolist(),
    }

    print(f"[{label}] sof={sof} L2={l2_val:.3e} wall={wall:.1f}s n={n_params}",
          flush=True)

    ### ============= ### ###  Salvando checkpoint + metricas  ### ###  ============= ###
    out_record = {
        "label": label, "kind": kind, "n_layers": n_layers, "width": width,
        "mode": mode, "seed": seed, "lr": lr, "iterations": iterations,
        "sof": sof, "n_params": n_params,
        "l2_val": l2_val, "wall_clock": wall,
        "epochs_log": epochs_log,
        "loss_momento": loss_momento,
        "loss_massa": loss_massa,
        "loss_total": loss_total,
        "l2_val_curve": l2_val_curve,
    }

    try:
        out_dir = Path(CHECKPOINT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{label}.json").write_text(json.dumps(out_record))
        tc.save(net.state_dict(), out_dir / f"{label}.pt")
        # extrapolacao + analitica (separado para arquivo nao explodir o JSON)
        import numpy as np
        np.savez(
            out_dir / f"{label}.npz",
            x_lin=np.array(extrap_x_grid["x_lin"]),
            y_lin=np.array(extrap_x_grid["y_lin"]),
            u_pred=np.array(extrap_x_grid["u_pred"]),
            u_analytic=np.array(extrap_x_grid["u_analytic"]),
        )
        volume.commit()
    except Exception as e:
        print(f"[{label}] WARN commit: {e}", flush=True)

    return out_record


### ============= ### ###  Local entrypoint  ### ###  ============= ###

@app.local_entrypoint()
def main():
    cfg = []

    # === Tabela 1: sup vs nsup, 3 seeds (12 jobs) ===
    for mode in ["sup", "nsup"]:
        for seed in SEEDS_T1:
            cfg.append((
                f"pinn_4x32_{mode}_s{seed}", "pinn", 4, 32,
                mode, seed, LR_PINN, ITER_T1, False,
            ))
    for mode in ["sup", "nsup"]:
        for seed in SEEDS_T1:
            cfg.append((
                f"mix_3x1_{mode}_s{seed}", "mix", 3, 1,
                mode, seed, LR_MIX, ITER_T1, False,
            ))

    # === Tabela 2: sweep PINN (9 jobs) ===
    for nL in [4, 6, 8]:
        for nW in [16, 32, 64]:
            cfg.append((
                f"arch_pinn_{nL}x{nW}", "pinn", nL, nW,
                "nsup", SEED_T2, LR_PINN, ITER_T2, False,
            ))

    # === Tabela 2: sweep MixFunn sof=False (9 jobs) ===
    for nL in [1, 2, 3]:
        for nh in [1, 2, 3]:
            cfg.append((
                f"arch_mix_{nL}x{nh}", "mix", nL, nh,
                "nsup", SEED_T2, LR_MIX, ITER_T2, False,
            ))

    # === MixFunn 1x1 com sof=True (1 job) ===
    cfg.append((
        "mix_1x1_sof_true", "mix", 1, 1,
        "nsup", SEED_T2, LR_MIX, ITER_T1, True,
    ))

    print(f"[main] launching {len(cfg)} jobs Modal T4 (nnphysics-exp01)...",
          flush=True)
    t0 = time.perf_counter()
    results = list(train_one.starmap(cfg))
    wall = time.perf_counter() - t0
    print(f"[main] {len(results)} done in {wall:.1f}s", flush=True)

    # Resumo local
    summary = {r["label"]: r["l2_val"] for r in results}
    print(json.dumps(summary, indent=2))
