#### Importando Bibliotecas
import json
import math
import time
from pathlib import Path

import modal

### ============= ### ###  Hiperparametros do Treino  ### ###  ============= ###

SEED = 21                                        # mesma semente do pre-processamento
RE = 40.0                                        # numero de Reynolds (Kovasznay)
LAM = RE / 2.0 - math.sqrt((RE / 2.0) ** 2 + 4.0 * math.pi ** 2)
X_MIN, X_MAX = -0.5, 1.0                         # dominio em x
Y_MIN, Y_MAX = -0.5, 1.5                         # dominio em y

N_INT = 4000                                     # pontos de colocacao (deve casar com pre-processamento)
LR_PINN = 1e-3                                   # learning rate da PINN
LR_MIX = 1e-2                                    # learning rate da MixFunn
SCHED_STEP = 3000                                # step do StepLR
SCHED_GAMMA = 0.5                                # gamma do StepLR
T_INIT_MIX = 5.0                                 # temperatura inicial do softmax (Mix)
T_FINAL_MIX = 0.05                               # temperatura final do softmax (Mix)
EPOCHS_LOG = 200                                 # frequencia de log da loss
ITERATIONS = 15000                               # iteracoes de treino (mesmas para PINN e Mix)

S_BC = 5.0                                       # parametro do tanh na distancia de Coons
NX_VAL, NY_VAL = 60, 80                          # malha de validacao durante treino
NX_EVAL, NY_EVAL = 80, 100                       # malha de avaliacao final (L2 reportado)

### ============= ### ###  Imagem e Volume Modal  ### ###  ============= ###

PARENT = Path(__file__).resolve().parent
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "numpy")
    .add_local_file(str(PARENT / "mixfunn.py"), "/root/mixfunn.py", copy=True)
)
app = modal.App("nnphysics-exp02", image=image)
volume = modal.Volume.from_name("tcc", create_if_missing=True)

VOLUME_PATH = "/data"
PREPROCESS_DIR = "preprocess/exp_02"
CHECKPOINTS_DIR = "checkpoints/exp_02"


### ============= ### ###  Treino de Uma Configuracao  ### ###  ============= ###

@app.function(gpu="T4", timeout=120 * 60, volumes={VOLUME_PATH: volume}, max_containers=15)
def train_one(
    label: str,
    kind: str,
    n_layers: int,
    width: int,
    dropout: float,
    subsample: float,
    lr: float,
) -> dict:
    import sys
    sys.path.insert(0, "/root")
    import numpy as np
    import torch as tc
    from torch import nn

    tc.set_default_dtype(tc.float64)
    tc.manual_seed(SEED)
    np.random.seed(SEED)
    device = tc.device("cuda" if tc.cuda.is_available() else "cpu")
    lam = LAM

    ### --- Solucao analitica ---
    def kovasznay(x, y):
        u = 1.0 - tc.exp(lam * x) * tc.cos(2.0 * math.pi * y)
        v = (lam / (2.0 * math.pi)) * tc.exp(lam * x) * tc.sin(2.0 * math.pi * y)
        p = 0.5 * (1.0 - tc.exp(2.0 * lam * x))
        return u, v, p

    ### --- Carregamento dos pontos de colocacao pre-processados ---
    pre_dir = Path(VOLUME_PATH) / PREPROCESS_DIR
    treino_xy = tc.load(pre_dir / "treino_colocacao.pt", map_location=device)
    assert treino_xy.shape[0] == N_INT, (
        f"pontos de colocacao com N={treino_xy.shape[0]} != N_INT={N_INT}; rode 1_preprocess.py"
    )
    x_int_pool = treino_xy[:, 0:1].contiguous()
    y_int_pool = treino_xy[:, 1:2].contiguous()

    ### --- Construcao da rede ---
    if kind == "pinn":
        layers = [2] + [width] * n_layers + [3]
        mods = []
        for i in range(len(layers) - 1):
            mods.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                mods.append(nn.Tanh())
                if dropout > 0.0:
                    mods.append(nn.Dropout(dropout))
        net = nn.Sequential(*mods).to(device)
        for m in net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    else:
        from mixfunn import Mix2Funn
        net = Mix2Funn(
            n_in=2, n_out=3, n_layers=n_layers, n_hidden=width,
            use_softmax=True, T_init=T_INIT_MIX, T_final=T_FINAL_MIX,
            n_anneal_epochs=ITERATIONS,
            second_order_function=False,
            dropout=dropout, init_alpha_std=0.1,
        ).to(device)

    n_params = int(sum(p.numel() for p in net.parameters() if p.requires_grad))

    ### --- BC hard via interpolacao transfinita de Coons ---
    def dist_fn(xy):
        x = xy[:, 0:1]; y = xy[:, 1:2]
        return (tc.tanh(S_BC * (x - X_MIN))
                * tc.tanh(S_BC * (X_MAX - x))
                * tc.tanh(S_BC * (y - Y_MIN))
                * tc.tanh(S_BC * (Y_MAX - y)))

    def u_bc_fn(xy):
        x = xy[:, 0:1]; y = xy[:, 1:2]
        xi = (x - X_MIN) / (X_MAX - X_MIN)
        eta = (y - Y_MIN) / (Y_MAX - Y_MIN)
        xm_v = tc.full_like(x, X_MIN); xM_v = tc.full_like(x, X_MAX)
        ym_v = tc.full_like(y, Y_MIN); yM_v = tc.full_like(y, Y_MAX)
        uw, vw, pw = kovasznay(xm_v, y); ue, ve, pe = kovasznay(xM_v, y)
        us, vs, ps = kovasznay(x, ym_v); un, vn, pn = kovasznay(x, yM_v)
        usw, vsw, psw = kovasznay(xm_v, ym_v); use_, vse, pse = kovasznay(xM_v, ym_v)
        unw, vnw, pnw = kovasznay(xm_v, yM_v); une, vne, pne = kovasznay(xM_v, yM_v)

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

    ### --- Otimizador e scheduler ---
    opt = tc.optim.Adam(net.parameters(), lr=lr)
    sched = tc.optim.lr_scheduler.StepLR(opt, step_size=SCHED_STEP, gamma=SCHED_GAMMA)

    ### --- Loop de treino ---
    t0 = time.perf_counter()
    epochs_log, l2_val_curve, loss_total = [], [], []

    for ep in range(ITERATIONS + 1):
        if kind == "mix":
            net.update_temperature_from_epoch(ep)

        # Sub-amostragem dos pontos de colocacao
        if subsample < 1.0:
            n_keep = max(1, int(subsample * N_INT))
            idx = tc.randperm(N_INT, device=device)[:n_keep]
            x_int = x_int_pool[idx].detach().clone().requires_grad_(True)
            y_int = y_int_pool[idx].detach().clone().requires_grad_(True)
        else:
            x_int = x_int_pool.detach().clone().requires_grad_(True)
            y_int = y_int_pool.detach().clone().requires_grad_(True)

        xy_int = tc.cat([x_int, y_int], dim=1)
        u, v, p = forward_hard(xy_int)

        def grad(out, inp):
            return tc.autograd.grad(out, inp, grad_outputs=tc.ones_like(out), create_graph=True)[0]

        u_x = grad(u, x_int); u_y = grad(u, y_int)
        v_x = grad(v, x_int); v_y = grad(v, y_int)
        p_x = grad(p, x_int); p_y = grad(p, y_int)
        u_xx = grad(u_x, x_int); u_yy = grad(u_y, y_int)
        v_xx = grad(v_x, x_int); v_yy = grad(v_y, y_int)

        res_u = u * u_x + v * u_y + p_x - (1.0 / RE) * (u_xx + u_yy)
        res_v = u * v_x + v * v_y + p_y - (1.0 / RE) * (v_xx + v_yy)
        res_div = u_x + v_y
        loss = tc.mean(res_u ** 2) + tc.mean(res_v ** 2) + tc.mean(res_div ** 2)

        opt.zero_grad(); loss.backward(); opt.step(); sched.step()

        if ep % EPOCHS_LOG == 0:
            with tc.no_grad():
                xv = tc.linspace(X_MIN, X_MAX, NX_VAL, device=device)
                yv = tc.linspace(Y_MIN, Y_MAX, NY_VAL, device=device)
                XV, YV = tc.meshgrid(xv, yv, indexing="xy")
                xy_v = tc.stack([XV.reshape(-1), YV.reshape(-1)], dim=1)
                u_v, v_v, p_v = forward_hard(xy_v)
                u_ref, v_ref, p_ref = kovasznay(XV.reshape(-1, 1), YV.reshape(-1, 1))
                p_v_n = p_v - p_v.mean(); p_ref_n = p_ref - p_ref.mean()
                num = ((u_v - u_ref) ** 2 + (v_v - v_ref) ** 2 + (p_v_n - p_ref_n) ** 2).sum()
                den = (u_ref ** 2 + v_ref ** 2 + p_ref_n ** 2).sum()
                l2 = float(tc.sqrt(num / den))
            l2_val_curve.append(l2)
            loss_total.append(float(loss.item()))
            epochs_log.append(ep)

    wall = time.perf_counter() - t0
    net.eval()

    ### --- Avaliacao final 80x100 ---
    xe = tc.linspace(X_MIN, X_MAX, NX_EVAL, device=device)
    ye = tc.linspace(Y_MIN, Y_MAX, NY_EVAL, device=device)
    XE, YE = tc.meshgrid(xe, ye, indexing="xy")
    xy_e = tc.stack([XE.reshape(-1), YE.reshape(-1)], dim=1)
    with tc.no_grad():
        u_e, v_e, p_e = forward_hard(xy_e)
        u_r, v_r, p_r = kovasznay(XE.reshape(-1, 1), YE.reshape(-1, 1))
        p_e_n = p_e - p_e.mean(); p_r_n = p_r - p_r.mean()
        num_total = ((u_e - u_r) ** 2 + (v_e - v_r) ** 2 + (p_e_n - p_r_n) ** 2).sum()
        den_total = (u_r ** 2 + v_r ** 2 + p_r_n ** 2).sum()
        l2_val = float(tc.sqrt(num_total / den_total))

    print(f"[{label}] L2={l2_val:.3e} wall={wall:.1f}s n_params={n_params}", flush=True)

    ### --- Persistencia no volume ---
    out_record = {
        "label": label, "kind": kind, "n_layers": n_layers, "width": width,
        "dropout": dropout, "subsample": subsample, "lr": lr,
        "iterations": ITERATIONS, "seed": SEED,
        "n_params": n_params, "l2_val": l2_val, "wall_clock": wall,
        "epochs_log": epochs_log, "l2_val_curve": l2_val_curve,
        "loss_total": loss_total,
    }
    out_dir = Path(VOLUME_PATH) / CHECKPOINTS_DIR / "by_label"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{label}.json").write_text(json.dumps(out_record))
    tc.save(net.state_dict(), out_dir / f"{label}.pt")
    volume.commit()
    return out_record


### ============= ### ###  Despacho Paralelo das 12 Configuracoes  ### ###  ============= ###

@app.local_entrypoint()
def main():
    configs = []

    # PINN 4x32 — 6 regularizacoes
    configs.append(("pinn_4x32_baseline", "pinn", 4, 32, 0.00, 1.00, LR_PINN))
    configs.append(("pinn_4x32_drop10",   "pinn", 4, 32, 0.10, 1.00, LR_PINN))
    configs.append(("pinn_4x32_drop15",   "pinn", 4, 32, 0.15, 1.00, LR_PINN))
    configs.append(("pinn_4x32_sub70",    "pinn", 4, 32, 0.00, 0.70, LR_PINN))
    configs.append(("pinn_4x32_sub50",    "pinn", 4, 32, 0.00, 0.50, LR_PINN))
    configs.append(("pinn_4x32_combo",    "pinn", 4, 32, 0.10, 0.50, LR_PINN))

    # MixFunn 3x1 — 6 regularizacoes
    configs.append(("mix_3x1_baseline",  "mix",  3,  1, 0.00, 1.00, LR_MIX))
    configs.append(("mix_3x1_drop10",    "mix",  3,  1, 0.10, 1.00, LR_MIX))
    configs.append(("mix_3x1_drop15",    "mix",  3,  1, 0.15, 1.00, LR_MIX))
    configs.append(("mix_3x1_sub70",     "mix",  3,  1, 0.00, 0.70, LR_MIX))
    configs.append(("mix_3x1_sub50",     "mix",  3,  1, 0.00, 0.50, LR_MIX))
    configs.append(("mix_3x1_combo",     "mix",  3,  1, 0.10, 0.50, LR_MIX))

    print(f"[main] despachando {len(configs)} jobs em paralelo no Modal T4...", flush=True)
    t0 = time.perf_counter()
    results = list(train_one.starmap(configs))
    wall = time.perf_counter() - t0
    print(f"[main] {len(results)} jobs concluidos em {wall:.1f}s", flush=True)
    for r in results:
        print(f"  {r['label']:25s} L2={r['l2_val']:.3e} wall={r['wall_clock']:.1f}s n={r['n_params']}", flush=True)
