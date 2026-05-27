"""Modal v19-extras: 6 PINN regularizacao Coons (G2.15) + 2 ref Eficiencia 4x16/2x1 (G2.16).
8 jobs paralelos em nicolas-quinta.
"""

import json
import math
import time
from pathlib import Path

import modal

PARENT = Path(__file__).resolve().parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "numpy")
    .add_local_file(str(PARENT / "mixfunn.py"), "/root/mixfunn.py", copy=True)
)
app = modal.App("tcc-kov-extras", image=image)
volume = modal.Volume.from_name("tcc", create_if_missing=True)

VOLUME_PATH = "/data"
EXPERIMENT = "final/kov_v19_extras"

RE = 40.0
LAM = RE / 2.0 - math.sqrt((RE / 2.0) ** 2 + 4.0 * math.pi ** 2)
X_MIN, X_MAX = -0.5, 1.0
Y_MIN, Y_MAX = -0.5, 1.5

N_INT = 4000
LR_PINN = 1e-3
LR_MIX = 1e-2
SCHED_STEP = 3000
SCHED_GAMMA = 0.5
T_INIT_MIX = 5.0
T_FINAL_MIX = 0.05
EPOCHS_LOG = 200


@app.function(gpu="T4", timeout=120 * 60, volumes={VOLUME_PATH: volume}, max_containers=10)
def train_one(
    label: str,
    kind: str,
    n_layers: int,
    width: int,
    mode: str,
    seed: int,
    dropout: float,
    subsample: float,
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
    lam = LAM

    def kovasznay(x, y):
        u = 1.0 - tc.exp(lam * x) * tc.cos(2.0 * math.pi * y)
        v = (lam / (2.0 * math.pi)) * tc.exp(lam * x) * tc.sin(2.0 * math.pi * y)
        p = 0.5 * (1.0 - tc.exp(2.0 * lam * x))
        return u, v, p

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
            n_anneal_epochs=iterations,
            second_order_function=sof,
            dropout=dropout, init_alpha_std=0.1,
        ).to(device)

    n_params = int(sum(p.numel() for p in net.parameters() if p.requires_grad))

    # BC hard Coons
    S_BC = 5.0

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

    opt = tc.optim.Adam(net.parameters(), lr=lr)
    sched = tc.optim.lr_scheduler.StepLR(opt, step_size=SCHED_STEP, gamma=SCHED_GAMMA)

    g = tc.Generator(device=device).manual_seed(seed)
    x_int_pool = tc.rand(N_INT, 1, device=device, generator=g) * (X_MAX - X_MIN) + X_MIN
    y_int_pool = tc.rand(N_INT, 1, device=device, generator=g) * (Y_MAX - Y_MIN) + Y_MIN

    t0 = time.perf_counter()
    epochs_log, l2_val_curve = [], []

    for ep in range(iterations + 1):
        if kind == "mix":
            net.update_temperature_from_epoch(ep)

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
                nx_v, ny_v = 60, 80
                xv = tc.linspace(X_MIN, X_MAX, nx_v, device=device)
                yv = tc.linspace(Y_MIN, Y_MAX, ny_v, device=device)
                XV, YV = tc.meshgrid(xv, yv, indexing="xy")
                xy_v = tc.stack([XV.reshape(-1), YV.reshape(-1)], dim=1)
                u_v, v_v, p_v = forward_hard(xy_v)
                u_ref, v_ref, p_ref = kovasznay(XV.reshape(-1, 1), YV.reshape(-1, 1))
                p_v_n = p_v - p_v.mean(); p_ref_n = p_ref - p_ref.mean()
                num = ((u_v - u_ref) ** 2 + (v_v - v_ref) ** 2 + (p_v_n - p_ref_n) ** 2).sum()
                den = (u_ref ** 2 + v_ref ** 2 + p_ref_n ** 2).sum()
                l2 = float(tc.sqrt(num / den))
            l2_val_curve.append(l2)
            epochs_log.append(ep)

    wall = time.perf_counter() - t0
    net.eval()

    # Aval final 80x100
    nx_e, ny_e = 80, 100
    xe = tc.linspace(X_MIN, X_MAX, nx_e, device=device)
    ye = tc.linspace(Y_MIN, Y_MAX, ny_e, device=device)
    XE, YE = tc.meshgrid(xe, ye, indexing="xy")
    xy_e = tc.stack([XE.reshape(-1), YE.reshape(-1)], dim=1)

    # Tempo de inferencia (forward + eval)
    t_inf = time.perf_counter()
    with tc.no_grad():
        u_e, v_e, p_e = forward_hard(xy_e)
    if device.type == "cuda":
        tc.cuda.synchronize()
    t_inf = time.perf_counter() - t_inf
    inference_us_per_pt = (t_inf * 1e6) / (nx_e * ny_e)

    with tc.no_grad():
        u_r, v_r, p_r = kovasznay(XE.reshape(-1, 1), YE.reshape(-1, 1))
        p_e_n = p_e - p_e.mean(); p_r_n = p_r - p_r.mean()
        num_total = ((u_e - u_r) ** 2 + (v_e - v_r) ** 2 + (p_e_n - p_r_n) ** 2).sum()
        den_total = (u_r ** 2 + v_r ** 2 + p_r_n ** 2).sum()
        l2_val = float(tc.sqrt(num_total / den_total))

    print(f"[{label}] L2={l2_val:.3e} wall={wall:.1f}s inf={inference_us_per_pt:.1f}us/pt n={n_params}", flush=True)

    out_record = {
        "label": label, "kind": kind, "n_layers": n_layers, "width": width,
        "mode": mode, "seed": seed, "dropout": dropout, "subsample": subsample,
        "lr": lr, "iterations": iterations, "sof": sof,
        "n_params": n_params, "l2_val": l2_val, "wall_clock": wall,
        "inference_us_per_pt": inference_us_per_pt,
        "epochs_log": epochs_log, "l2_val_curve": l2_val_curve,
    }

    out_dir = Path(VOLUME_PATH) / EXPERIMENT / "by_label"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{label}.json"
    with open(out_path, "w") as f:
        json.dump(out_record, f, indent=2)
    volume.commit()
    return out_record


@app.local_entrypoint()
def run():
    iters_pinn = 15000
    iters_mix = 15000

    configs = []

    # G2.15: 6 PINN regularizacao Coons
    seed = 21
    configs.append(("reg_pinn_baseline", "pinn", 4, 32, "nsup", seed, 0.00, 1.00, LR_PINN, iters_pinn, False))
    configs.append(("reg_pinn_drop10",   "pinn", 4, 32, "nsup", seed, 0.10, 1.00, LR_PINN, iters_pinn, False))
    configs.append(("reg_pinn_drop15",   "pinn", 4, 32, "nsup", seed, 0.15, 1.00, LR_PINN, iters_pinn, False))
    configs.append(("reg_pinn_sub70",    "pinn", 4, 32, "nsup", seed, 0.00, 0.70, LR_PINN, iters_pinn, False))
    configs.append(("reg_pinn_sub50",    "pinn", 4, 32, "nsup", seed, 0.00, 0.50, LR_PINN, iters_pinn, False))
    configs.append(("reg_pinn_combo",    "pinn", 4, 32, "nsup", seed, 0.10, 0.50, LR_PINN, iters_pinn, False))

    # G2.16: 2 ref Eficiencia
    configs.append(("eff_pinn_4x16",     "pinn", 4, 16, "nsup", seed, 0.00, 1.00, LR_PINN, iters_pinn, False))
    configs.append(("eff_mix_2x1",       "mix",  2,  1, "nsup", seed, 0.00, 1.00, LR_MIX,  iters_mix,  False))

    print(f"Launching {len(configs)} jobs em paralelo...", flush=True)
    results = list(train_one.starmap(configs))
    print(f"Completed {len(results)} jobs.", flush=True)
    for r in results:
        print(f"  {r['label']:25s} L2={r['l2_val']:.3e} wall={r['wall_clock']:.1f}s n={r['n_params']}", flush=True)
