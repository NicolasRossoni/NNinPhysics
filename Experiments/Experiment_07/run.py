"""Train PINN and Mix2Funn (unsupervised) on the 2D magnetostatic problem.

Hard boundary conditions are imposed via a lift function f(x) * d_0(x) + V(x)
so that the residual loss focuses on the Maxwell equations
(curl H = 0, div(mu H) = 0) inside the domain.
"""

import json
import time
from pathlib import Path

import modal

PARENT = Path(__file__).resolve().parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "numpy", "scipy")
    .add_local_file(str(PARENT / "mixfunn.py"), "/root/mixfunn.py", copy=True)
)
app = modal.App("tcc-baldan-v23", image=image)
volume = modal.Volume.from_name("tcc", create_if_missing=True)

VOLUME_PATH = "/data"
EXPERIMENT = "final/baldan_v23"

# Dominio + parametros fisicos
X_MIN, X_MAX = 0.0, 1.0
Z_MIN, Z_MAX = 0.0, 1.0
DISC_CX, DISC_CZ = 0.5, 0.5
DISC_R = 0.2
MU_OUT, MU_DISC = 1.0, 3.0
SIGMOID_ALPHA = 500.0

NGRID = 100

T_INIT_MIX = 5.0
T_FINAL_MIX = 0.05
EPOCHS_LOG = 250


# =====================================================================
# FEM reference: potencial escalar com phi = -z no bordo (mesmo v21/v22).
# Equivalente a impor Hx=0 e Hz=1 nas 4 paredes via potencial.
# =====================================================================
def fem_reference(ngrid: int = NGRID):
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    nx = nz = ngrid
    x = np.linspace(X_MIN, X_MAX, nx)
    z = np.linspace(Z_MIN, Z_MAX, nz)
    dx = x[1] - x[0]
    dz = z[1] - z[0]
    X, Z = np.meshgrid(x, z, indexing="xy")

    def k_of(ix, iz):
        return iz * nx + ix

    N = nx * nz
    R = np.sqrt((X - DISC_CX) ** 2 + (Z - DISC_CZ) ** 2)
    s = 1.0 / (1.0 + np.exp(-SIGMOID_ALPHA * (DISC_R - R)))
    mu = MU_OUT + (MU_DISC - MU_OUT) * s
    mu_xface = 2.0 * mu[:, :-1] * mu[:, 1:] / (mu[:, :-1] + mu[:, 1:])
    mu_zface = 2.0 * mu[:-1, :] * mu[1:, :] / (mu[:-1, :] + mu[1:, :])

    rows, cols, vals = [], [], []
    rhs = np.zeros(N)

    def add(r, c, v):
        rows.append(r); cols.append(c); vals.append(v)

    for iz in range(nz):
        for ix in range(nx):
            k = k_of(ix, iz)
            if ix == 0 or ix == nx - 1 or iz == 0 or iz == nz - 1:
                add(k, k, 1.0)
                rhs[k] = -z[iz]
                continue
            mu_e = mu_xface[iz, ix]
            mu_w = mu_xface[iz, ix - 1]
            mu_n = mu_zface[iz, ix]
            mu_s = mu_zface[iz - 1, ix]
            add(k, k_of(ix + 1, iz), mu_e / (dx * dx))
            add(k, k_of(ix - 1, iz), mu_w / (dx * dx))
            add(k, k_of(ix, iz + 1), mu_n / (dz * dz))
            add(k, k_of(ix, iz - 1), mu_s / (dz * dz))
            add(k, k, -(mu_e + mu_w) / (dx * dx) - (mu_n + mu_s) / (dz * dz))
            rhs[k] = 0.0

    A = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
    phi = spla.spsolve(A, rhs).reshape(nz, nx)
    Hx = np.zeros_like(phi)
    Hz = np.zeros_like(phi)
    Hx[:, 1:-1] = -(phi[:, 2:] - phi[:, :-2]) / (2 * dx)
    Hx[:, 0] = -(phi[:, 1] - phi[:, 0]) / dx
    Hx[:, -1] = -(phi[:, -1] - phi[:, -2]) / dx
    Hz[1:-1, :] = -(phi[2:, :] - phi[:-2, :]) / (2 * dz)
    Hz[0, :] = -(phi[1, :] - phi[0, :]) / dz
    Hz[-1, :] = -(phi[-1, :] - phi[-2, :]) / dz
    return X, Z, mu, Hx, Hz


@app.function(gpu="T4", timeout=3600, volumes={VOLUME_PATH: volume}, max_containers=4)
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
    sof: bool = True,
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

    # ---------- FEM reference ----------
    Xg, Zg, MUg, Hx_fem, Hz_fem = fem_reference(NGRID)
    Xg_t = tc.tensor(Xg, dtype=tc.float64, device=device)
    Zg_t = tc.tensor(Zg, dtype=tc.float64, device=device)
    Hx_fem_t = tc.tensor(Hx_fem, dtype=tc.float64, device=device)
    Hz_fem_t = tc.tensor(Hz_fem, dtype=tc.float64, device=device)

    # ---------- Rede bruta ----------
    if kind == "pinn":
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
        from mixfunn import Mix2Funn
        net = Mix2Funn(
            n_in=2, n_out=2, n_layers=n_layers, n_hidden=width,
            use_softmax=True, T_init=T_INIT_MIX, T_final=T_FINAL_MIX,
            n_anneal_epochs=adam_iters,
            second_order_function=sof,
            dropout=0.0, init_alpha_std=0.1,
        ).to(device)

    n_params = int(sum(p.numel() for p in net.parameters() if p.requires_grad))

    # Distancia ao bordo do quadrado [0,1]^2 e fator d0 com gradiente fechado
    def d0_fn(x, z):
        # min distance to four walls
        d = tc.minimum(tc.minimum(x, 1.0 - x), tc.minimum(z, 1.0 - z))
        return 1.0 - tc.exp(-10.0 * d)

    def forward(xz):
        out = net(xz)
        f_x = out[:, 0:1]
        f_z = out[:, 1:2]
        d0 = d0_fn(xz[:, 0:1], xz[:, 1:2])
        Hx = f_x * d0
        Hz = f_z * d0 + 1.0
        return Hx, Hz

    def mu_fn(x, z):
        r = tc.sqrt((x - DISC_CX) ** 2 + (z - DISC_CZ) ** 2)
        s = tc.sigmoid(SIGMOID_ALPHA * (DISC_R - r))
        return MU_OUT + (MU_DISC - MU_OUT) * s

    # ---------- Pontos de colocacao (LHS) ----------
    N_INT = 8000

    def latin_hypercube(n, d, gen):
        # LHS simples [0,1]^d
        cut = tc.linspace(0.0, 1.0, n + 1, device=device)
        u = tc.rand(n, d, device=device, generator=gen)
        a = cut[:-1].unsqueeze(1)
        b = cut[1:].unsqueeze(1)
        rdraw = a + u * (b - a)
        out = tc.zeros_like(rdraw)
        for j in range(d):
            perm = tc.randperm(n, device=device, generator=gen)
            out[:, j] = rdraw[perm, j]
        return out

    g = tc.Generator(device=device).manual_seed(seed)
    xz_pool = latin_hypercube(N_INT, 2, g)
    x_pool = xz_pool[:, 0:1] * (X_MAX - X_MIN) + X_MIN
    z_pool = xz_pool[:, 1:2] * (Z_MAX - Z_MIN) + Z_MIN

    def grad(out, inp):
        return tc.autograd.grad(out, inp, grad_outputs=tc.ones_like(out), create_graph=True)[0]

    def pde_loss():
        x_int = x_pool.detach().clone().requires_grad_(True)
        z_int = z_pool.detach().clone().requires_grad_(True)
        xz_int = tc.cat([x_int, z_int], dim=1)
        Hx, Hz = forward(xz_int)

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
            Hx_v, Hz_v = forward(xz_v)
            Hx_v = Hx_v.reshape(NGRID, NGRID); Hz_v = Hz_v.reshape(NGRID, NGRID)
            num_x = ((Hx_v - Hx_fem_t) ** 2).sum()
            den_x = (Hx_fem_t ** 2).sum().clamp_min(1e-20)
            num_z = ((Hz_v - Hz_fem_t) ** 2).sum()
            den_z = (Hz_fem_t ** 2).sum().clamp_min(1e-20)
            l2x = float(tc.sqrt(num_x / den_x))
            l2z = float(tc.sqrt(num_z / den_z))
        net.train()
        return l2x, l2z

    # ============= Fase 1: Adam =============
    print(f"[{label}] phase=adam adam_iters={adam_iters} lr={lr}", flush=True)
    opt = tc.optim.Adam(net.parameters(), lr=lr)
    sched = tc.optim.lr_scheduler.StepLR(opt, step_size=max(1, adam_iters // 4), gamma=0.5)

    loss_curl_hist, loss_div_hist, loss_total_hist = [], [], []
    l2_curve, epochs_log = [], []
    t0 = time.perf_counter()
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

    # ============= Fase 2: L-BFGS =============
    print(f"[{label}] phase=lbfgs outer={lbfgs_outer} inner={lbfgs_inner}", flush=True)
    t_lbfgs_start = time.perf_counter()

    opt_lbfgs = tc.optim.LBFGS(
        net.parameters(),
        max_iter=lbfgs_inner,
        history_size=50,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
    )

    state = {"L": None, "Lc": None, "Ld": None}

    def closure():
        opt_lbfgs.zero_grad()
        loss, L_curl, L_div = pde_loss()
        loss.backward()
        state["L"] = loss; state["Lc"] = L_curl; state["Ld"] = L_div
        return loss

    for k in range(lbfgs_outer):
        try:
            opt_lbfgs.step(closure)
        except RuntimeError as e:
            print(f"[{label}] LBFGS RuntimeError at k={k}: {e}", flush=True)
            break
        if k % max(1, lbfgs_outer // 20) == 0:
            l2x, l2z = eval_l2()
            ep_log = adam_iters + (k + 1) * lbfgs_inner
            l2_curve.append([l2x, l2z])
            loss_curl_hist.append(float(state["Lc"].item()))
            loss_div_hist.append(float(state["Ld"].item()))
            loss_total_hist.append(float(state["L"].item()))
            epochs_log.append(ep_log)
            print(f"[{label}] lbfgs k={k} L={state['L'].item():.3e} "
                  f"L_curl={state['Lc'].item():.3e} L_div={state['Ld'].item():.3e} "
                  f"L2x={l2x:.3e} L2z={l2z:.3e}", flush=True)

    wall_lbfgs = time.perf_counter() - t_lbfgs_start
    wall = time.perf_counter() - t0

    # ============= Final eval =============
    net.eval()
    with tc.no_grad():
        xz_v = tc.stack([Xg_t.reshape(-1), Zg_t.reshape(-1)], dim=1)
        Hx_pred_t, Hz_pred_t = forward(xz_v)
        Hx_pred_arr = Hx_pred_t.reshape(NGRID, NGRID).cpu().numpy()
        Hz_pred_arr = Hz_pred_t.reshape(NGRID, NGRID).cpu().numpy()
        Hx_fem_arr = Hx_fem_t.cpu().numpy()
        Hz_fem_arr = Hz_fem_t.cpu().numpy()
        num_x = ((Hx_pred_arr - Hx_fem_arr) ** 2).sum()
        den_x = (Hx_fem_arr ** 2).sum() + 1e-20
        num_z = ((Hz_pred_arr - Hz_fem_arr) ** 2).sum()
        den_z = (Hz_fem_arr ** 2).sum() + 1e-20
        l2_hx = float((num_x / den_x) ** 0.5)
        l2_hz = float((num_z / den_z) ** 0.5)
        l2_tot = float((l2_hx ** 2 + l2_hz ** 2) ** 0.5)

    print(f"[{label}] DONE kind={kind} {n_layers}x{width} "
          f"L2_Hx={l2_hx:.3e} L2_Hz={l2_hz:.3e} L2_tot={l2_tot:.3e} "
          f"wall_total={wall:.1f}s wall_adam={wall_adam:.1f}s wall_lbfgs={wall_lbfgs:.1f}s "
          f"n_params={n_params}", flush=True)

    out_record = {
        "label": label, "kind": kind, "n_layers": n_layers, "width": width,
        "mode": "nsup_lift", "seed": seed, "lr": lr,
        "adam_iters": adam_iters, "lbfgs_outer": lbfgs_outer, "lbfgs_inner": lbfgs_inner,
        "sof": sof,
        "n_params": n_params,
        "l2_hx": l2_hx, "l2_hz": l2_hz, "l2_tot": l2_tot,
        "wall_clock": wall, "wall_adam": wall_adam, "wall_lbfgs": wall_lbfgs,
        "epochs_log": epochs_log,
        "loss_curl": loss_curl_hist, "loss_div": loss_div_hist, "loss_total": loss_total_hist,
        "l2_curve": l2_curve,
    }

    try:
        out_dir = Path(VOLUME_PATH) / EXPERIMENT / "by_label"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{label}.json").write_text(json.dumps(out_record))
        np.savez(
            out_dir / f"{label}_pred.npz",
            x=Xg, z=Zg, mu_r=MUg,
            Hx_fem=Hx_fem_arr, Hz_fem=Hz_fem_arr,
            Hx_pred=Hx_pred_arr, Hz_pred=Hz_pred_arr,
        )
        volume.commit()
    except Exception as e:
        print(f"[{label}] WARN commit: {e}", flush=True)

    return out_record


@app.local_entrypoint()
def main():
    # Dois jobs: PINN raissi-style 6x64 + MixFunn 3x3 sof
    cfg = [
        ("raissi_lift_6x64", "pinn", 6, 64, 21, 1e-3, 5000, 100, 50, False),
        ("mix_3x3_sof_lift", "mix",  3,  3, 21, 1e-3, 5000, 100, 50, True),
    ]
    print(f"[main] launching {len(cfg)} jobs Modal T4 v23 (Adam->LBFGS with lift)...", flush=True)
    t0 = time.perf_counter()
    results = list(train_one.starmap(cfg))
    wall = time.perf_counter() - t0
    print(f"[main] {len(results)} done in {wall:.1f}s", flush=True)
    for r in results:
        print(f"  {r['label']:30s} L2_Hx={r['l2_hx']:.3e} L2_Hz={r['l2_hz']:.3e} "
              f"wall={r['wall_clock']:.0f}s", flush=True)
    Path("/tmp/baldan_v23_results.json").write_text(json.dumps(results, indent=2))
