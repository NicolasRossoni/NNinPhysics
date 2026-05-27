"""Modal sprint v22: STRICTLY NON-SUPERVISED PINN vs MixFunn for Burgers viscoso.

EDP: u_t + u u_x = (0.01/pi) u_xx, x in [-1, 1], t in [0, 1]
CI:  u(x, 0) = -sin(pi x)
BC:  u(-1, t) = u(1, t) = 0

NSUP-only: loss = L_pde + lambda_ic * L_ic + lambda_bc * L_bc.
Reference: pseudo-spectral Fourier (N=256) + RK4 (dt=1e-4). USED ONLY POST-TRAIN
to compute L^2 for reporting. ABSOLUTELY NO supervised term sneaks into the loss.

Configs are passed as keyword dicts; entrypoints in this file dispatch batches.
"""

from __future__ import annotations

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
app = modal.App("tcc-burgers-v22", image=image)
volume = modal.Volume.from_name("tcc", create_if_missing=True)

VOLUME_PATH = "/data"
EXPERIMENT = "final/burgers_v22"

X_MIN, X_MAX = -1.0, 1.0
T_MIN, T_MAX = 0.0, 1.0
NU = 0.01 / 3.141592653589793

# Output grid for L2 evaluation
NX_EVAL = 200
NT_EVAL = 100

# Spectral reference resolution
N_SPECTRAL = 256
DT_SPECTRAL = 1e-4


def _spectral_reference(N: int = N_SPECTRAL, dt: float = DT_SPECTRAL):
    import numpy as np

    L = X_MAX - X_MIN
    x = np.linspace(X_MIN, X_MAX, N, endpoint=False)
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=L / N)
    ik = 1j * k
    k2 = -(k ** 2)

    def rhs(u):
        uhat = np.fft.fft(u)
        ux = np.real(np.fft.ifft(ik * uhat))
        uxx = np.real(np.fft.ifft(k2 * uhat))
        return -u * ux + NU * uxx

    nt = int(round((T_MAX - T_MIN) / dt)) + 1
    t_grid = np.linspace(T_MIN, T_MAX, nt)
    u = np.empty((nt, N), dtype=np.float64)
    u[0] = -np.sin(np.pi * x)
    cur = u[0].copy()
    for n in range(nt - 1):
        k1 = rhs(cur)
        k2_ = rhs(cur + 0.5 * dt * k1)
        k3 = rhs(cur + 0.5 * dt * k2_)
        k4 = rhs(cur + dt * k3)
        cur = cur + (dt / 6.0) * (k1 + 2 * k2_ + 2 * k3 + k4)
        u[n + 1] = cur
    return x, t_grid, u


def _build_reference_grid():
    import numpy as np
    from scipy.interpolate import RegularGridInterpolator

    x_sp, t_sp, u_sp = _spectral_reference(N_SPECTRAL, DT_SPECTRAL)
    x_ext = np.concatenate([x_sp, [X_MAX]])
    u_ext = np.concatenate([u_sp, u_sp[:, 0:1]], axis=1)
    interp = RegularGridInterpolator((t_sp, x_ext), u_ext, bounds_error=False, fill_value=0.0)
    x_eval = np.linspace(X_MIN, X_MAX, NX_EVAL)
    t_eval = np.linspace(T_MIN, T_MAX, NT_EVAL)
    XE, TE = np.meshgrid(x_eval, t_eval, indexing="ij")
    pts = np.stack([TE.ravel(), XE.ravel()], axis=1)
    u_ref = interp(pts).reshape(NX_EVAL, NT_EVAL)
    return x_eval, t_eval, u_ref


def _lhs(n: int, d: int, low, high, rng):
    import numpy as np
    cut = np.linspace(0.0, 1.0, n + 1)
    a = cut[:n]
    b = cut[1:]
    u = rng.random((n, d))
    pts = a[:, None] + u * (b - a)[:, None]
    for j in range(d):
        rng.shuffle(pts[:, j])
    return low + pts * (high - low)


@app.function(gpu="T4", timeout=2400, volumes={VOLUME_PATH: volume}, max_containers=8)
def train_one(
    label: str,
    kind: str,        # "pinn" or "mix"
    NL: int,
    NW: int,
    sof: bool,
    iters: int,
    lr: float,
    lambda_ic: float,
    lambda_bc: float,
    n_col: int,
    n_ic: int,
    n_bc: int,
    resample_every: int,  # 0 = never resample (fixed LHS), else N
    sched_kind: str,      # "step" or "cosine"
) -> dict:
    import numpy as np
    import torch as tc
    from torch import nn
    import sys

    sys.path.insert(0, "/root")

    seed = 22
    tc.set_default_dtype(tc.float32)
    tc.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    device = tc.device("cuda" if tc.cuda.is_available() else "cpu")

    if kind == "pinn":
        layers = [2] + [NW] * NL + [1]
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
    elif kind == "mix":
        from mixfunn import Mix2Funn
        net = Mix2Funn(
            n_in=2, n_out=1,
            n_layers=NL, n_hidden=NW,
            use_softmax=True, T_init=5.0, T_final=0.05,
            n_anneal_epochs=iters,
            second_order_function=bool(sof),
            dropout=0.0,
            init_alpha_std=0.1,
        ).to(device)
    else:
        raise ValueError(kind)

    n_params = int(sum(p.numel() for p in net.parameters() if p.requires_grad))

    opt = tc.optim.Adam(net.parameters(), lr=lr)
    if sched_kind == "cosine":
        sched = tc.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters, eta_min=lr * 0.01)
    else:
        sched_step = max(1, iters // 4)
        sched = tc.optim.lr_scheduler.StepLR(opt, step_size=sched_step, gamma=0.5)

    # ---------- Reference grid (cached on shared volume) ----------
    ref_path = Path(VOLUME_PATH) / EXPERIMENT / "reference.npz"
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    if ref_path.exists():
        try:
            d = np.load(ref_path)
            x_eval_np = d["x"].astype(np.float64)
            t_eval_np = d["t"].astype(np.float64)
            u_ref_np = d["u"].astype(np.float64)
        except Exception:
            x_eval_np, t_eval_np, u_ref_np = _build_reference_grid()
    else:
        x_eval_np, t_eval_np, u_ref_np = _build_reference_grid()
        try:
            np.savez(ref_path, x=x_eval_np, t=t_eval_np, u=u_ref_np)
            volume.commit()
        except Exception:
            pass

    XE, TE = np.meshgrid(x_eval_np, t_eval_np, indexing="ij")
    xt_eval = tc.tensor(np.stack([XE.ravel(), TE.ravel()], axis=1),
                         device=device, dtype=tc.float32)

    # ---------- NSUP training data ----------
    def _sample_colloc():
        colloc_np = _lhs(n_col, 2,
                         np.array([X_MIN, T_MIN]),
                         np.array([X_MAX, T_MAX]),
                         rng)
        return tc.tensor(colloc_np, device=device, dtype=tc.float32)

    colloc = _sample_colloc()

    x_ic = tc.linspace(X_MIN, X_MAX, n_ic, device=device).unsqueeze(1)
    t_ic = tc.zeros_like(x_ic)
    xt_ic = tc.cat([x_ic, t_ic], dim=1)
    u_ic = -tc.sin(tc.pi * x_ic)

    t_bc = tc.linspace(T_MIN, T_MAX, n_bc // 2, device=device).unsqueeze(1)
    x_bc_l = tc.full_like(t_bc, X_MIN)
    x_bc_r = tc.full_like(t_bc, X_MAX)
    xt_bc = tc.cat([tc.cat([x_bc_l, t_bc], dim=1),
                    tc.cat([x_bc_r, t_bc], dim=1)], dim=0)

    # ---------- Train ----------
    loss_curve = []
    l2_curve = []
    pde_curve = []
    ic_curve = []
    bc_curve = []
    log_every = max(1, iters // 100)
    t0 = time.perf_counter()

    for ep in range(iters + 1):
        opt.zero_grad()
        xf = colloc.detach().clone().requires_grad_(True)
        u = net(xf)
        g1 = tc.autograd.grad(u, xf, grad_outputs=tc.ones_like(u),
                              create_graph=True)[0]
        u_x = g1[:, 0:1]
        u_t = g1[:, 1:2]
        g2 = tc.autograd.grad(u_x, xf, grad_outputs=tc.ones_like(u_x),
                              create_graph=True)[0]
        u_xx = g2[:, 0:1]
        res = u_t + u * u_x - NU * u_xx
        loss_res = tc.mean(res ** 2)

        u_ic_pred = net(xt_ic)
        loss_ic = tc.mean((u_ic_pred - u_ic) ** 2)
        u_bc_pred = net(xt_bc)
        loss_bc = tc.mean(u_bc_pred ** 2)
        loss = loss_res + lambda_ic * loss_ic + lambda_bc * loss_bc

        loss.backward()
        opt.step()
        sched.step()
        if kind == "mix":
            net.update_temperature_from_epoch(ep)

        if resample_every > 0 and ep > 0 and ep % resample_every == 0:
            colloc = _sample_colloc()

        if ep % log_every == 0:
            with tc.no_grad():
                u_pred_eval = net(xt_eval).cpu().numpy().reshape(NX_EVAL, NT_EVAL)
                l2 = float(np.linalg.norm(u_pred_eval - u_ref_np) /
                           (np.linalg.norm(u_ref_np) + 1e-30))
            loss_curve.append(float(loss.item()))
            l2_curve.append(l2)
            pde_curve.append(float(loss_res.item()))
            ic_curve.append(float(loss_ic.item()))
            bc_curve.append(float(loss_bc.item()))

    wall = time.perf_counter() - t0

    net.eval()
    with tc.no_grad():
        u_pred_final = net(xt_eval).cpu().numpy().reshape(NX_EVAL, NT_EVAL)
    l2_final = float(np.linalg.norm(u_pred_final - u_ref_np) /
                     (np.linalg.norm(u_ref_np) + 1e-30))

    out_dir = Path(VOLUME_PATH) / EXPERIMENT / "by_label"
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "label": label,
        "kind": kind,
        "NL": NL,
        "NW": NW,
        "sof": bool(sof),
        "lr": float(lr),
        "iters": int(iters),
        "lambda_ic": float(lambda_ic),
        "lambda_bc": float(lambda_bc),
        "n_col": int(n_col),
        "n_ic": int(n_ic),
        "n_bc": int(n_bc),
        "resample_every": int(resample_every),
        "sched_kind": sched_kind,
        "n_params": n_params,
        "wall": float(wall),
        "l2_final": l2_final,
        "loss_curve": loss_curve,
        "l2_curve": l2_curve,
        "pde_curve": pde_curve,
        "ic_curve": ic_curve,
        "bc_curve": bc_curve,
    }
    (out_dir / f"{label}.json").write_text(json.dumps(meta))
    np.savez(
        out_dir / f"{label}_pred.npz",
        u_pred=u_pred_final.astype(np.float32),
        x=x_eval_np.astype(np.float32),
        t=t_eval_np.astype(np.float32),
        u_ref=u_ref_np.astype(np.float32),
    )
    volume.commit()

    print(f"[{label}] kind={kind} L={NL} W={NW} sof={sof} "
          f"L2={l2_final:.3e} wall={wall:.1f}s np={n_params} iters={iters}",
          flush=True)

    return meta


# Common defaults for a config
def cfg(label, kind, NL, NW, sof, iters, lr,
        lambda_ic=10.0, lambda_bc=10.0,
        n_col=8000, n_ic=200, n_bc=200,
        resample_every=0, sched_kind="step"):
    return (label, kind, NL, NW, sof, iters, lr,
            lambda_ic, lambda_bc, n_col, n_ic, n_bc, resample_every, sched_kind)


@app.local_entrypoint()
def batch1() -> None:
    """First batch: starting point per protocol."""
    configs = [
        cfg("pinn_6x64_b1", "pinn", 6, 64, False, 30000, 1e-3),
        cfg("pinn_4x64_b1", "pinn", 4, 64, False, 30000, 1e-3),
        cfg("mix_3x3_sofT_b1", "mix", 3, 3, True, 30000, 1e-2),
        cfg("mix_2x2_sofT_b1", "mix", 2, 2, True, 30000, 1e-2),
    ]
    _run(configs, "batch1")


@app.local_entrypoint()
def batch2() -> None:
    """Second batch: tuned per batch1 results (edit in place per protocol)."""
    configs = [
        # Boost PINN: more iters, cosine schedule, more colloc, higher IC weight
        cfg("pinn_6x64_b2_long", "pinn", 6, 64, False, 50000, 1e-3,
            lambda_ic=100.0, lambda_bc=100.0, n_col=10000, sched_kind="cosine"),
        cfg("pinn_4x64_b2_long", "pinn", 4, 64, False, 50000, 1e-3,
            lambda_ic=100.0, lambda_bc=100.0, n_col=10000, sched_kind="cosine"),
        # Boost Mix: more iters, resample collocation
        cfg("mix_3x3_sofT_b2_long", "mix", 3, 3, True, 50000, 1e-2,
            lambda_ic=10.0, lambda_bc=10.0, n_col=8000, resample_every=2000),
        cfg("mix_3x6_sofT_b2", "mix", 3, 6, True, 30000, 1e-2,
            lambda_ic=10.0, lambda_bc=10.0, n_col=8000),
    ]
    _run(configs, "batch2")


@app.local_entrypoint()
def batch3() -> None:
    """Final refinement after batch2."""
    configs = [
        cfg("pinn_6x64_final", "pinn", 6, 64, False, 60000, 1e-3,
            lambda_ic=100.0, lambda_bc=100.0, n_col=10000, sched_kind="cosine"),
        cfg("mix_3x3_sofT_final", "mix", 3, 3, True, 60000, 1e-2,
            lambda_ic=10.0, lambda_bc=10.0, n_col=8000, resample_every=2000),
    ]
    _run(configs, "batch3")


def _run(configs, tag):
    print(f"[{tag}] launching {len(configs)} jobs on Modal T4...", flush=True)
    t0 = time.perf_counter()
    results = []
    for r in train_one.starmap(configs, return_exceptions=True):
        if isinstance(r, BaseException):
            print(f"[{tag}] FAILED job: {type(r).__name__}: {r}", flush=True)
            continue
        results.append(r)
        print(f"[{tag}] progress {len(results)}/{len(configs)} "
              f"({r['label']} L2={r['l2_final']:.3e} wall={r['wall']:.1f}s)",
              flush=True)
    wall = time.perf_counter() - t0
    print(f"[{tag}] done {len(results)} jobs in {wall:.1f}s", flush=True)

    pinn_r = [r for r in results if r["kind"] == "pinn"]
    mix_r = [r for r in results if r["kind"] == "mix"]
    if pinn_r:
        best = min(pinn_r, key=lambda r: r["l2_final"])
        print(f"[{tag}] BEST PINN: {best['label']} L2={best['l2_final']:.3e} "
              f"np={best['n_params']} wall={best['wall']:.1f}s", flush=True)
    if mix_r:
        best = min(mix_r, key=lambda r: r["l2_final"])
        print(f"[{tag}] BEST MIX:  {best['label']} L2={best['l2_final']:.3e} "
              f"np={best['n_params']} wall={best['wall']:.1f}s", flush=True)
