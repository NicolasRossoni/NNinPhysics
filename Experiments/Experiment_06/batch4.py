"""Schrodinger sprint, batch 4: matched-iters PINN for the best Mix2Funn.

Adds the missing PINN iteration count needed to pair with the best
Mix2Funn run for the figure in the monograph.
"""

import json
import math
import time
from pathlib import Path

import modal

PARENT = Path(__file__).resolve().parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "numpy", "scipy")
    .add_local_file(str(PARENT / "mixfunn.py"), "/root/mixfunn.py", copy=True)
)
app = modal.App("tcc-final-schrod-v22-b4", image=image)
volume = modal.Volume.from_name("tcc", create_if_missing=True)

VOLUME_PATH = "/data"
EXPERIMENT = "final/schrod_v22"

X_MIN, X_MAX = -5.0, 5.0
T_MIN, T_MAX = 0.0, math.pi / 2.0

N_INT = 4000
N_IC = 1000
N_BC = 1000
LAMBDA_IC = 100.0
LAMBDA_BC = 100.0

T_INIT_MIX = 5.0
T_FINAL_MIX = 0.05

EPOCHS_LOG = 500
NX_EVAL = 200
NT_EVAL = 100


@app.function(
    gpu="T4",
    timeout=1700,
    volumes={VOLUME_PATH: volume},
    max_containers=4,
    retries=0,
)
def train_one(label, kind, n_layers, width, seed, lr, iterations):
    import numpy as np
    import torch as tc
    from torch import nn
    import sys
    sys.path.insert(0, "/root")

    tc.set_default_dtype(tc.float32)
    tc.manual_seed(seed)
    np.random.seed(seed)
    device = tc.device("cuda" if tc.cuda.is_available() else "cpu")

    ref_path = Path(VOLUME_PATH) / EXPERIMENT / "ref.npz"
    ref = np.load(ref_path)
    u_ref_grid = ref["u_ref"]
    v_ref_grid = ref["v_ref"]
    abs_psi_ref = ref["abs_psi_ref"]
    x_eval_np = ref["x"]
    t_ref_eval = ref["t"]

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
            n_in=2, n_out=2,
            n_layers=n_layers, n_hidden=width,
            use_softmax=True, T_init=T_INIT_MIX, T_final=T_FINAL_MIX,
            n_anneal_epochs=iterations,
            second_order_function=True, dropout=0.0, init_alpha_std=0.1,
        ).to(device)

    n_params = int(sum(p.numel() for p in net.parameters() if p.requires_grad))
    opt = tc.optim.Adam(net.parameters(), lr=lr)
    sched = tc.optim.lr_scheduler.StepLR(opt, step_size=max(1, iterations // 4), gamma=0.5)

    g_cpu = tc.Generator().manual_seed(seed)
    x_int_pool = (tc.rand(N_INT, 1, generator=g_cpu) * (X_MAX - X_MIN) + X_MIN).to(device)
    t_int_pool = (tc.rand(N_INT, 1, generator=g_cpu) * (T_MAX - T_MIN) + T_MIN).to(device)
    x_ic = (tc.rand(N_IC, 1, generator=g_cpu) * (X_MAX - X_MIN) + X_MIN).to(device)
    t_ic = tc.zeros(N_IC, 1, device=device)
    u_ic_ref = 2.0 / tc.cosh(x_ic)
    v_ic_ref = tc.zeros_like(u_ic_ref)
    t_bc = (tc.rand(N_BC, 1, generator=g_cpu) * (T_MAX - T_MIN) + T_MIN).to(device)
    x_bc_left = tc.full((N_BC, 1), X_MIN, device=device)
    x_bc_right = tc.full((N_BC, 1), X_MAX, device=device)

    XEV, TEV = np.meshgrid(x_eval_np, t_ref_eval, indexing="ij")
    xy_full = tc.from_numpy(np.stack([XEV.ravel(), TEV.ravel()], axis=1).astype(np.float32)).to(device)

    loss_curve, l2_curve, epochs_log = [], [], []
    t0 = time.perf_counter()

    def forward_uv(x, t):
        xy = tc.cat([x, t], dim=1)
        out = net(xy)
        return out[:, 0:1], out[:, 1:2]

    print(f"[{label}] start: n_params={n_params} kind={kind} iters={iterations}", flush=True)

    def grad(out, inp):
        return tc.autograd.grad(out, inp, grad_outputs=tc.ones_like(out), create_graph=True)[0]

    SAVE_EVERY = 2000

    def save_now(ep_now, final=False):
        with tc.no_grad():
            out_eval = net(xy_full)
            up_e = out_eval[:, 0:1]; vp_e = out_eval[:, 1:2]
            u_p = up_e.cpu().numpy().reshape(NX_EVAL, NT_EVAL)
            v_p = vp_e.cpu().numpy().reshape(NX_EVAL, NT_EVAL)
            abs_p = np.sqrt(u_p ** 2 + v_p ** 2)
            num = ((abs_p - abs_psi_ref) ** 2).sum()
            den = (abs_psi_ref ** 2).sum() + 1e-12
            l2v = float(np.sqrt(num / den))
            l2u_ = float(np.linalg.norm(u_p - u_ref_grid) / (np.linalg.norm(u_ref_grid) + 1e-12))
            l2v_ = float(np.linalg.norm(v_p - v_ref_grid) / (np.linalg.norm(v_ref_grid) + 1e-12))
        record = {
            "label": label, "kind": kind, "partial": (not final), "ep_at_save": ep_now,
            "n_layers": n_layers, "width": width, "mode": "nsup",
            "seed": seed, "lr": lr, "iterations": iterations,
            "lambda_ic": LAMBDA_IC, "lambda_bc": LAMBDA_BC,
            "n_params": n_params,
            "l2_val": l2v, "l2_u": l2u_, "l2_v": l2v_,
            "final_loss": float(loss.item()), "wall_clock": time.perf_counter() - t0,
            "epochs_log": epochs_log, "loss_curve": loss_curve, "l2_curve": l2_curve,
        }
        out_dir = Path(VOLUME_PATH) / EXPERIMENT / "by_label"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{label}.json").write_text(json.dumps(record))
        np.savez_compressed(
            out_dir / f"{label}_pred.npz",
            u_pred=u_p.astype(np.float32), v_pred=v_p.astype(np.float32),
            abs_psi_pred=abs_p.astype(np.float32),
            u_ref=u_ref_grid.astype(np.float32), v_ref=v_ref_grid.astype(np.float32),
            abs_psi_ref=abs_psi_ref.astype(np.float32),
            x=x_eval_np.astype(np.float32), t=t_ref_eval.astype(np.float32),
        )
        volume.commit()
        return record, u_p, v_p, abs_p, l2v, l2u_, l2v_

    for ep in range(iterations + 1):
        if kind == "mix":
            net.update_temperature_from_epoch(ep)

        x_int = x_int_pool.detach().clone().requires_grad_(True)
        t_int = t_int_pool.detach().clone().requires_grad_(True)
        u, v = forward_uv(x_int, t_int)

        u_x = grad(u, x_int); u_t = grad(u, t_int); u_xx = grad(u_x, x_int)
        v_x = grad(v, x_int); v_t = grad(v, t_int); v_xx = grad(v_x, x_int)
        mag2 = u * u + v * v
        f_u = -v_t + 0.5 * u_xx + mag2 * u
        f_v =  u_t + 0.5 * v_xx + mag2 * v
        loss_pde = tc.mean(f_u ** 2) + tc.mean(f_v ** 2)

        u_ic_p, v_ic_p = forward_uv(x_ic, t_ic)
        loss_ic = tc.mean((u_ic_p - u_ic_ref) ** 2) + tc.mean((v_ic_p - v_ic_ref) ** 2)

        u_bcl, v_bcl = forward_uv(x_bc_left, t_bc)
        u_bcr, v_bcr = forward_uv(x_bc_right, t_bc)
        loss_bc = (tc.mean(u_bcl ** 2) + tc.mean(v_bcl ** 2)
                   + tc.mean(u_bcr ** 2) + tc.mean(v_bcr ** 2))

        loss = loss_pde + LAMBDA_IC * loss_ic + LAMBDA_BC * loss_bc
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()

        if ep % EPOCHS_LOG == 0:
            with tc.no_grad():
                out_eval = net(xy_full)
                up_e = out_eval[:, 0:1]; vp_e = out_eval[:, 1:2]
                u_g = up_e.cpu().numpy().reshape(NX_EVAL, NT_EVAL)
                v_g = vp_e.cpu().numpy().reshape(NX_EVAL, NT_EVAL)
                abs_p_ = np.sqrt(u_g ** 2 + v_g ** 2)
                num = ((abs_p_ - abs_psi_ref) ** 2).sum()
                den = (abs_psi_ref ** 2).sum() + 1e-12
                l2_now = float(np.sqrt(num / den))
            loss_curve.append(float(loss.item()))
            l2_curve.append(l2_now)
            epochs_log.append(ep)
            print(f"[{label}] ep={ep} L_pde={float(loss_pde.item()):.2e} L2|psi|={l2_now:.3e} t={time.perf_counter()-t0:.0f}s", flush=True)

        if ep > 0 and ep % SAVE_EVERY == 0:
            save_now(ep, final=False)

    save_now(iterations, final=True)
    wall = time.perf_counter() - t0
    final_loss = float(loss.item())

    # reload for return record
    record = json.load(open(Path(VOLUME_PATH) / EXPERIMENT / "by_label" / f"{label}.json"))
    print(f"[{label}] DONE L2|psi|={record['l2_val']:.3e} wall={wall:.1f}s n={n_params}", flush=True)
    return record


@app.function(timeout=600, volumes={VOLUME_PATH: volume})
def save_results(results, tag):
    out = Path(VOLUME_PATH) / EXPERIMENT
    out.mkdir(parents=True, exist_ok=True)
    (out / f"results_{tag}.json").write_text(json.dumps(results, indent=2))
    volume.commit()
    return f"saved {len(results)} as results_{tag}.json"


@app.local_entrypoint()
def main():
    LR_PINN = 1e-3
    LR_MIX = 1e-2
    configs = [
        ("pinn_8x100_nsup_10k", "pinn", 8, 100, 21, LR_PINN, 10000),
        ("mix_3x6_sof_nsup_15k", "mix", 3, 6, 21, LR_MIX, 15000),
    ]
    print(f"[main] Batch 4: {len(configs)} configs", flush=True)
    t0 = time.perf_counter()
    results = list(train_one.starmap(configs))
    wall = time.perf_counter() - t0
    print(f"[main] {len(results)} finished in {wall:.1f}s", flush=True)
    print(save_results.remote(results, "b4"), flush=True)
    for r in sorted(results, key=lambda r: r["l2_val"]):
        print(f"  {r['label']}: L2|psi|={r['l2_val']:.3e} n={r['n_params']} wall={r['wall_clock']:.0f}s", flush=True)
