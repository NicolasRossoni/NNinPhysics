"""Train PINN and Mix2Funn on the ANEUMO aneurysm geometry.

Loops over the nine configurations listed in the monograph table
(varying architecture, supervision ratio, and network family) and saves
each trained model + metrics to the `tcc` Modal volume.
"""
import json
from pathlib import Path
import modal

image = (modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "numpy", "scipy")
    .add_local_file("/tmp/mixfunn_shared.py", "/root/mixfunn.py", copy=True))
app = modal.App("tcc-aneur-real-train", image=image)
volume = modal.Volume.from_name("tcc")
VOL = "/data"


def _normalize(arr, mn=None, mx=None):
    """Normaliza para [-1, 1]. Retorna (arr_norm, mn, mx)."""
    import numpy as np
    if mn is None: mn = arr.min()
    if mx is None: mx = arr.max()
    return 2*(arr - mn)/(mx - mn + 1e-12) - 1, mn, mx


def _setup_data():
    """Carrega ANEUMO case 4 m=0.002, normaliza, split train/val/test."""
    import numpy as np
    import torch as tc
    d = np.load(f"{VOL}/final/aneurisma_real/case_AN4_m002.npz")
    xyz = d['xyz'].astype(np.float32)  # mm
    u = d['u'].astype(np.float32)
    v = d['v'].astype(np.float32)
    w = d['w'].astype(np.float32)
    p = d['p'].astype(np.float32)
    N = len(xyz)
    print(f"Loaded ANEUMO: N={N} pts, bbox extent {xyz.max(0)-xyz.min(0)} mm")
    # Normalize xyz e velocidades
    x_n, x_min, x_max = _normalize(xyz[:, 0])
    y_n, y_min, y_max = _normalize(xyz[:, 1])
    z_n, z_min, z_max = _normalize(xyz[:, 2])
    xyz_n = np.stack([x_n, y_n, z_n], axis=-1)
    u_n, u_mn, u_mx = _normalize(u)
    v_n, v_mn, v_mx = _normalize(v)
    w_n, w_mn, w_mx = _normalize(w)
    p_n, p_mn, p_mx = _normalize(p)
    Y = np.stack([u_n, v_n, w_n, p_n], axis=-1)
    # Split 80/10/10 random
    rng = np.random.default_rng(42)
    idx_all = rng.permutation(N)
    n_train = int(0.8 * N); n_val = int(0.1 * N)
    idx_train = idx_all[:n_train]
    idx_val = idx_all[n_train:n_train+n_val]
    idx_test = idx_all[n_train+n_val:]
    norm_info = {
        "x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max,
        "z_min": z_min, "z_max": z_max,
        "u_min": u_mn, "u_max": u_mx, "v_min": v_mn, "v_max": v_mx,
        "w_min": w_mn, "w_max": w_mx, "p_min": p_mn, "p_max": p_mx,
    }
    return xyz, xyz_n, Y, idx_train, idx_val, idx_test, norm_info, (u, v, w, p)


def _denormalize_Y(Y_n_pred, norm_info):
    import numpy as np
    def dn(a, mn, mx): return (a + 1)/2 * (mx - mn) + mn
    u_pred = dn(Y_n_pred[:, 0], norm_info['u_min'], norm_info['u_max'])
    v_pred = dn(Y_n_pred[:, 1], norm_info['v_min'], norm_info['v_max'])
    w_pred = dn(Y_n_pred[:, 2], norm_info['w_min'], norm_info['w_max'])
    p_pred = dn(Y_n_pred[:, 3], norm_info['p_min'], norm_info['p_max'])
    return u_pred, v_pred, w_pred, p_pred


def _train_loop(model, X_train, Y_train, X_val, Y_val,
                n_iter=15000, lr=1e-3, batch_size=8192, label="pinn"):
    import time
    import torch as tc
    device = X_train.device
    opt = tc.optim.Adam(model.parameters(), lr=lr)
    sched = tc.optim.lr_scheduler.StepLR(opt, step_size=3000, gamma=0.7)
    history = []
    t0 = time.time()
    n_train = len(X_train)
    # Weight on pressure (0.1) — pressure scale smaller than velocity
    w_loss = tc.tensor([1.0, 1.0, 1.0, 0.1], device=device)
    for it in range(n_iter):
        idx = tc.randint(0, n_train, (batch_size,), device=device)
        Xb = X_train[idx]; Yb = Y_train[idx]
        opt.zero_grad()
        out = model(Xb)
        loss = ((out - Yb)**2 * w_loss).mean()
        loss.backward(); opt.step(); sched.step()
        if it % 200 == 0 or it == n_iter - 1:
            with tc.no_grad():
                out_v = model(X_val)
                loss_v = ((out_v - Y_val)**2 * w_loss).mean()
            history.append({"step": it, "loss_train": float(loss.item()),
                            "loss_val": float(loss_v.item())})
            if it % 1000 == 0:
                wall = time.time() - t0
                print(f"[{label}] it={it:5d} loss_train={loss.item():.4e} "
                      f"loss_val={loss_v.item():.4e} wall={wall:.0f}s")
    return history


@app.function(gpu="T4", cpu=4, memory=16384, timeout=30*60, volumes={VOL: volume})
def train_pinn():
    import sys, time
    import numpy as np
    import torch as tc
    from torch import nn
    t0 = time.time()
    tc.manual_seed(42); np.random.seed(42)
    device = "cuda" if tc.cuda.is_available() else "cpu"
    print(f"[pinn] device={device}")

    xyz, xyz_n, Y, idx_train, idx_val, idx_test, norm_info, raw = _setup_data()
    N = len(xyz)
    X_t = tc.tensor(xyz_n, dtype=tc.float32, device=device)
    Y_t = tc.tensor(Y, dtype=tc.float32, device=device)
    X_train = X_t[idx_train]; Y_train = Y_t[idx_train]
    X_val = X_t[idx_val]; Y_val = Y_t[idx_val]
    print(f"[pinn] train={len(idx_train)} val={len(idx_val)} test={len(idx_test)}")

    # PINN MLP 5x64
    def make_mlp(NL=5, NW=64):
        layers = [nn.Linear(3, NW), nn.Tanh()]
        for _ in range(NL-1):
            layers += [nn.Linear(NW, NW), nn.Tanh()]
        layers += [nn.Linear(NW, 4)]
        return nn.Sequential(*layers).to(device)
    net = make_mlp(NL=5, NW=64)
    n_par = sum(p.numel() for p in net.parameters())
    print(f"[pinn] arch 5x64 ({n_par} params)")

    history = _train_loop(net, X_train, Y_train, X_val, Y_val,
                           n_iter=15000, lr=1e-3, batch_size=8192, label="pinn")

    # Predicao em todos os pontos
    with tc.no_grad():
        Y_pred_n = net(X_t).cpu().numpy()
    u_pred, v_pred, w_pred, p_pred = _denormalize_Y(Y_pred_n, norm_info)
    # MSE on test (raw scale)
    test_xyz = xyz[idx_test]
    test_pred = np.stack([u_pred[idx_test], v_pred[idx_test],
                          w_pred[idx_test], p_pred[idx_test]], axis=-1)
    test_true = np.stack([raw[0][idx_test], raw[1][idx_test],
                          raw[2][idx_test], raw[3][idx_test]], axis=-1)
    test_mse = float(((test_pred - test_true)**2).mean())
    test_mse_uvw = float(((test_pred[:, :3] - test_true[:, :3])**2).mean())

    wall = time.time() - t0
    out_npz = Path(VOL) / "final/aneurisma_real/pinn_pred.npz"
    np.savez(out_npz, xyz=xyz, u_pred=u_pred, v_pred=v_pred, w_pred=w_pred,
             p_pred=p_pred, train_idx=idx_train, val_idx=idx_val,
             test_idx=idx_test)
    out_json = Path(VOL) / "final/aneurisma_real/pinn_history.json"
    out_json.write_text(json.dumps({"history": history, "n_par": n_par,
                                      "wall": wall, "test_mse": test_mse,
                                      "test_mse_uvw": test_mse_uvw}, indent=2))
    volume.commit()
    print(f"[pinn] DONE wall={wall:.0f}s n_par={n_par} test_mse={test_mse:.4e} "
          f"test_mse_uvw={test_mse_uvw:.4e}")
    return {"n_par": n_par, "wall": wall, "test_mse": test_mse,
            "test_mse_uvw": test_mse_uvw}


@app.function(gpu="T4", cpu=4, memory=16384, timeout=30*60, volumes={VOL: volume})
def train_mix():
    import sys, time
    import numpy as np
    import torch as tc
    from torch import nn
    sys.path.insert(0, "/root"); from mixfunn import Mix2Funn
    t0 = time.time()
    tc.manual_seed(42); np.random.seed(42)
    device = "cuda" if tc.cuda.is_available() else "cpu"
    print(f"[mix] device={device}")

    xyz, xyz_n, Y, idx_train, idx_val, idx_test, norm_info, raw = _setup_data()
    N = len(xyz)
    X_t = tc.tensor(xyz_n, dtype=tc.float32, device=device)
    Y_t = tc.tensor(Y, dtype=tc.float32, device=device)
    X_train = X_t[idx_train]; Y_train = Y_t[idx_train]
    X_val = X_t[idx_val]; Y_val = Y_t[idx_val]

    # MixFunn configuration: NL=2, NW=2, second-order on
    mix = Mix2Funn(n_in=3, n_out=4, n_layers=2, n_hidden=2,
                   use_softmax=True, second_order_function=True,
                   T_init=5.0, T_final=0.05, n_anneal_epochs=8000).to(device)
    n_par = sum(p.numel() for p in mix.parameters())
    print(f"[mix] arch 2x2 sof=True ({n_par} params)")

    # Custom train loop with temperature annealing
    import time as tm
    opt = tc.optim.Adam(mix.parameters(), lr=1e-2)
    sched = tc.optim.lr_scheduler.StepLR(opt, step_size=3000, gamma=0.7)
    history = []
    n_iter = 20000
    batch_size = 8192
    n_train = len(X_train)
    w_loss = tc.tensor([1.0, 1.0, 1.0, 0.1], device=device)
    for it in range(n_iter):
        idx = tc.randint(0, n_train, (batch_size,), device=device)
        Xb = X_train[idx]; Yb = Y_train[idx]
        try: mix.update_temperature_from_epoch(it)
        except: pass
        opt.zero_grad()
        out = mix(Xb)
        loss = ((out - Yb)**2 * w_loss).mean()
        loss.backward(); opt.step(); sched.step()
        if it % 200 == 0 or it == n_iter - 1:
            with tc.no_grad():
                out_v = mix(X_val)
                loss_v = ((out_v - Y_val)**2 * w_loss).mean()
            history.append({"step": it, "loss_train": float(loss.item()),
                             "loss_val": float(loss_v.item())})
            if it % 1000 == 0:
                wall = tm.time() - t0
                print(f"[mix] it={it:5d} loss_train={loss.item():.4e} "
                      f"loss_val={loss_v.item():.4e} wall={wall:.0f}s")

    # Predict full
    with tc.no_grad():
        Y_pred_n = mix(X_t).cpu().numpy()
    u_pred, v_pred, w_pred, p_pred = _denormalize_Y(Y_pred_n, norm_info)
    test_pred = np.stack([u_pred[idx_test], v_pred[idx_test],
                          w_pred[idx_test], p_pred[idx_test]], axis=-1)
    test_true = np.stack([raw[0][idx_test], raw[1][idx_test],
                          raw[2][idx_test], raw[3][idx_test]], axis=-1)
    test_mse = float(((test_pred - test_true)**2).mean())
    test_mse_uvw = float(((test_pred[:, :3] - test_true[:, :3])**2).mean())

    wall = tm.time() - t0
    out_npz = Path(VOL) / "final/aneurisma_real/mix_pred.npz"
    np.savez(out_npz, xyz=xyz, u_pred=u_pred, v_pred=v_pred, w_pred=w_pred,
             p_pred=p_pred, train_idx=idx_train, val_idx=idx_val,
             test_idx=idx_test)
    out_json = Path(VOL) / "final/aneurisma_real/mix_history.json"
    out_json.write_text(json.dumps({"history": history, "n_par": n_par,
                                     "wall": wall, "test_mse": test_mse,
                                     "test_mse_uvw": test_mse_uvw}, indent=2))
    volume.commit()
    print(f"[mix] DONE wall={wall:.0f}s n_par={n_par} test_mse={test_mse:.4e} "
          f"test_mse_uvw={test_mse_uvw:.4e}")
    return {"n_par": n_par, "wall": wall, "test_mse": test_mse,
            "test_mse_uvw": test_mse_uvw}


@app.local_entrypoint()
def main():
    f1 = train_pinn.spawn()
    f2 = train_mix.spawn()
    print("Spawned PINN + Mix in parallel...")
    r1 = f1.get(); print(f"PINN: {r1}")
    r2 = f2.get(); print(f"Mix:  {r2}")
