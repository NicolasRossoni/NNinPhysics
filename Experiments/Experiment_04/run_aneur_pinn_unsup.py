"""Modal: PINN 3D NS UNSUPERVISIONADO no aneurisma ANEUMO.

Sem leak: u_bc, v_bc, w_bc, p NÃO usam dados ANEUMO no interior. Apenas:
  - BC inlet via xyz_inlet (anatomia conhecida — equivale a impor a velocidade
    medida no inlet pelo MRI/CT, BC física padrão)
  - BC wall (no-slip u=0)
  - Outlet pressure = 0 (referência)

Loss = PDE residual (NS 3D + continuidade) + soft penalty wall (u=0) +
       soft penalty inlet (u=u_aneumo[inlet])

A "verdade" ANEUMO só entra na AVALIAÇÃO (depois do treino).

ANEUMO npz keys: xyz (129693,3), u, v, w, p, sdf, xyz_inlet (716,3), simple_inlet (7,).
SDF é positivo no interior; SDF=0 no contorno da malha fluida.
"""
import json
import time
from pathlib import Path

import modal

PARENT = Path(__file__).resolve().parent
MIXFUNN_PATH = PARENT.parent / "final" / "ns_kovasznay" / "mixfunn.py"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "numpy")
    .add_local_file(str(MIXFUNN_PATH), "/root/mixfunn.py", copy=True)
)
app = modal.App("tcc-aneur-unsup", image=image)
volume = modal.Volume.from_name("tcc", create_if_missing=True)
VOLUME_PATH = "/data"
EXPERIMENT = "final/aneur_unsup"

ITER = 15000
SCHED_STEP = 3000
SCHED_GAMMA = 0.5
LR_PINN = 1e-3
LR_MIX = 1e-2
T_INIT_MIX = 5.0
T_FINAL_MIX = 0.05
N_INT_BATCH = 4000   # collocation points per iter
RHO = 1060.0         # density blood (kg/m^3)
MU = 0.0035          # viscosity blood (Pa.s)
# Re-scale: comprimento ~8mm, vel max ~0.77 m/s. Reynolds ~ 0.001 * 0.77 * 1060 / 0.0035 ~ 230
# Para PINN sem dimensionalizar: usar diretamente os valores ANEUMO em SI.

EPOCHS_LOG = 100


@app.function(gpu="T4", timeout=120 * 60, volumes={VOLUME_PATH: volume}, max_containers=4)
def train_one(
    label: str,
    kind: str,     # "pinn" or "mix"
    n_layers: int,
    width: int,
    sof: bool,
    seed: int,
    lr: float,
    iterations: int,
) -> dict:
    import sys
    sys.path.insert(0, "/root")
    import numpy as np
    import torch as tc
    from torch import nn

    tc.set_default_dtype(tc.float32)
    tc.manual_seed(seed); np.random.seed(seed)
    device = tc.device("cuda" if tc.cuda.is_available() else "cpu")

    # ---- Load ANEUMO data ----
    npz = np.load("/data/final/aneurisma_real/case_AN4_m002.npz")
    xyz = npz["xyz"].astype(np.float32)              # (N, 3) mm
    u_gt = npz["u"].astype(np.float32)
    v_gt = npz["v"].astype(np.float32)
    w_gt = npz["w"].astype(np.float32)
    p_gt = npz["p"].astype(np.float32)
    sdf = npz["sdf"].astype(np.float32)              # (N,) positive interior
    xyz_inlet = npz["xyz_inlet"].astype(np.float32)  # (716, 3)

    # Convert mm -> m for SI physics (ANEUMO usa mm)
    SCALE_LEN = 1e-3
    xyz_m = xyz * SCALE_LEN
    xyz_inlet_m = xyz_inlet * SCALE_LEN
    sdf_m = sdf * SCALE_LEN

    # Bounding box for normalization
    bbox_min = xyz_m.min(axis=0)
    bbox_max = xyz_m.max(axis=0)
    bbox_center = (bbox_min + bbox_max) / 2
    bbox_scale = (bbox_max - bbox_min).max()

    def normalize(xyz_in):
        return (xyz_in - bbox_center) / bbox_scale

    xyz_norm = normalize(xyz_m)
    xyz_inlet_norm = normalize(xyz_inlet_m)

    # u_inlet from ANEUMO (BC física conhecida, padrão PINN)
    # Find indices of inlet points in xyz
    inlet_idx = np.zeros(xyz_inlet.shape[0], dtype=int)
    for i, p_in in enumerate(xyz_inlet):
        dists = np.linalg.norm(xyz - p_in, axis=1)
        inlet_idx[i] = int(np.argmin(dists))
    u_inlet = u_gt[inlet_idx]
    v_inlet = v_gt[inlet_idx]
    w_inlet = w_gt[inlet_idx]

    # Wall points: pontos próximos da parede (SDF baixo positivo).
    # SDF tem mín=0 (na parede); pegar os 10% com menor SDF positivo.
    nonzero_sdf = sdf_m[sdf_m > 0]
    if len(nonzero_sdf) > 0:
        sdf_thresh = np.quantile(nonzero_sdf, 0.10)
    else:
        sdf_thresh = 1e-5
    wall_mask = sdf_m < sdf_thresh
    wall_idx = np.where(wall_mask)[0]
    if len(wall_idx) == 0:
        # Fallback: pegar os 5000 pontos com menor SDF
        wall_idx = np.argsort(sdf_m)[:5000]
    print(f"[{label}] wall pts: {len(wall_idx)}", flush=True)

    # Convert to torch
    xyz_norm_t = tc.from_numpy(xyz_norm).to(device)
    xyz_inlet_norm_t = tc.from_numpy(xyz_inlet_norm).to(device)
    u_inlet_t = tc.from_numpy(u_inlet).to(device).unsqueeze(1)
    v_inlet_t = tc.from_numpy(v_inlet).to(device).unsqueeze(1)
    w_inlet_t = tc.from_numpy(w_inlet).to(device).unsqueeze(1)
    sdf_t = tc.from_numpy(sdf_m).to(device).unsqueeze(1)
    u_gt_t = tc.from_numpy(u_gt).to(device).unsqueeze(1)
    v_gt_t = tc.from_numpy(v_gt).to(device).unsqueeze(1)
    w_gt_t = tc.from_numpy(w_gt).to(device).unsqueeze(1)
    p_gt_t = tc.from_numpy(p_gt).to(device).unsqueeze(1)

    N_total = xyz_norm.shape[0]
    wall_xyz_norm_t = tc.from_numpy(xyz_norm[wall_idx]).to(device)
    N_wall = wall_xyz_norm_t.shape[0]
    N_inlet = xyz_inlet_norm_t.shape[0]

    # Re-scale physics for normalized coordinates: derivatives in normalized space need scaling
    # We minimize PDE residual evaluated in normalized space, requires re-scale.
    # For simplicity: just use bbox_scale as length scale; let optimizer find pressure scale.
    L_PHYS = bbox_scale  # m

    # Network
    if kind == "pinn":
        layers = [3] + [width] * n_layers + [4]  # in: xyz (3), out: u, v, w, p (4)
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
            n_in=3, n_out=4, n_layers=n_layers, n_hidden=width,
            use_softmax=True, T_init=T_INIT_MIX, T_final=T_FINAL_MIX,
            n_anneal_epochs=iterations,
            second_order_function=sof,
            dropout=0.0, init_alpha_std=0.1,
        ).to(device)

    n_params = int(sum(p.numel() for p in net.parameters() if p.requires_grad))

    opt = tc.optim.Adam(net.parameters(), lr=lr)
    sched = tc.optim.lr_scheduler.StepLR(opt, step_size=SCHED_STEP, gamma=SCHED_GAMMA)

    # Losses log
    loss_total_log = []; loss_pde_log = []; loss_wall_log = []; loss_inlet_log = []
    l2_test_log = []
    epochs_log = []

    LAMBDA_WALL = 100.0
    LAMBDA_INLET = 100.0

    t0 = time.perf_counter()
    for ep in range(iterations + 1):
        if kind == "mix":
            net.update_temperature_from_epoch(ep)

        # Sample collocation points
        idx_col = tc.randint(0, N_total, (N_INT_BATCH,), device=device)
        xc = xyz_norm_t[idx_col].clone().requires_grad_(True)

        out = net(xc)
        u = out[:, 0:1]; v = out[:, 1:2]; w = out[:, 2:3]; p = out[:, 3:4]

        def grad(out_t, inp_t):
            return tc.autograd.grad(out_t, inp_t, grad_outputs=tc.ones_like(out_t), create_graph=True)[0]

        # First derivatives wrt normalized coords; chain rule: d/dx_phys = (1/L) d/dx_norm
        gu = grad(u, xc); u_x = gu[:, 0:1] / L_PHYS; u_y = gu[:, 1:2] / L_PHYS; u_z = gu[:, 2:3] / L_PHYS
        gv = grad(v, xc); v_x = gv[:, 0:1] / L_PHYS; v_y = gv[:, 1:2] / L_PHYS; v_z = gv[:, 2:3] / L_PHYS
        gw = grad(w, xc); w_x = gw[:, 0:1] / L_PHYS; w_y = gw[:, 1:2] / L_PHYS; w_z = gw[:, 2:3] / L_PHYS
        gp = grad(p, xc); p_x = gp[:, 0:1] / L_PHYS; p_y = gp[:, 1:2] / L_PHYS; p_z = gp[:, 2:3] / L_PHYS

        # Second derivatives for Laplacian
        gux = grad(u_x, xc); u_xx = gux[:, 0:1] / L_PHYS
        guy = grad(u_y, xc); u_yy = guy[:, 1:2] / L_PHYS
        guz = grad(u_z, xc); u_zz = guz[:, 2:3] / L_PHYS
        gvx = grad(v_x, xc); v_xx = gvx[:, 0:1] / L_PHYS
        gvy = grad(v_y, xc); v_yy = gvy[:, 1:2] / L_PHYS
        gvz = grad(v_z, xc); v_zz = gvz[:, 2:3] / L_PHYS
        gwx = grad(w_x, xc); w_xx = gwx[:, 0:1] / L_PHYS
        gwy = grad(w_y, xc); w_yy = gwy[:, 1:2] / L_PHYS
        gwz = grad(w_z, xc); w_zz = gwz[:, 2:3] / L_PHYS

        # NS 3D incompressível steady:
        # rho (u·grad u) = -grad p + mu lap u
        # div u = 0
        res_u = RHO * (u * u_x + v * u_y + w * u_z) + p_x - MU * (u_xx + u_yy + u_zz)
        res_v = RHO * (u * v_x + v * v_y + w * v_z) + p_y - MU * (v_xx + v_yy + v_zz)
        res_w = RHO * (u * w_x + v * w_y + w * w_z) + p_z - MU * (w_xx + w_yy + w_zz)
        res_div = u_x + v_y + w_z

        # Scale residuals (NS has dimension Pa/m ~ 1e3*1=1e3, div has dim 1/s)
        L_pde = (tc.mean(res_u ** 2) + tc.mean(res_v ** 2) + tc.mean(res_w ** 2)) * 1e-6
        L_div = tc.mean(res_div ** 2) * 1.0

        # Wall BC: u=v=w=0
        idx_w = tc.randint(0, N_wall, (min(N_INT_BATCH, N_wall),), device=device)
        xw = wall_xyz_norm_t[idx_w]
        out_w = net(xw)
        L_wall = tc.mean(out_w[:, 0] ** 2 + out_w[:, 1] ** 2 + out_w[:, 2] ** 2)

        # Inlet BC: u=ANEUMO[inlet]
        idx_i = tc.randint(0, N_inlet, (min(N_INT_BATCH, N_inlet),), device=device)
        xi = xyz_inlet_norm_t[idx_i]
        out_i = net(xi)
        L_inlet = (tc.mean((out_i[:, 0:1] - u_inlet_t[idx_i]) ** 2)
                   + tc.mean((out_i[:, 1:2] - v_inlet_t[idx_i]) ** 2)
                   + tc.mean((out_i[:, 2:3] - w_inlet_t[idx_i]) ** 2))

        loss = L_pde + L_div + LAMBDA_WALL * L_wall + LAMBDA_INLET * L_inlet

        opt.zero_grad(); loss.backward(); opt.step(); sched.step()

        if ep % EPOCHS_LOG == 0:
            with tc.no_grad():
                # Evaluate on full ANEUMO grid
                out_full = net(xyz_norm_t)
                u_pred = out_full[:, 0:1]; v_pred = out_full[:, 1:2]
                w_pred = out_full[:, 2:3]; p_pred = out_full[:, 3:4]
                # L2 relativo da velocidade (combinada)
                num = ((u_pred - u_gt_t) ** 2 + (v_pred - v_gt_t) ** 2 + (w_pred - w_gt_t) ** 2).sum()
                den = (u_gt_t ** 2 + v_gt_t ** 2 + w_gt_t ** 2).sum()
                l2_test = float(tc.sqrt(num / den))
            l2_test_log.append(l2_test)
            loss_total_log.append(float(loss.item()))
            loss_pde_log.append(float((L_pde + L_div).item()))
            loss_wall_log.append(float(L_wall.item()))
            loss_inlet_log.append(float(L_inlet.item()))
            epochs_log.append(ep)

    wall = time.perf_counter() - t0

    # Avaliação final
    with tc.no_grad():
        out_full = net(xyz_norm_t)
        u_pred = out_full[:, 0:1].cpu().numpy()
        v_pred = out_full[:, 1:2].cpu().numpy()
        w_pred = out_full[:, 2:3].cpu().numpy()
        p_pred = out_full[:, 3:4].cpu().numpy()
        num = ((u_pred - u_gt[:, None]) ** 2 + (v_pred - v_gt[:, None]) ** 2 + (w_pred - w_gt[:, None]) ** 2).sum()
        den = (u_gt ** 2 + v_gt ** 2 + w_gt ** 2).sum()
        l2_test = float(np.sqrt(num / den))

    print(f"[{label}] L2_test={l2_test:.3e} wall={wall:.1f}s n={n_params}", flush=True)

    record = {
        "label": label, "kind": kind, "n_layers": n_layers, "width": width,
        "sof": sof, "seed": seed, "iterations": iterations, "n_params": n_params,
        "l2_test_final": l2_test, "wall_clock": wall,
        "epochs_log": epochs_log,
        "loss_total": loss_total_log, "loss_pde": loss_pde_log,
        "loss_wall": loss_wall_log, "loss_inlet": loss_inlet_log,
        "l2_test_curve": l2_test_log,
    }

    try:
        out_dir = Path(VOLUME_PATH) / EXPERIMENT / "by_label"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{label}.json").write_text(json.dumps(record))
        # Save predictions for figures
        import numpy as np
        np.savez(out_dir / f"{label}_pred.npz",
                 u_pred=u_pred.ravel(), v_pred=v_pred.ravel(),
                 w_pred=w_pred.ravel(), p_pred=p_pred.ravel())
        volume.commit()
    except Exception as e:
        print(f"[{label}] WARN commit: {e}", flush=True)

    return record


@app.local_entrypoint()
def main():
    cfg = [
        ("aneur_unsup_pinn_5x64", "pinn", 5, 64, False, 21, LR_PINN, ITER),
        ("aneur_unsup_mix_3x1",   "mix",  3, 1,  False, 21, LR_MIX,  ITER),
    ]
    print(f"[main] launching {len(cfg)} jobs aneurisma unsup...", flush=True)
    t0 = time.perf_counter()
    results = list(train_one.starmap(cfg))
    wall = time.perf_counter() - t0
    print(f"[main] {len(results)} done in {wall:.1f}s", flush=True)
    Path("/tmp/aneur_unsup_results.json").write_text(json.dumps(results, indent=2))
