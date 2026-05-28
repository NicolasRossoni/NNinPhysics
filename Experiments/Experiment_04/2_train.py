#### Importando Bibliotecas
import json
import time
from pathlib import Path

import modal

### ============= ### ###  Hiperparametros  ### ###  ============= ###

# Treino das nove configuracoes da Tabela 5 do monograph.
# Cada job roda em um container T4 paralelo em ate 9 containers simultaneos.
#
# Cada configuracao e uma tupla:
#   (label, kind, n_layers, width, sof, sup_ratio, pde_weight, iterations)
# Onde:
#   - kind: "pinn" (MLP tanh) ou "mix" (Mix2Funn).
#   - sup_ratio: fracao de pontos da nuvem ANEUMO usados como dado supervisionado
#       (0.0 = nao-supervisionado, 1.0 = supervisao total sem residuo da EDP).
#   - pde_weight: peso do residuo de Navier-Stokes na loss; 0 quando sup_ratio=1.

SEED = 21
RHO = 1060.0          # densidade do sangue (kg/m^3)
MU = 0.0035           # viscosidade do sangue (Pa.s)

LR_PINN = 1e-3
LR_MIX = 1e-2
T_INIT_MIX = 5.0
T_FINAL_MIX = 0.05

SCHED_STEP = 3000
SCHED_GAMMA = 0.5
N_INT_BATCH = 4000    # pontos de colocacao por iteracao
N_SUP_BATCH = 4000    # pontos supervisionados por iteracao
LAMBDA_WALL = 100.0
LAMBDA_INLET = 100.0
LAMBDA_SUP = 100.0
EPOCHS_LOG = 100

ITERATIONS = 15000

# (label, kind, n_layers, width, sof, sup_ratio, pde_weight, iterations)
CONFIGS = [
    ("pinn_5x64_unsup",          "pinn", 5,  64,  False, 0.00, 1.0, ITERATIONS),
    ("pinn_8x128_unsup",         "pinn", 8, 128,  False, 0.00, 1.0, ITERATIONS),
    ("mix_3x1_unsup",            "mix",  3,  1,   False, 0.00, 1.0, ITERATIONS),
    ("mix_3x1_sof_unsup",        "mix",  3,  1,   True,  0.00, 1.0, ITERATIONS),
    ("pinn_5x64_semi_25",        "pinn", 5,  64,  False, 0.25, 1.0, ITERATIONS),
    ("pinn_5x64_semi_50",        "pinn", 5,  64,  False, 0.50, 1.0, ITERATIONS),
    ("mix_2x2_sof_semi_50",      "mix",  2,  2,   True,  0.50, 1.0, ITERATIONS),
    ("pinn_5x64_sup_full",       "pinn", 5,  64,  False, 1.00, 0.0, ITERATIONS),
    ("mix_2x2_sof_sup_full",     "mix",  2,  2,   True,  1.00, 0.0, ITERATIONS),
]

### ============= ### ###  Modal App  ### ###  ============= ###

PARENT = Path(__file__).resolve().parent
app = modal.App(
    "nnphysics-exp04",
    image=(modal.Image.debian_slim(python_version="3.11")
           .pip_install("torch==2.5.1", "numpy")
           .add_local_file(str(PARENT / "mixfunn.py"), "/root/mixfunn.py", copy=True)),
)
volume = modal.Volume.from_name("tcc", create_if_missing=True)
VOLUME_PATH = "/data"
DATA_NPZ = "/data/preprocess/exp_04/case_AN4_m002.npz"
CHECKPOINT_DIR = "/data/checkpoints/exp_04"


### ============= ### ###  Funcao remota de treino  ### ###  ============= ###

@app.function(gpu="T4", timeout=120 * 60, volumes={VOLUME_PATH: volume},
              max_containers=12)
def train_one(label: str, kind: str, n_layers: int, width: int, sof: bool,
              sup_ratio: float, pde_weight: float, iterations: int) -> dict:
    import sys
    sys.path.insert(0, "/root")
    import numpy as np
    import torch as tc
    from torch import nn

    tc.set_default_dtype(tc.float32)
    tc.manual_seed(SEED)
    np.random.seed(SEED)
    device = tc.device("cuda" if tc.cuda.is_available() else "cpu")

    # ---- Carrega dataset ANEUMO ----
    npz = np.load(DATA_NPZ)
    xyz = npz["xyz"].astype(np.float32)             # (N, 3) mm
    u_gt = npz["u"].astype(np.float32)
    v_gt = npz["v"].astype(np.float32)
    w_gt = npz["w"].astype(np.float32)
    p_gt = npz["p"].astype(np.float32)
    sdf = npz["sdf"].astype(np.float32)             # (N,) mm
    xyz_inlet = npz["xyz_inlet"].astype(np.float32) # (M, 3) mm

    # Convertendo mm -> m para o residuo fisico em SI
    SCALE_LEN = 1e-3
    xyz_m = xyz * SCALE_LEN
    xyz_inlet_m = xyz_inlet * SCALE_LEN
    sdf_m = sdf * SCALE_LEN

    bbox_min = xyz_m.min(axis=0); bbox_max = xyz_m.max(axis=0)
    bbox_center = (bbox_min + bbox_max) / 2
    bbox_scale = float((bbox_max - bbox_min).max())

    def normalize(p):
        return (p - bbox_center) / bbox_scale

    xyz_norm = normalize(xyz_m).astype(np.float32)
    xyz_inlet_norm = normalize(xyz_inlet_m).astype(np.float32)

    # ---- Mapeia entrada para pontos de contorno e supervisao ----
    # Inlet ground truth: para cada ponto de entrada, valor verdadeiro de uvw.
    inlet_idx = np.zeros(xyz_inlet.shape[0], dtype=int)
    for i, pp in enumerate(xyz_inlet):
        d = np.linalg.norm(xyz - pp, axis=1)
        inlet_idx[i] = int(np.argmin(d))
    u_inlet = u_gt[inlet_idx]; v_inlet = v_gt[inlet_idx]; w_inlet = w_gt[inlet_idx]

    # Parede: pontos com SDF positivo no decil inferior.
    nonzero_sdf = sdf_m[sdf_m > 0]
    sdf_thresh = float(np.quantile(nonzero_sdf, 0.10)) if len(nonzero_sdf) else 1e-5
    wall_idx = np.where(sdf_m < sdf_thresh)[0]
    if len(wall_idx) == 0:
        wall_idx = np.argsort(sdf_m)[:5000]

    # Supervisao: subconjunto aleatorio da nuvem (alvo uvw vem do ANEUMO).
    N_total = xyz_norm.shape[0]
    n_sup = int(sup_ratio * N_total)
    rng = np.random.default_rng(SEED)
    sup_idx = rng.choice(N_total, size=n_sup, replace=False) if n_sup > 0 else np.array([], dtype=int)

    # ---- Move tudo para torch ----
    xyz_t = tc.from_numpy(xyz_norm).to(device)
    xyz_inlet_t = tc.from_numpy(xyz_inlet_norm).to(device)
    u_in_t = tc.from_numpy(u_inlet).to(device).unsqueeze(1)
    v_in_t = tc.from_numpy(v_inlet).to(device).unsqueeze(1)
    w_in_t = tc.from_numpy(w_inlet).to(device).unsqueeze(1)
    u_gt_t = tc.from_numpy(u_gt).to(device).unsqueeze(1)
    v_gt_t = tc.from_numpy(v_gt).to(device).unsqueeze(1)
    w_gt_t = tc.from_numpy(w_gt).to(device).unsqueeze(1)
    p_gt_t = tc.from_numpy(p_gt).to(device).unsqueeze(1)
    wall_t = tc.from_numpy(xyz_norm[wall_idx]).to(device)
    sup_t_xyz = tc.from_numpy(xyz_norm[sup_idx]).to(device) if n_sup > 0 else None
    sup_t_u = u_gt_t[sup_idx] if n_sup > 0 else None
    sup_t_v = v_gt_t[sup_idx] if n_sup > 0 else None
    sup_t_w = w_gt_t[sup_idx] if n_sup > 0 else None
    sup_t_p = p_gt_t[sup_idx] if n_sup > 0 else None

    L_PHYS = bbox_scale
    N_wall = wall_t.shape[0]; N_inlet = xyz_inlet_t.shape[0]
    print(f"[{label}] wall={N_wall} inlet={N_inlet} sup={n_sup} pde_w={pde_weight}", flush=True)

    ### ============= ### ###  Construindo a rede  ### ###  ============= ###

    if kind == "pinn":
        layers = [3] + [width] * n_layers + [4]
        mods = []
        for i in range(len(layers) - 1):
            mods.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                mods.append(nn.Tanh())
        rede = nn.Sequential(*mods).to(device)
        for m in rede.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        lr = LR_PINN
    else:
        # MixFunn canonica (Mix2Funn de mixfunn.py): 7 funcoes de base,
        # pre-mistura quadratica das entradas e produtos cruzados de 2a ordem.
        from mixfunn import Mix2Funn
        rede = Mix2Funn(
            n_in=3, n_out=4, n_layers=n_layers, n_hidden=width,
            use_softmax=True, T_init=T_INIT_MIX, T_final=T_FINAL_MIX,
            n_anneal_epochs=iterations,
            second_order_function=sof, init_alpha_std=0.1,
        ).to(device)
        lr = LR_MIX

    n_params = int(sum(p.numel() for p in rede.parameters() if p.requires_grad))
    opt = tc.optim.Adam(rede.parameters(), lr=lr)
    sched = tc.optim.lr_scheduler.StepLR(opt, step_size=SCHED_STEP, gamma=SCHED_GAMMA)

    ### ============= ### ###  Loop de treino  ### ###  ============= ###

    epochs_log = []; loss_total_log = []; l2_curve_log = []
    loss_pde_log = []; loss_wall_log = []; loss_inlet_log = []; loss_sup_log = []
    t0 = time.perf_counter()

    for ep in range(iterations + 1):
        if kind == "mix":
            rede.update_temperature_from_epoch(ep)

        loss = tc.tensor(0.0, device=device)
        l_pde_val = 0.0; l_wall_val = 0.0; l_in_val = 0.0; l_sup_val = 0.0

        # ---- Residuo de Navier-Stokes (incompressivel estacionario) ----
        if pde_weight > 0:
            idx_col = tc.randint(0, N_total, (N_INT_BATCH,), device=device)
            xc = xyz_t[idx_col].clone().requires_grad_(True)
            out = rede(xc)
            u = out[:, 0:1]; v = out[:, 1:2]; w = out[:, 2:3]; p = out[:, 3:4]

            def grad(o, i):
                return tc.autograd.grad(o, i, grad_outputs=tc.ones_like(o), create_graph=True)[0]

            gu = grad(u, xc); u_x = gu[:, 0:1]/L_PHYS; u_y = gu[:, 1:2]/L_PHYS; u_z = gu[:, 2:3]/L_PHYS
            gv = grad(v, xc); v_x = gv[:, 0:1]/L_PHYS; v_y = gv[:, 1:2]/L_PHYS; v_z = gv[:, 2:3]/L_PHYS
            gw = grad(w, xc); w_x = gw[:, 0:1]/L_PHYS; w_y = gw[:, 1:2]/L_PHYS; w_z = gw[:, 2:3]/L_PHYS
            gp = grad(p, xc); p_x = gp[:, 0:1]/L_PHYS; p_y = gp[:, 1:2]/L_PHYS; p_z = gp[:, 2:3]/L_PHYS
            u_xx = grad(u_x, xc)[:, 0:1]/L_PHYS
            u_yy = grad(u_y, xc)[:, 1:2]/L_PHYS
            u_zz = grad(u_z, xc)[:, 2:3]/L_PHYS
            v_xx = grad(v_x, xc)[:, 0:1]/L_PHYS
            v_yy = grad(v_y, xc)[:, 1:2]/L_PHYS
            v_zz = grad(v_z, xc)[:, 2:3]/L_PHYS
            w_xx = grad(w_x, xc)[:, 0:1]/L_PHYS
            w_yy = grad(w_y, xc)[:, 1:2]/L_PHYS
            w_zz = grad(w_z, xc)[:, 2:3]/L_PHYS

            res_u = RHO*(u*u_x + v*u_y + w*u_z) + p_x - MU*(u_xx + u_yy + u_zz)
            res_v = RHO*(u*v_x + v*v_y + w*v_z) + p_y - MU*(v_xx + v_yy + v_zz)
            res_w = RHO*(u*w_x + v*w_y + w*w_z) + p_z - MU*(w_xx + w_yy + w_zz)
            res_div = u_x + v_y + w_z
            # Escala do residuo do momento (Pa/m ~ 1e3): reduz para faixa do divergente
            L_pde = (tc.mean(res_u**2) + tc.mean(res_v**2) + tc.mean(res_w**2)) * 1e-6
            L_div = tc.mean(res_div**2)
            loss = loss + pde_weight * (L_pde + L_div)
            l_pde_val = float((L_pde + L_div).item())

        # ---- Contorno na parede: u=v=w=0 ----
        idx_w = tc.randint(0, N_wall, (min(N_INT_BATCH, N_wall),), device=device)
        out_w = rede(wall_t[idx_w])
        L_wall = tc.mean(out_w[:, 0]**2 + out_w[:, 1]**2 + out_w[:, 2]**2)
        loss = loss + LAMBDA_WALL * L_wall
        l_wall_val = float(L_wall.item())

        # ---- Contorno na entrada: velocidade fisiologica ----
        idx_i = tc.randint(0, N_inlet, (min(N_INT_BATCH, N_inlet),), device=device)
        out_i = rede(xyz_inlet_t[idx_i])
        L_inlet = (tc.mean((out_i[:, 0:1] - u_in_t[idx_i])**2)
                   + tc.mean((out_i[:, 1:2] - v_in_t[idx_i])**2)
                   + tc.mean((out_i[:, 2:3] - w_in_t[idx_i])**2))
        loss = loss + LAMBDA_INLET * L_inlet
        l_in_val = float(L_inlet.item())

        # ---- Supervisao parcial / total ----
        if n_sup > 0:
            idx_s = tc.randint(0, n_sup, (min(N_SUP_BATCH, n_sup),), device=device)
            out_s = rede(sup_t_xyz[idx_s])
            L_sup = (tc.mean((out_s[:, 0:1] - sup_t_u[idx_s])**2)
                     + tc.mean((out_s[:, 1:2] - sup_t_v[idx_s])**2)
                     + tc.mean((out_s[:, 2:3] - sup_t_w[idx_s])**2))
            # No regime 100% sup tambem aprendemos a pressao por dado
            if pde_weight == 0:
                L_sup = L_sup + tc.mean((out_s[:, 3:4] - sup_t_p[idx_s])**2)
            loss = loss + LAMBDA_SUP * L_sup
            l_sup_val = float(L_sup.item())

        opt.zero_grad(); loss.backward(); opt.step(); sched.step()

        if ep % EPOCHS_LOG == 0:
            with tc.no_grad():
                of = rede(xyz_t)
                up = of[:, 0:1]; vp = of[:, 1:2]; wp = of[:, 2:3]
                num = ((up - u_gt_t)**2 + (vp - v_gt_t)**2 + (wp - w_gt_t)**2).sum()
                den = (u_gt_t**2 + v_gt_t**2 + w_gt_t**2).sum()
                l2 = float(tc.sqrt(num/den))
            epochs_log.append(ep)
            loss_total_log.append(float(loss.item()))
            l2_curve_log.append(l2)
            loss_pde_log.append(l_pde_val); loss_wall_log.append(l_wall_val)
            loss_inlet_log.append(l_in_val); loss_sup_log.append(l_sup_val)
            if ep % 1000 == 0:
                wall = time.perf_counter() - t0
                print(f"[{label}] ep={ep:5d} loss={loss.item():.4e} L2={l2:.4e} wall={wall:.0f}s", flush=True)

    wall = time.perf_counter() - t0

    ### ============= ### ###  Avaliacao final  ### ###  ============= ###

    with tc.no_grad():
        of = rede(xyz_t)
        u_pred = of[:, 0:1].cpu().numpy().ravel()
        v_pred = of[:, 1:2].cpu().numpy().ravel()
        w_pred = of[:, 2:3].cpu().numpy().ravel()
        p_pred = of[:, 3:4].cpu().numpy().ravel()
        # MSE total considerando u, v, w (Pa-scale da pressao fica em diagnostico)
        mse_uvw = float(((u_pred - u_gt)**2 + (v_pred - v_gt)**2 + (w_pred - w_gt)**2).mean() / 3.0)
        num = ((u_pred - u_gt)**2 + (v_pred - v_gt)**2 + (w_pred - w_gt)**2).sum()
        den = (u_gt**2 + v_gt**2 + w_gt**2).sum()
        l2_test = float((num/den) ** 0.5)
        mse_p = float(((p_pred - p_gt)**2).mean())

    print(f"[{label}] DONE wall={wall:.0f}s n_par={n_params} MSE_uvw={mse_uvw:.4e} L2={l2_test:.4e}", flush=True)

    record = {
        "label": label, "kind": kind, "n_layers": n_layers, "width": width,
        "sof": sof, "seed": SEED, "iterations": iterations,
        "sup_ratio": sup_ratio, "pde_weight": pde_weight,
        "n_params": n_params,
        "mse_uvw": mse_uvw, "mse_p": mse_p, "l2_test_final": l2_test,
        "wall_clock": wall, "epochs_log": epochs_log,
        "loss_total": loss_total_log, "loss_pde": loss_pde_log,
        "loss_wall": loss_wall_log, "loss_inlet": loss_inlet_log,
        "loss_sup": loss_sup_log, "l2_curve": l2_curve_log,
    }

    out_dir = Path(CHECKPOINT_DIR) / "by_label"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{label}.json").write_text(json.dumps(record))
    import numpy as np
    np.savez(out_dir / f"{label}_pred.npz",
             u_pred=u_pred, v_pred=v_pred, w_pred=w_pred, p_pred=p_pred)
    volume.commit()

    return record


### ============= ### ###  Entrypoint  ### ###  ============= ###

@app.local_entrypoint()
def main():
    run_cfgs = list(CONFIGS)
    print(f"[main] disparando {len(run_cfgs)} jobs ANEUMO em paralelo...", flush=True)
    t0 = time.perf_counter()
    args = [(lbl, kind, nl, w, sof, sup, pde, it)
            for (lbl, kind, nl, w, sof, sup, pde, it) in run_cfgs]
    results = list(train_one.starmap(args))
    wall = time.perf_counter() - t0
    print(f"[main] {len(results)} runs em {wall:.1f}s", flush=True)
    print("\nResumo das nove configuracoes (MSE_uvw):")
    for r in results:
        sup_pct = int(round(100 * r["sup_ratio"]))
        print(f"  {r['label']:28s} sup={sup_pct:3d}% MSE_uvw={r['mse_uvw']:.4e} L2={r['l2_test_final']:.4e}")
