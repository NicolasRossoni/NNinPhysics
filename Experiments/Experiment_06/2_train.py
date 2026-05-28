#### Importando Bibliotecas
import json
import math
import time
from pathlib import Path

import modal

### ============= ### ###  Hiperparametros do Treino  ### ###  ============= ###

SEED = 21                                  # mesma semente do pre-processamento
X_MIN, X_MAX = -5.0, 5.0                   # dominio em x
T_MIN, T_MAX = 0.0, math.pi / 2.0          # dominio em t

N_INT = 4000                               # pontos de colocacao (deve casar com pre-processamento)
N_IC = 1000                                # pontos da condicao inicial
N_BC = 1000                                # pontos das condicoes de contorno

LR_PINN = 1e-3                             # learning rate da PINN
LR_MIX = 1e-2                              # learning rate da MixFunn
SCHED_GAMMA = 0.5                          # gamma do StepLR
ITERS_PINN = 25000                         # iteracoes da PINN (mais profunda, demora pra convergir)
ITERS_MIX = 10000                          # iteracoes da MixFunn (converge cedo, depois oscila)

LAMBDA_IC = 100.0                          # peso da condicao inicial
LAMBDA_BC = 100.0                          # peso das condicoes de contorno

T_INIT_MIX = 5.0                           # temperatura inicial do softmax (Mix)
T_FINAL_MIX = 0.05                         # temperatura final do softmax (Mix)

EPOCHS_LOG = 500                           # frequencia de log da loss
NX_EVAL = 200                              # malha de avaliacao final em x
NT_EVAL = 100                              # malha de avaliacao final em t

### ============= ### ###  Imagem e Volume Modal  ### ###  ============= ###

PARENT = Path(__file__).resolve().parent
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "numpy", "scipy")
    .add_local_file(str(PARENT / "mixfunn.py"), "/root/mixfunn.py", copy=True)
)
app = modal.App("nnphysics-exp06", image=image)
volume = modal.Volume.from_name("tcc", create_if_missing=True)

VOLUME_PATH = "/data"
PREPROCESS_DIR = "preprocess/exp_06"
CHECKPOINTS_DIR = "checkpoints/exp_06"


### ============= ### ###  Treino de Uma Configuracao  ### ###  ============= ###

@app.function(
    gpu="T4",
    timeout=120 * 60,
    volumes={VOLUME_PATH: volume},
    max_containers=4,
    retries=0,
)
def train_one(
    label: str,
    kind: str,
    n_layers: int,
    width: int,
    lr: float,
    iterations: int,
) -> dict:
    import sys
    sys.path.insert(0, "/root")
    import numpy as np
    import torch as tc
    from torch import nn

    tc.set_default_dtype(tc.float32)
    tc.manual_seed(SEED)
    np.random.seed(SEED)
    device = tc.device("cuda" if tc.cuda.is_available() else "cpu")

    ### --- Carregamento dos dados pre-processados ---
    pre_dir = Path(VOLUME_PATH) / PREPROCESS_DIR
    treino = np.load(pre_dir / "treino.npz")
    referencia = np.load(pre_dir / "referencia.npz")

    treino_int = treino["treino_int"].astype(np.float32)   # (N_INT, 2): x, t
    treino_ci = treino["treino_ci"].astype(np.float32)     # (N_IC, 4): x, t, Re, Im
    treino_cc = treino["treino_cc"].astype(np.float32)     # (2*N_BC, 2): x, t

    x_eval_np = referencia["x"]
    t_eval_np = referencia["t"]
    u_ref_grid = referencia["u_ref"]
    v_ref_grid = referencia["v_ref"]
    abs_psi_ref = referencia["abs_psi_ref"]

    x_int_pool = tc.from_numpy(treino_int[:, 0:1]).to(device)
    t_int_pool = tc.from_numpy(treino_int[:, 1:2]).to(device)
    x_ic = tc.from_numpy(treino_ci[:, 0:1]).to(device)
    t_ic = tc.from_numpy(treino_ci[:, 1:2]).to(device)
    u_ic_ref = tc.from_numpy(treino_ci[:, 2:3]).to(device)
    v_ic_ref = tc.from_numpy(treino_ci[:, 3:4]).to(device)
    x_bc = tc.from_numpy(treino_cc[:, 0:1]).to(device)
    t_bc = tc.from_numpy(treino_cc[:, 1:2]).to(device)

    ### --- Construcao da rede ---
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
    elif kind == "mix":
        from mixfunn import Mix2Funn
        net = Mix2Funn(
            n_in=2, n_out=2,
            n_layers=n_layers, n_hidden=width,
            use_softmax=True, T_init=T_INIT_MIX, T_final=T_FINAL_MIX,
            n_anneal_epochs=iterations,
            second_order_function=True,
            dropout=0.0,
            init_alpha_std=0.1,
        ).to(device)
    else:
        raise ValueError(kind)

    n_params = int(sum(p.numel() for p in net.parameters() if p.requires_grad))

    ### --- Otimizador e scheduler ---
    opt = tc.optim.Adam(net.parameters(), lr=lr)
    sched = tc.optim.lr_scheduler.StepLR(
        opt, step_size=max(1, iterations // 4), gamma=SCHED_GAMMA,
    )

    ### --- Grade de avaliacao ---
    XEV, TEV = np.meshgrid(x_eval_np, t_eval_np, indexing="ij")
    xy_eval = tc.from_numpy(
        np.stack([XEV.ravel(), TEV.ravel()], axis=1).astype(np.float32)
    ).to(device)

    def forward_uv(x, t):
        xy = tc.cat([x, t], dim=1)
        out = net(xy)
        return out[:, 0:1], out[:, 1:2]

    def grad(out, inp):
        return tc.autograd.grad(
            out, inp, grad_outputs=tc.ones_like(out), create_graph=True,
        )[0]

    print(f"[{label}] start: n_params={n_params} device={device} "
          f"kind={kind} N_L={n_layers} W={width} iters={iterations}", flush=True)

    ### --- Loop de treino ---
    t0 = time.perf_counter()
    epochs_log, l2_curve, loss_curve = [], [], []
    pde_curve, ic_curve, bc_curve = [], [], []

    for ep in range(iterations + 1):
        if kind == "mix":
            net.update_temperature_from_epoch(ep)

        x_int = x_int_pool.detach().clone().requires_grad_(True)
        t_int = t_int_pool.detach().clone().requires_grad_(True)
        u, v = forward_uv(x_int, t_int)

        u_x = grad(u, x_int)
        u_t = grad(u, t_int)
        u_xx = grad(u_x, x_int)
        v_x = grad(v, x_int)
        v_t = grad(v, t_int)
        v_xx = grad(v_x, x_int)

        # NLS: i psi_t + 0.5 psi_xx + |psi|^2 psi = 0
        # Re: -v_t + 0.5 u_xx + |psi|^2 u = 0
        # Im:  u_t + 0.5 v_xx + |psi|^2 v = 0
        mag2 = u * u + v * v
        f_u = -v_t + 0.5 * u_xx + mag2 * u
        f_v = u_t + 0.5 * v_xx + mag2 * v
        loss_pde = tc.mean(f_u ** 2) + tc.mean(f_v ** 2)

        u_ic_p, v_ic_p = forward_uv(x_ic, t_ic)
        loss_ic = tc.mean((u_ic_p - u_ic_ref) ** 2) + tc.mean((v_ic_p - v_ic_ref) ** 2)

        u_bc_p, v_bc_p = forward_uv(x_bc, t_bc)
        loss_bc = tc.mean(u_bc_p ** 2) + tc.mean(v_bc_p ** 2)

        loss = loss_pde + LAMBDA_IC * loss_ic + LAMBDA_BC * loss_bc

        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

        if ep % EPOCHS_LOG == 0:
            with tc.no_grad():
                out_eval = net(xy_eval)
                u_g = out_eval[:, 0:1].cpu().numpy().reshape(NX_EVAL, NT_EVAL)
                v_g = out_eval[:, 1:2].cpu().numpy().reshape(NX_EVAL, NT_EVAL)
                abs_p = np.sqrt(u_g ** 2 + v_g ** 2)
                num = ((abs_p - abs_psi_ref) ** 2).sum()
                den = (abs_psi_ref ** 2).sum() + 1e-12
                l2 = float(np.sqrt(num / den))
            epochs_log.append(ep)
            loss_curve.append(float(loss.item()))
            l2_curve.append(l2)
            pde_curve.append(float(loss_pde.item()))
            ic_curve.append(float(loss_ic.item()))
            bc_curve.append(float(loss_bc.item()))
            print(f"[{label}] ep={ep} L_pde={float(loss_pde.item()):.2e} "
                  f"L_ic={float(loss_ic.item()):.2e} L_bc={float(loss_bc.item()):.2e} "
                  f"L2|psi|={l2:.3e} t={time.perf_counter()-t0:.0f}s",
                  flush=True)

    wall = time.perf_counter() - t0

    ### --- Avaliacao final ---
    net.eval()
    with tc.no_grad():
        out_eval = net(xy_eval)
        u_pred = out_eval[:, 0:1].cpu().numpy().reshape(NX_EVAL, NT_EVAL)
        v_pred = out_eval[:, 1:2].cpu().numpy().reshape(NX_EVAL, NT_EVAL)
        abs_p = np.sqrt(u_pred ** 2 + v_pred ** 2)
        num = ((abs_p - abs_psi_ref) ** 2).sum()
        den = (abs_psi_ref ** 2).sum() + 1e-12
        l2_val = float(np.sqrt(num / den))
        l2_u = float(np.linalg.norm(u_pred - u_ref_grid)
                     / (np.linalg.norm(u_ref_grid) + 1e-12))
        l2_v = float(np.linalg.norm(v_pred - v_ref_grid)
                     / (np.linalg.norm(v_ref_grid) + 1e-12))

    final_loss = float(loss.item())
    print(f"[{label}] DONE L2|psi|={l2_val:.3e} L2u={l2_u:.3e} L2v={l2_v:.3e} "
          f"loss={final_loss:.3e} wall={wall:.1f}s n={n_params}", flush=True)

    ### --- Persistencia no volume ---
    record = {
        "label": label, "kind": kind, "n_layers": n_layers, "width": width,
        "mode": "nsup", "seed": SEED, "lr": lr, "iterations": iterations,
        "lambda_ic": LAMBDA_IC, "lambda_bc": LAMBDA_BC,
        "n_params": n_params,
        "l2_val": l2_val, "l2_u": l2_u, "l2_v": l2_v,
        "final_loss": final_loss, "wall_clock": wall,
        "epochs_log": epochs_log,
        "loss_curve": loss_curve, "l2_curve": l2_curve,
        "pde_curve": pde_curve, "ic_curve": ic_curve, "bc_curve": bc_curve,
    }
    out_dir = Path(VOLUME_PATH) / CHECKPOINTS_DIR / "by_label"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{label}.json").write_text(json.dumps(record))
    import numpy as np_save
    np_save.savez_compressed(
        out_dir / f"{label}_pred.npz",
        u_pred=u_pred.astype(np_save.float32),
        v_pred=v_pred.astype(np_save.float32),
        abs_psi_pred=abs_p.astype(np_save.float32),
        u_ref=u_ref_grid.astype(np_save.float32),
        v_ref=v_ref_grid.astype(np_save.float32),
        abs_psi_ref=abs_psi_ref.astype(np_save.float32),
        x=x_eval_np.astype(np_save.float32),
        t=t_eval_np.astype(np_save.float32),
    )
    volume.commit()
    return record


### ============= ### ###  Despacho das Duas Configuracoes Finais  ### ###  ============= ###

@app.local_entrypoint()
def main():
    # (label, kind, n_layers, width, lr, iters)
    configs = [
        ("pinn_5x100_nsup", "pinn", 5, 100, LR_PINN, ITERS_PINN),
        ("mix_3x6_sof_nsup", "mix",  3,   6, LR_MIX,  ITERS_MIX),
    ]

    print(f"[main] despachando {len(configs)} jobs em paralelo no Modal T4...", flush=True)
    t0 = time.perf_counter()
    results = list(train_one.starmap(configs))
    wall = time.perf_counter() - t0
    print(f"[main] {len(results)} jobs concluidos em {wall:.1f}s", flush=True)
    for r in sorted(results, key=lambda r: r["l2_val"]):
        print(f"  {r['label']:22s} L2|psi|={r['l2_val']:.3e} "
              f"n_params={r['n_params']} wall={r['wall_clock']:.1f}s", flush=True)
