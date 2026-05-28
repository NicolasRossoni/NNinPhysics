#### Importando Bibliotecas
import json
import time
from pathlib import Path

import modal

### ============= ### ###  Hiperparametros do Treino  ### ###  ============= ###

SEED = 22                                        # mesma semente do pre-processamento

# Dominio fisico (deve casar com 1_preprocess.py)
X_MIN, X_MAX = -1.0, 1.0
T_MIN, T_MAX =  0.0, 1.0
NU = 0.01 / 3.141592653589793

# Treinamento (Adam)
ITERATIONS = 30000
LR_PINN = 1e-3
LR_MIX  = 1e-2
SCHED_STEP_FRACTION = 0.25                       # step_size = iters * fraction
SCHED_GAMMA = 0.5

# Pesos das parcelas da loss (residuo + CI + BC)
LAMBDA_CI = 10.0
LAMBDA_BC = 10.0

# Mixfunn: annealing de temperatura softmax
T_INIT_MIX  = 5.0
T_FINAL_MIX = 0.05

# Frequencia de logging
EPOCHS_LOG = 300

### ============= ### ###  Imagem e Volume Modal  ### ###  ============= ###

PARENT = Path(__file__).resolve().parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "numpy")
    .add_local_file(str(PARENT / "mixfunn.py"), "/root/mixfunn.py", copy=True)
)
app = modal.App("nnphysics-exp05", image=image)
volume = modal.Volume.from_name("tcc", create_if_missing=True)

VOLUME_PATH = "/data"
PREPROCESS_DIR = "preprocess/exp_05"
CHECKPOINT_DIR = "checkpoints/exp_05"


### ============= ### ###  Funcao Remota: treina uma rede  ### ###  ============= ###

@app.function(gpu="T4", timeout=60 * 60, volumes={VOLUME_PATH: volume}, max_containers=4)
def train_one(
    label: str,
    kind: str,          # "pinn" ou "mix"
    n_layers: int,
    width: int,
    sof: bool,          # second-order cross-products (Mix)
    iters: int,
    lr: float,
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

    ### ============= ### ###  Carregando Dados de Pre-Processamento  ### ###  ============= ###

    data = np.load(Path(VOLUME_PATH) / PREPROCESS_DIR / "dataset.npz")
    x_eval = data["x_eval"]
    t_eval = data["t_eval"]
    u_ref  = data["u_ref"]
    treino_colocacao = data["treino_colocacao"]
    xt_ci = data["xt_ci"]
    u_ci  = data["u_ci"]
    xt_bc = data["xt_bc"]

    # Tensores no device
    colocacao = tc.tensor(treino_colocacao, device=device, dtype=tc.float32)
    xt_ci_t = tc.tensor(xt_ci, device=device, dtype=tc.float32)
    u_ci_t  = tc.tensor(u_ci,  device=device, dtype=tc.float32)
    xt_bc_t = tc.tensor(xt_bc, device=device, dtype=tc.float32)

    XE, TE = np.meshgrid(x_eval, t_eval, indexing="ij")
    xt_eval = tc.tensor(
        np.stack([XE.ravel(), TE.ravel()], axis=1),
        device=device, dtype=tc.float32,
    )

    ### ============= ### ###  Construindo a Rede  ### ###  ============= ###

    if kind == "pinn":
        layers = [2] + [width] * n_layers + [1]
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
            n_in=2, n_out=1, n_layers=n_layers, n_hidden=width,
            use_softmax=True, T_init=T_INIT_MIX, T_final=T_FINAL_MIX,
            n_anneal_epochs=iters,
            second_order_function=bool(sof),
            dropout=0.0, init_alpha_std=0.1,
        ).to(device)
    else:
        raise ValueError(f"kind invalido: {kind}")

    n_params = int(sum(p.numel() for p in net.parameters() if p.requires_grad))

    opt = tc.optim.Adam(net.parameters(), lr=lr)
    sched_step = max(1, int(iters * SCHED_STEP_FRACTION))
    sched = tc.optim.lr_scheduler.StepLR(opt, step_size=sched_step, gamma=SCHED_GAMMA)

    ### ============= ### ###  Loop de Treino (nao-supervisionado)  ### ###  ============= ###
    # Loss = mean(residuo^2) + LAMBDA_CI * mean(CI^2) + LAMBDA_BC * mean(BC^2)

    loss_curve = []
    l2_curve = []
    pde_curve = []
    ci_curve  = []
    bc_curve  = []
    epochs_log = []

    t0 = time.perf_counter()

    for ep in range(iters + 1):
        opt.zero_grad()

        # Resíduo da EDP nos pontos de colocacao
        xf = colocacao.detach().clone().requires_grad_(True)
        u = net(xf)
        g1 = tc.autograd.grad(u, xf, grad_outputs=tc.ones_like(u),
                              create_graph=True)[0]
        u_x = g1[:, 0:1]
        u_t = g1[:, 1:2]
        g2 = tc.autograd.grad(u_x, xf, grad_outputs=tc.ones_like(u_x),
                              create_graph=True)[0]
        u_xx = g2[:, 0:1]
        residuo = u_t + u * u_x - NU * u_xx
        loss_pde = tc.mean(residuo ** 2)

        # Condicao inicial u(x, 0) = -sin(pi x)
        u_ci_pred = net(xt_ci_t)
        loss_ci = tc.mean((u_ci_pred - u_ci_t) ** 2)

        # Condicao de contorno u(+-1, t) = 0
        u_bc_pred = net(xt_bc_t)
        loss_bc = tc.mean(u_bc_pred ** 2)

        loss = loss_pde + LAMBDA_CI * loss_ci + LAMBDA_BC * loss_bc

        loss.backward()
        opt.step()
        sched.step()

        if kind == "mix":
            net.update_temperature_from_epoch(ep)

        if ep % EPOCHS_LOG == 0:
            with tc.no_grad():
                u_pred_eval = net(xt_eval).cpu().numpy().reshape(len(x_eval), len(t_eval))
                l2 = float(
                    np.linalg.norm(u_pred_eval - u_ref) /
                    (np.linalg.norm(u_ref) + 1e-30)
                )
            loss_curve.append(float(loss.item()))
            l2_curve.append(l2)
            pde_curve.append(float(loss_pde.item()))
            ci_curve.append(float(loss_ci.item()))
            bc_curve.append(float(loss_bc.item()))
            epochs_log.append(ep)
            print(f"[{label}] ep={ep} loss={loss.item():.3e} "
                  f"L2={l2:.3e}", flush=True)

    wall = time.perf_counter() - t0

    ### ============= ### ###  Avaliacao Final  ### ###  ============= ###

    net.eval()
    with tc.no_grad():
        u_pred_final = net(xt_eval).cpu().numpy().reshape(len(x_eval), len(t_eval))
    l2_final = float(
        np.linalg.norm(u_pred_final - u_ref) /
        (np.linalg.norm(u_ref) + 1e-30)
    )

    ### ============= ### ###  Salvando Checkpoint  ### ###  ============= ###

    out_dir = Path(VOLUME_PATH) / CHECKPOINT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "label": label,
        "kind": kind,
        "n_layers": n_layers,
        "width": width,
        "sof": bool(sof),
        "lr": float(lr),
        "iters": int(iters),
        "lambda_ci": float(LAMBDA_CI),
        "lambda_bc": float(LAMBDA_BC),
        "n_params": n_params,
        "wall": float(wall),
        "l2_final": l2_final,
        "epochs_log": epochs_log,
        "loss_curve": loss_curve,
        "l2_curve": l2_curve,
        "pde_curve": pde_curve,
        "ci_curve": ci_curve,
        "bc_curve": bc_curve,
    }
    (out_dir / f"{label}.json").write_text(json.dumps(meta))
    np.savez(
        out_dir / f"{label}_pred.npz",
        u_pred=u_pred_final.astype(np.float32),
        x_eval=x_eval.astype(np.float32),
        t_eval=t_eval.astype(np.float32),
        u_ref=u_ref.astype(np.float32),
    )
    volume.commit()

    print(f"[{label}] kind={kind} L={n_layers} W={width} sof={sof} "
          f"L2={l2_final:.3e} wall={wall:.1f}s np={n_params}", flush=True)
    return meta


### ============= ### ###  Entrypoint  ### ###  ============= ###

@app.local_entrypoint()
def main() -> None:
    # Duas configuracoes principais (estritamente nao-supervisionadas):
    #   - PINN 6x64 com tanh
    #   - MixFunn-sof 3x3 (segunda ordem ligada, base completa de 7 funcoes)
    configs = [
        ("pinn_6x64", "pinn", 6, 64, False, ITERATIONS, LR_PINN),
        ("mix_3x3_sof", "mix", 3, 3, True,  ITERATIONS, LR_MIX),
    ]

    print(f"[main] lancando {len(configs)} jobs Modal T4...", flush=True)
    t0 = time.perf_counter()
    results = []
    for r in train_one.starmap(configs, return_exceptions=True):
        if isinstance(r, BaseException):
            print(f"[main] FALHOU: {type(r).__name__}: {r}", flush=True)
            continue
        results.append(r)
        print(f"[main] {len(results)}/{len(configs)} "
              f"({r['label']} L2={r['l2_final']:.3e} wall={r['wall']:.1f}s)",
              flush=True)
    wall = time.perf_counter() - t0
    print(f"[main] concluido: {len(results)} jobs em {wall:.1f}s", flush=True)
