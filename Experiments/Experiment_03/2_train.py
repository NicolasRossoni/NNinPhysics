#### Importando Bibliotecas
import json
import math
import time
from pathlib import Path

import modal


### ======================================== ###
###          Configuracao do Modal           ###
### ======================================== ###

PARENT = Path(__file__).resolve().parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "numpy")
    .add_local_file(str(PARENT / "mixfunn.py"), "/root/mixfunn.py", copy=True)
)
app = modal.App("nnphysics-exp03", image=image)
volume = modal.Volume.from_name("tcc", create_if_missing=True)

VOLUME_PATH = "/data"
PREP_PATH = "preprocess/exp_03"
CKPT_PATH = "checkpoints/exp_03"


### ======================================== ###
###      Parametros fisicos do problema       ###
### ======================================== ###

RE = 40.0
LAM = RE / 2.0 - math.sqrt((RE / 2.0) ** 2 + 4.0 * math.pi ** 2)
X_MIN, X_MAX = -0.5, 1.0
Y_MIN, Y_MAX = -0.5, 1.5


### ======================================== ###
###            Hiperparametros                ###
### ======================================== ###

SEED = 21
N_INT = 4000
LR_MIX = 1e-2
ITER = 15000
SCHED_STEP = 3000
SCHED_GAMMA = 0.5
T_INIT_MIX = 5.0
T_FINAL_MIX = 0.05
EPOCHS_LOG = 500
S_BC = 5.0  # nitidez do tanh na funcao de distancia
PRUNE_RATIOS = [0.00, 0.30, 0.50, 0.70, 0.90]


@app.function(gpu="T4", timeout=120 * 60, volumes={VOLUME_PATH: volume}, max_containers=4)
def train_one(label: str, sof: bool) -> dict:
    import sys
    sys.path.insert(0, "/root")
    import numpy as np
    import torch as tc

    tc.set_default_dtype(tc.float64)
    tc.manual_seed(SEED)
    np.random.seed(SEED)
    device = tc.device("cuda" if tc.cuda.is_available() else "cpu")
    print(f"[{label}] device={device}, sof={sof}", flush=True)

    from mixfunn import Mix2Funn

    ### ======================================== ###
    ###            Carregando dados              ###
    ### ======================================== ###

    prep_dir = Path(VOLUME_PATH) / PREP_PATH
    treino_xy = tc.load(prep_dir / "treino.pt", map_location=device)
    validacao = tc.load(prep_dir / "validacao.pt", map_location=device)
    validacao_uvp = tc.load(prep_dir / "validacao_uvp.pt", map_location=device)

    ### ======================================== ###
    ###           Solucao analitica              ###
    ### ======================================== ###

    def kovasznay(x: tc.Tensor, y: tc.Tensor):
        u = 1.0 - tc.exp(LAM * x) * tc.cos(2.0 * math.pi * y)
        v = (LAM / (2.0 * math.pi)) * tc.exp(LAM * x) * tc.sin(2.0 * math.pi * y)
        p = 0.5 * (1.0 - tc.exp(2.0 * LAM * x))
        return u, v, p

    ### ======================================== ###
    ###    Condicao de contorno via Coons         ###
    ### ======================================== ###

    def dist_fn(xy: tc.Tensor) -> tc.Tensor:
        x = xy[:, 0:1]; y = xy[:, 1:2]
        return (tc.tanh(S_BC * (x - X_MIN))
                * tc.tanh(S_BC * (X_MAX - x))
                * tc.tanh(S_BC * (y - Y_MIN))
                * tc.tanh(S_BC * (Y_MAX - y)))

    def u_bc_fn(xy: tc.Tensor):
        # Interpolacao transfinita de Coons usando os valores analiticos
        # da solucao de Kovasznay avaliados nas quatro bordas do retangulo.
        x = xy[:, 0:1]; y = xy[:, 1:2]
        xi = (x - X_MIN) / (X_MAX - X_MIN)
        eta = (y - Y_MIN) / (Y_MAX - Y_MIN)
        xm_v = tc.full_like(x, X_MIN); xM_v = tc.full_like(x, X_MAX)
        ym_v = tc.full_like(y, Y_MIN); yM_v = tc.full_like(y, Y_MAX)

        uw, vw, pw = kovasznay(xm_v, y)
        ue, ve, pe = kovasznay(xM_v, y)
        us, vs, ps = kovasznay(x, ym_v)
        un, vn, pn = kovasznay(x, yM_v)

        usw, vsw, psw = kovasznay(xm_v, ym_v)
        use_, vse, pse = kovasznay(xM_v, ym_v)
        unw, vnw, pnw = kovasznay(xm_v, yM_v)
        une, vne, pne = kovasznay(xM_v, yM_v)

        def coons(b_w, b_e, b_s, b_n, c_sw, c_se, c_nw, c_ne):
            horiz = (1.0 - xi) * b_w + xi * b_e
            vert = (1.0 - eta) * b_s + eta * b_n
            corner = ((1.0 - xi) * (1.0 - eta) * c_sw
                      + xi * (1.0 - eta) * c_se
                      + (1.0 - xi) * eta * c_nw
                      + xi * eta * c_ne)
            return horiz + vert - corner

        u_bc = coons(uw, ue, us, un, usw, use_, unw, une)
        v_bc = coons(vw, ve, vs, vn, vsw, vse, vnw, vne)
        p_bc = coons(pw, pe, ps, pn, psw, pse, pnw, pne)
        return u_bc, v_bc, p_bc

    ### ======================================== ###
    ###             Definindo MixFunn             ###
    ### ======================================== ###
    # Duas variantes:
    #   sof=False -> Mix2Funn(2,3, n_layers=3, n_hidden=1)  -> 35 pesos alpha
    #   sof=True  -> Mix2Funn(2,3, n_layers=1, n_hidden=1)  -> 105 pesos alpha

    if sof:
        n_layers, n_hidden = 1, 1
    else:
        n_layers, n_hidden = 3, 1

    net = Mix2Funn(
        n_in=2, n_out=3,
        n_layers=n_layers, n_hidden=n_hidden,
        use_softmax=True, T_init=T_INIT_MIX, T_final=T_FINAL_MIX,
        n_anneal_epochs=ITER,
        second_order_function=sof,
        dropout=0.0, init_alpha_std=0.1,
    ).to(device)

    n_params = int(sum(p.numel() for p in net.parameters() if p.requires_grad))
    n_alpha = sum(p.numel() for n, p in net.named_parameters() if "alpha" in n)
    print(f"[{label}] n_params={n_params}, n_alpha={n_alpha}", flush=True)

    def forward_hard(xy: tc.Tensor):
        out = net(xy)
        d = dist_fn(xy)
        u_bc, v_bc, p_bc = u_bc_fn(xy)
        u = u_bc + d * out[:, 0:1]
        v = v_bc + d * out[:, 1:2]
        p = p_bc + d * out[:, 2:3]
        return u, v, p

    ### ======================================== ###
    ###               Treinando                   ###
    ### ======================================== ###

    opt = tc.optim.Adam(net.parameters(), lr=LR_MIX)
    sched = tc.optim.lr_scheduler.StepLR(opt, step_size=SCHED_STEP, gamma=SCHED_GAMMA)

    x_int_pool = treino_xy[:, 0:1].detach().clone()
    y_int_pool = treino_xy[:, 1:2].detach().clone()

    loss_log: list[float] = []
    l2_curve: list[tuple[int, float]] = []
    t0 = time.perf_counter()

    for ep in range(ITER + 1):
        net.update_temperature_from_epoch(ep)

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

        # Residuos das equacoes incompressiveis de Navier-Stokes 2D estacionarias.
        res_u = u * u_x + v * u_y + p_x - (1.0 / RE) * (u_xx + u_yy)
        res_v = u * v_x + v * v_y + p_y - (1.0 / RE) * (v_xx + v_yy)
        res_div = u_x + v_y
        loss = tc.mean(res_u ** 2) + tc.mean(res_v ** 2) + tc.mean(res_div ** 2)

        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

        if ep % EPOCHS_LOG == 0:
            with tc.no_grad():
                u_v, v_v, p_v = forward_hard(validacao)
                u_r = validacao_uvp[:, 0:1]
                v_r = validacao_uvp[:, 1:2]
                p_r = validacao_uvp[:, 2:3]
                p_v_n = p_v - p_v.mean(); p_r_n = p_r - p_r.mean()
                num = ((u_v - u_r) ** 2 + (v_v - v_r) ** 2 + (p_v_n - p_r_n) ** 2).sum()
                den = (u_r ** 2 + v_r ** 2 + p_r_n ** 2).sum()
                l2 = float(tc.sqrt(num / den))
            l2_curve.append((ep, l2))
            loss_log.append(float(loss.item()))
            print(f"[{label}] ep={ep:5d}  loss={loss.item():.3e}  L2_val={l2:.3e}", flush=True)

    wall = time.perf_counter() - t0
    net.eval()

    ### ======================================== ###
    ###      Avaliacao final + Pruning            ###
    ### ======================================== ###
    # Para cada razao r aplicamos prune_alpha(r), medimos L2, e
    # restauramos os alphas para a proxima razao (avaliacao independente
    # por razao, nao cumulativa).

    def avaliar_l2() -> float:
        with tc.no_grad():
            u_v, v_v, p_v = forward_hard(validacao)
            u_r = validacao_uvp[:, 0:1]
            v_r = validacao_uvp[:, 1:2]
            p_r = validacao_uvp[:, 2:3]
            p_v_n = p_v - p_v.mean(); p_r_n = p_r - p_r.mean()
            num = ((u_v - u_r) ** 2 + (v_v - v_r) ** 2 + (p_v_n - p_r_n) ** 2).sum()
            den = (u_r ** 2 + v_r ** 2 + p_r_n ** 2).sum()
            return float(tc.sqrt(num / den))

    def contar_zerados() -> tuple[int, int]:
        n_zero = 0; n_total = 0
        for n, p in net.named_parameters():
            if "alpha" in n:
                n_zero += int((p.abs() < 1e-9).sum().item())
                n_total += p.numel()
        return n_zero, n_total

    prune_results = []
    with tc.no_grad():
        for r in PRUNE_RATIOS:
            state_before = {n: p.clone() for n, p in net.named_parameters() if "alpha" in n}
            if r > 0:
                net.prune_alpha(r)
            l2_r = avaliar_l2()
            n_zero, n_total = contar_zerados()
            prune_results.append({
                "ratio": r, "l2": l2_r, "n_zero": n_zero, "n_total": n_total,
            })
            print(f"[{label}] r={r:.2f}  zerados={n_zero}/{n_total}  L2={l2_r:.3e}", flush=True)
            # restaura alphas para a proxima iteracao
            for n, p in net.named_parameters():
                if "alpha" in n and n in state_before:
                    p.data.copy_(state_before[n])

    l2_baseline = prune_results[0]["l2"]
    print(f"[{label}] wall={wall:.1f}s  L2_baseline={l2_baseline:.3e}", flush=True)

    ### ======================================== ###
    ###            Salvando checkpoint            ###
    ### ======================================== ###

    out_dir = Path(VOLUME_PATH) / CKPT_PATH
    out_dir.mkdir(parents=True, exist_ok=True)

    record = {
        "label": label,
        "sof": sof,
        "n_layers": n_layers,
        "n_hidden": n_hidden,
        "n_params": n_params,
        "n_alpha": n_alpha,
        "seed": SEED,
        "iterations": ITER,
        "lr": LR_MIX,
        "T_init": T_INIT_MIX,
        "T_final": T_FINAL_MIX,
        "T_at_final_epoch": T_FINAL_MIX,
        "wall_clock": wall,
        "l2_baseline": l2_baseline,
        "prune_results": prune_results,
        "loss_log": loss_log,
        "l2_curve": l2_curve,
    }

    (out_dir / f"{label}.json").write_text(json.dumps(record, indent=2))
    tc.save(net.state_dict(), out_dir / f"{label}.statedict.pt")
    volume.commit()

    return record


@app.local_entrypoint()
def main() -> None:
    configs = [
        ("mix_3x1",     False),  # MixFunn 3x1, 35 alphas
        ("mix_sof_1x1", True),   # MixFunn-sof 1x1, 105 alphas
    ]
    print(f"[main] lancando {len(configs)} jobs Modal T4...", flush=True)
    t0 = time.perf_counter()
    results = list(train_one.starmap(configs))
    wall = time.perf_counter() - t0
    print(f"[main] {len(results)} jobs concluidos em {wall:.1f}s", flush=True)
    for r in results:
        print(f"  {r['label']:20s} L2_baseline={r['l2_baseline']:.3e}  n_params={r['n_params']}",
              flush=True)
