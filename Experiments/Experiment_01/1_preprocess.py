#### Importando Bibliotecas
import json
import math
from pathlib import Path

import modal

### ============= ### ###  Hiperparametros  ### ###  ============= ###

# Numero de Reynolds e parametro analitico de Kovasznay
RE = 40.0
LAM = RE / 2.0 - math.sqrt((RE / 2.0) ** 2 + 4.0 * math.pi ** 2)

# Dominio fisico
X_MIN, X_MAX = -0.5, 1.0
Y_MIN, Y_MAX = -0.5, 1.5

# Numero de pontos
N_INT = 4000          # pontos interiores via Hipercubo Latino
N_SUP_GRID = 30       # grade densa para supervisao analitica
N_VAL_X, N_VAL_Y = 60, 80      # grade de validacao em treino
N_EVAL_X, N_EVAL_Y = 80, 100   # grade de avaliacao final
N_EXTRAP_X, N_EXTRAP_Y = 200, 120
X_INFER_MAX = 2.5     # extrapolacao em x

# Seed (precisa ser determinista entre 1_preprocess e 2_train)
SEED = 21

### ============= ### ###  Modal App  ### ###  ============= ###

app = modal.App(
    "nnphysics-exp01-preprocess",
    image=modal.Image.debian_slim(python_version="3.11").pip_install(
        "torch==2.5.1", "numpy"
    ),
)
volume = modal.Volume.from_name("tcc", create_if_missing=True)
VOLUME_PATH = "/data"
OUT_DIR = "/data/preprocess/exp_01"


### ============= ### ###  Solucao analitica de Kovasznay  ### ###  ============= ###

def kovasznay_np(x, y, lam):
    import numpy as np
    u = 1.0 - np.exp(lam * x) * np.cos(2.0 * np.pi * y)
    v = (lam / (2.0 * np.pi)) * np.exp(lam * x) * np.sin(2.0 * np.pi * y)
    p = 0.5 * (1.0 - np.exp(2.0 * lam * x))
    return u, v, p


### ============= ### ###  Hipercubo Latino 2D  ### ###  ============= ###

def latin_hypercube(n_points, x_min, x_max, y_min, y_max, seed):
    import numpy as np
    rng = np.random.default_rng(seed)
    intervals = np.linspace(0.0, 1.0, n_points + 1)
    points = np.zeros((n_points, 2))
    for i in range(2):
        u = intervals[:-1] + (intervals[1] - intervals[0]) * rng.random(n_points)
        perm = rng.permutation(n_points)
        points[:, i] = u[perm]
    points[:, 0] = points[:, 0] * (x_max - x_min) + x_min
    points[:, 1] = points[:, 1] * (y_max - y_min) + y_min
    return points


### ============= ### ###  Funcao remota  ### ###  ============= ###

@app.function(volumes={VOLUME_PATH: volume}, timeout=15 * 60)
def build_dataset():
    import numpy as np

    np.random.seed(SEED)

    # Pontos interiores (treino) via Hipercubo Latino
    treino = latin_hypercube(N_INT, X_MIN, X_MAX, Y_MIN, Y_MAX, seed=SEED)

    # Grade densa para supervisao analitica
    x_sup_lin = np.linspace(X_MIN, X_MAX, N_SUP_GRID)
    y_sup_lin = np.linspace(Y_MIN, Y_MAX, N_SUP_GRID)
    XS, YS = np.meshgrid(x_sup_lin, y_sup_lin, indexing="xy")
    x_sup = XS.reshape(-1, 1); y_sup = YS.reshape(-1, 1)
    u_sup, v_sup, p_sup = kovasznay_np(x_sup, y_sup, LAM)
    supervisionado = np.concatenate(
        [x_sup, y_sup, u_sup, v_sup, p_sup], axis=1
    )  # colunas: x, y, u, v, p

    # Grade de validacao (usada durante treino para curva L2)
    xv = np.linspace(X_MIN, X_MAX, N_VAL_X)
    yv = np.linspace(Y_MIN, Y_MAX, N_VAL_Y)
    XV, YV = np.meshgrid(xv, yv, indexing="xy")
    u_v, v_v, p_v = kovasznay_np(XV, YV, LAM)
    validacao = {
        "x_lin": xv, "y_lin": yv,
        "u_ref": u_v, "v_ref": v_v, "p_ref": p_v,
    }

    # Grade de avaliacao final (L2 reportado)
    xe = np.linspace(X_MIN, X_MAX, N_EVAL_X)
    ye = np.linspace(Y_MIN, Y_MAX, N_EVAL_Y)
    XE, YE = np.meshgrid(xe, ye, indexing="xy")
    u_e, v_e, p_e = kovasznay_np(XE, YE, LAM)
    teste = {
        "x_lin": xe, "y_lin": ye,
        "u_ref": u_e, "v_ref": v_e, "p_ref": p_e,
    }

    # Grade de extrapolacao em x estendido
    xex = np.linspace(X_MIN, X_INFER_MAX, N_EXTRAP_X)
    yex = np.linspace(Y_MIN, Y_MAX, N_EXTRAP_Y)
    XEX, YEX = np.meshgrid(xex, yex, indexing="xy")
    u_ex, _, _ = kovasznay_np(XEX, YEX, LAM)
    extrap = {
        "x_lin": xex, "y_lin": yex, "u_analytic": u_ex,
    }

    # Metadados
    metadata = {
        "RE": RE, "LAM": LAM,
        "X_MIN": X_MIN, "X_MAX": X_MAX,
        "Y_MIN": Y_MIN, "Y_MAX": Y_MAX,
        "N_INT": N_INT, "N_SUP_GRID": N_SUP_GRID,
        "N_VAL_X": N_VAL_X, "N_VAL_Y": N_VAL_Y,
        "N_EVAL_X": N_EVAL_X, "N_EVAL_Y": N_EVAL_Y,
        "N_EXTRAP_X": N_EXTRAP_X, "N_EXTRAP_Y": N_EXTRAP_Y,
        "X_INFER_MAX": X_INFER_MAX,
        "SEED": SEED,
    }

    # Salvando no volume
    out = Path(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / "treino.npz",
        treino=treino,
        supervisionado=supervisionado,
    )
    np.savez(out / "validacao.npz", **validacao)
    np.savez(out / "teste.npz", **teste)
    np.savez(out / "extrap.npz", **extrap)
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    volume.commit()

    print(f"[preprocess] Hipercubo Latino: {treino.shape[0]} pontos interiores")
    print(f"[preprocess] Supervisionado: {supervisionado.shape[0]} pontos analiticos")
    print(f"[preprocess] Validacao: grade {N_VAL_X} x {N_VAL_Y}")
    print(f"[preprocess] Teste: grade {N_EVAL_X} x {N_EVAL_Y}")
    print(f"[preprocess] Extrap: grade {N_EXTRAP_X} x {N_EXTRAP_Y} (x ate {X_INFER_MAX})")
    print(f"[preprocess] Saida em {OUT_DIR}")
    return metadata


@app.local_entrypoint()
def main():
    meta = build_dataset.remote()
    print("[preprocess] metadata:", json.dumps(meta, indent=2))
