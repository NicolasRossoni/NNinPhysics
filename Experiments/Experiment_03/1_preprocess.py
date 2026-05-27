#### Importando Bibliotecas
import io
import math
from pathlib import Path

import modal


### ======================================== ###
###          Configuracao do Modal           ###
### ======================================== ###

PARENT = Path(__file__).resolve().parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "numpy")
)
app = modal.App("nnphysics-exp03-preprocess", image=image)
volume = modal.Volume.from_name("tcc", create_if_missing=True)

VOLUME_PATH = "/data"
PREP_PATH = "preprocess/exp_03"


### ======================================== ###
###          Parametros do Problema          ###
### ======================================== ###

# Numero de Reynolds e parametro lambda (solucao analitica de Kovasznay).
RE = 40.0
LAM = RE / 2.0 - math.sqrt((RE / 2.0) ** 2 + 4.0 * math.pi ** 2)

# Dominio retangular.
X_MIN, X_MAX = -0.5, 1.0
Y_MIN, Y_MAX = -0.5, 1.5

# Pontos de colocacao (interior) por Latin Hypercube 2D.
N_INT = 4000

# Grid de validacao analitica.
NX_VAL, NY_VAL = 80, 100

# Semente unica usada em todo o experimento.
SEED = 21


@app.function(volumes={VOLUME_PATH: volume})
def preprocess() -> None:
    import numpy as np
    import torch as tc

    tc.set_default_dtype(tc.float64)
    tc.manual_seed(SEED)
    np.random.seed(SEED)

    ### ==================================== ###
    ###  Hipercubo Latino 2D no dominio      ###
    ### ==================================== ###

    def latin_hypercube(n_points: int) -> tc.Tensor:
        intervalos = tc.linspace(0.0, 1.0, n_points + 1)
        pontos = tc.zeros(n_points, 2)
        for i in range(2):
            perm = tc.randperm(n_points)
            pontos[:, i] = intervalos[:-1] + (intervalos[1] - intervalos[0]) * tc.rand(n_points)
            pontos[:, i] = pontos[perm, i]
        pontos[:, 0] = pontos[:, 0] * (X_MAX - X_MIN) + X_MIN
        pontos[:, 1] = pontos[:, 1] * (Y_MAX - Y_MIN) + Y_MIN
        return pontos

    treino = latin_hypercube(N_INT)

    ### ==================================== ###
    ###  Solucao analitica de Kovasznay       ###
    ### ==================================== ###

    def kovasznay(x: tc.Tensor, y: tc.Tensor):
        u = 1.0 - tc.exp(LAM * x) * tc.cos(2.0 * math.pi * y)
        v = (LAM / (2.0 * math.pi)) * tc.exp(LAM * x) * tc.sin(2.0 * math.pi * y)
        p = 0.5 * (1.0 - tc.exp(2.0 * LAM * x))
        return u, v, p

    ### ==================================== ###
    ###       Grid de validacao 80x100        ###
    ### ==================================== ###

    xv = tc.linspace(X_MIN, X_MAX, NX_VAL)
    yv = tc.linspace(Y_MIN, Y_MAX, NY_VAL)
    XV, YV = tc.meshgrid(xv, yv, indexing="xy")
    validacao = tc.stack([XV.reshape(-1), YV.reshape(-1)], dim=1)
    u_val, v_val, p_val = kovasznay(validacao[:, 0:1], validacao[:, 1:2])
    validacao_uvp = tc.cat([u_val, v_val, p_val], dim=1)

    ### ==================================== ###
    ###         Salvando no volume            ###
    ### ==================================== ###

    out_dir = Path(VOLUME_PATH) / PREP_PATH
    out_dir.mkdir(parents=True, exist_ok=True)

    tc.save(treino, out_dir / "treino.pt")
    tc.save(validacao, out_dir / "validacao.pt")
    tc.save(validacao_uvp, out_dir / "validacao_uvp.pt")

    metadata = {
        "RE": RE,
        "LAM": LAM,
        "X_MIN": X_MIN, "X_MAX": X_MAX,
        "Y_MIN": Y_MIN, "Y_MAX": Y_MAX,
        "N_INT": N_INT,
        "NX_VAL": NX_VAL, "NY_VAL": NY_VAL,
        "SEED": SEED,
    }
    tc.save(metadata, out_dir / "metadata.pt")

    volume.commit()
    print(f"[preprocess] salvo em {VOLUME_PATH}/{PREP_PATH}/", flush=True)
    print(f"[preprocess] N_INT={N_INT}, val_grid={NX_VAL}x{NY_VAL}, lambda={LAM:.6f}", flush=True)


@app.local_entrypoint()
def main() -> None:
    preprocess.remote()
