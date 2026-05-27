#### Importando Bibliotecas
import json
import math
from pathlib import Path

import modal

### ============= ### ###  Hiperparametros do Pre-Processamento  ### ###  ============= ###

SEED = 21                                        # semente unica usada em todo o experimento
RE = 40.0                                        # numero de Reynolds (Kovasznay)
LAM = RE / 2.0 - math.sqrt((RE / 2.0) ** 2 + 4.0 * math.pi ** 2)
X_MIN, X_MAX = -0.5, 1.0                         # dominio em x
Y_MIN, Y_MAX = -0.5, 1.5                         # dominio em y

N_INT = 4000                                     # pontos de colocacao interiores (Latin-Hypercube)
N_REF_X, N_REF_Y = 80, 100                       # malha de referencia para validacao L2

### ============= ### ###  Imagem e Volume Modal  ### ###  ============= ###

PARENT = Path(__file__).resolve().parent
image = modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.5.1", "numpy")
app = modal.App("nnphysics-exp02-preprocess", image=image)
volume = modal.Volume.from_name("tcc", create_if_missing=True)

VOLUME_PATH = "/data"
PREPROCESS_DIR = "preprocess/exp_02"

### ============= ### ###  Geracao dos Dados no Modal  ### ###  ============= ###

@app.function(volumes={VOLUME_PATH: volume}, timeout=10 * 60)
def gerar_dados():
    import numpy as np
    import torch as tc

    tc.set_default_dtype(tc.float64)
    tc.manual_seed(SEED)
    np.random.seed(SEED)

    def kovasznay(x, y):
        u = 1.0 - np.exp(LAM * x) * np.cos(2.0 * np.pi * y)
        v = (LAM / (2.0 * np.pi)) * np.exp(LAM * x) * np.sin(2.0 * np.pi * y)
        p = 0.5 * (1.0 - np.exp(2.0 * LAM * x))
        return u, v, p

    ### ============= ### ###  Amostragem Latin-Hypercube  ### ###  ============= ###
    # LHS estratificado em N_INT celulas: 1 ponto por celula em [0,1) por eixo,
    # com permutacao independente. Compatibilidade com BC hard de Coons preservada
    # pois as bordas sao impostas analiticamente no forward (nao dependem da nuvem).

    rng = np.random.default_rng(SEED)
    u1 = (rng.permutation(N_INT) + rng.random(N_INT)) / N_INT
    u2 = (rng.permutation(N_INT) + rng.random(N_INT)) / N_INT
    x_int = X_MIN + (X_MAX - X_MIN) * u1
    y_int = Y_MIN + (Y_MAX - Y_MIN) * u2
    treino = np.stack([x_int, y_int], axis=1)

    ### ============= ### ###  Malha de Validacao  ### ###  ============= ###

    xe = np.linspace(X_MIN, X_MAX, N_REF_X)
    ye = np.linspace(Y_MIN, Y_MAX, N_REF_Y)
    XE, YE = np.meshgrid(xe, ye, indexing="xy")
    u_ref, v_ref, p_ref = kovasznay(XE, YE)
    validacao = {
        "x_lin": xe.tolist(),
        "y_lin": ye.tolist(),
        "u_ref": u_ref.tolist(),
        "v_ref": v_ref.tolist(),
        "p_ref": p_ref.tolist(),
    }

    ### ============= ### ###  Persistencia no Volume  ### ###  ============= ###

    out_dir = Path(VOLUME_PATH) / PREPROCESS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    tc.save(tc.tensor(treino, dtype=tc.float64), out_dir / "treino_colocacao.pt")

    metadata = {
        "seed": SEED,
        "Re": RE,
        "lambda": LAM,
        "x_min": X_MIN, "x_max": X_MAX,
        "y_min": Y_MIN, "y_max": Y_MAX,
        "n_int": N_INT,
        "n_ref_x": N_REF_X, "n_ref_y": N_REF_Y,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (out_dir / "validacao_ref.json").write_text(json.dumps(validacao))
    volume.commit()

    print(f"[preprocess] {N_INT} pontos LHS salvos em {PREPROCESS_DIR}/", flush=True)
    print(f"[preprocess] malha de validacao {N_REF_X}x{N_REF_Y} salva", flush=True)
    print(f"[preprocess] lambda = {LAM:.6f}", flush=True)
    return metadata


@app.local_entrypoint()
def main():
    meta = gerar_dados.remote()
    print(f"[preprocess] concluido: {meta}", flush=True)
