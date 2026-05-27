#### Importando Bibliotecas
import json
import math
from pathlib import Path

import modal

### ============= ### ###  Hiperparametros  ### ###  ============= ###

# Dominio
X_MIN, X_MAX = -5.0, 5.0
T_MIN, T_MAX = 0.0, math.pi / 2.0

# Pontos de colocacao
N_INT = 4000        # pontos interiores via Hipercubo Latino
N_IC = 1000         # pontos da condicao inicial em t=0
N_BC = 1000         # pontos das condicoes de contorno em x=+/-5

# Resolucao da referencia split-step Fourier
N_FFT = 512         # numero de modos espectrais
DT_REF = 5e-5       # passo temporal da integracao

# Grade de avaliacao final (L2 reportado)
NX_EVAL = 200
NT_EVAL = 100

# Seed (precisa ser determinista entre 1_preprocess e 2_train)
SEED = 21

### ============= ### ###  Modal App  ### ###  ============= ###

app = modal.App(
    "nnphysics-exp06-preprocess",
    image=modal.Image.debian_slim(python_version="3.11").pip_install(
        "torch==2.5.1", "numpy", "scipy"
    ),
)
volume = modal.Volume.from_name("tcc", create_if_missing=True)
VOLUME_PATH = "/data"
OUT_DIR = "/data/preprocess/exp_06"


### ============= ### ###  Solucao de Referencia (split-step Fourier)  ### ###  ============= ###

def split_step_reference(x_eval, t_eval):
    """Integracao split-step Fourier (Strang) da NLS:
        i psi_t + 0.5 psi_xx + |psi|^2 psi = 0,    psi(x,0) = 2 sech(x).
    Retorna |psi|, Re(psi), Im(psi) na grade (NX_EVAL, NT_EVAL).
    """
    import numpy as np

    L = X_MAX - X_MIN
    dx = L / N_FFT
    x_ref = X_MIN + np.arange(N_FFT) * dx
    k = 2.0 * np.pi * np.fft.fftfreq(N_FFT, d=dx)

    psi = (2.0 / np.cosh(x_ref)).astype(np.complex128)

    # split-step Strang: meio passo linear, passo nao-linear, meio passo linear
    # linear: psi_t = i * 0.5 * psi_xx   ->  no Fourier: exp(-i dt/2 * 0.5 k^2)
    lin_half = np.exp(-1j * (DT_REF / 2.0) * 0.5 * (k ** 2))

    psi_grid = np.zeros((NX_EVAL, NT_EVAL), dtype=np.complex128)

    def project(psi_now):
        re = np.interp(x_eval, x_ref, psi_now.real)
        im = np.interp(x_eval, x_ref, psi_now.imag)
        return re + 1j * im

    psi_grid[:, 0] = project(psi)
    n_steps = int(math.ceil(T_MAX / DT_REF))
    next_idx = 1
    next_t = t_eval[next_idx]
    t_cur = 0.0
    for _ in range(n_steps):
        psi_hat = np.fft.fft(psi)
        psi_hat = lin_half * psi_hat
        psi = np.fft.ifft(psi_hat)
        psi = psi * np.exp(1j * DT_REF * (np.abs(psi) ** 2))
        psi_hat = np.fft.fft(psi)
        psi_hat = lin_half * psi_hat
        psi = np.fft.ifft(psi_hat)
        t_cur += DT_REF
        while next_idx < NT_EVAL and t_cur >= next_t - 1e-12:
            psi_grid[:, next_idx] = project(psi)
            next_idx += 1
            if next_idx < NT_EVAL:
                next_t = t_eval[next_idx]
            else:
                break
        if next_idx >= NT_EVAL:
            break

    return psi_grid


### ============= ### ###  Hipercubo Latino 2D  ### ###  ============= ###

def latin_hypercube(n_points, x_min, x_max, t_min, t_max, seed):
    import numpy as np
    rng = np.random.default_rng(seed)
    intervals = np.linspace(0.0, 1.0, n_points + 1)
    pts = np.zeros((n_points, 2))
    for i in range(2):
        u = intervals[:-1] + (intervals[1] - intervals[0]) * rng.random(n_points)
        perm = rng.permutation(n_points)
        pts[:, i] = u[perm]
    pts[:, 0] = pts[:, 0] * (x_max - x_min) + x_min
    pts[:, 1] = pts[:, 1] * (t_max - t_min) + t_min
    return pts


### ============= ### ###  Funcao remota  ### ###  ============= ###

@app.function(volumes={VOLUME_PATH: volume}, timeout=20 * 60)
def build_dataset():
    import numpy as np

    np.random.seed(SEED)

    # Grades de avaliacao
    x_eval = np.linspace(X_MIN, X_MAX, NX_EVAL)
    t_eval = np.linspace(T_MIN, T_MAX, NT_EVAL)

    print("[preprocess] computando referencia split-step Fourier...")
    psi_ref = split_step_reference(x_eval, t_eval)
    u_ref = psi_ref.real.astype(np.float32)
    v_ref = psi_ref.imag.astype(np.float32)
    abs_psi_ref = np.abs(psi_ref).astype(np.float32)
    print(f"[preprocess] |psi| ref: min={abs_psi_ref.min():.3f} max={abs_psi_ref.max():.3f}")

    # Pontos interiores via Hipercubo Latino
    treino_int = latin_hypercube(N_INT, X_MIN, X_MAX, T_MIN, T_MAX, seed=SEED)

    # Pontos da condicao inicial (t=0)
    rng_ic = np.random.default_rng(SEED + 1)
    x_ic = rng_ic.uniform(X_MIN, X_MAX, size=(N_IC, 1))
    t_ic = np.zeros_like(x_ic)
    u_ic_ref = 2.0 / np.cosh(x_ic)
    v_ic_ref = np.zeros_like(u_ic_ref)
    treino_ci = np.concatenate(
        [x_ic, t_ic, u_ic_ref, v_ic_ref], axis=1
    )  # colunas: x, t, Re(psi), Im(psi)

    # Pontos das condicoes de contorno (x=+/-5)
    rng_bc = np.random.default_rng(SEED + 2)
    t_bc = rng_bc.uniform(T_MIN, T_MAX, size=(N_BC, 1))
    x_bc_left = np.full_like(t_bc, X_MIN)
    x_bc_right = np.full_like(t_bc, X_MAX)
    treino_cc = np.concatenate(
        [
            np.concatenate([x_bc_left, t_bc], axis=1),
            np.concatenate([x_bc_right, t_bc], axis=1),
        ],
        axis=0,
    )  # colunas: x, t

    # Metadados
    metadata = {
        "X_MIN": X_MIN, "X_MAX": X_MAX,
        "T_MIN": T_MIN, "T_MAX": T_MAX,
        "N_INT": N_INT, "N_IC": N_IC, "N_BC": N_BC,
        "N_FFT": N_FFT, "DT_REF": DT_REF,
        "NX_EVAL": NX_EVAL, "NT_EVAL": NT_EVAL,
        "SEED": SEED,
    }

    out = Path(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / "treino.npz",
        treino_int=treino_int,
        treino_ci=treino_ci,
        treino_cc=treino_cc,
    )
    np.savez(
        out / "referencia.npz",
        x=x_eval.astype(np.float32),
        t=t_eval.astype(np.float32),
        u_ref=u_ref,
        v_ref=v_ref,
        abs_psi_ref=abs_psi_ref,
    )
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    volume.commit()

    print(f"[preprocess] Hipercubo Latino: {treino_int.shape[0]} pontos interiores")
    print(f"[preprocess] Condicao inicial: {treino_ci.shape[0]} pontos em t=0")
    print(f"[preprocess] Condicao de contorno: {treino_cc.shape[0]} pontos em x=+/-5")
    print(f"[preprocess] Referencia: grade {NX_EVAL} x {NT_EVAL}")
    print(f"[preprocess] Saida em {OUT_DIR}")
    return metadata


@app.local_entrypoint()
def main():
    meta = build_dataset.remote()
    print("[preprocess] metadata:", json.dumps(meta, indent=2))
