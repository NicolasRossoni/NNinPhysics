#### Importando Bibliotecas
import json
import math
from pathlib import Path

import modal

### ============= ### ###  Hiperparametros do Pre-Processamento  ### ###  ============= ###

SEED = 22                                       # semente unica usada em todo o experimento

# Dominio fisico
X_MIN, X_MAX = -1.0, 1.0
T_MIN, T_MAX =  0.0, 1.0

# Viscosidade (Burgers)
NU = 0.01 / math.pi

# Pontos de colocacao (Hipercubo Latino) e condicoes inicial/contorno
N_COL = 10000                                   # pontos interiores
N_CI  = 200                                     # condicao inicial u(x, 0) = -sin(pi x)
N_BC  = 200                                     # condicao de contorno u(+-1, t) = 0

# Malha de avaliacao L2 (saida do espectral interpolada)
NX_EVAL = 200
NT_EVAL = 100

# Resolucao da solucao de referencia espectral + RK4
N_SPECTRAL = 256
DT_SPECTRAL = 1e-4

### ============= ### ###  Imagem e Volume Modal  ### ###  ============= ###

PARENT = Path(__file__).resolve().parent
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.5.1", "numpy", "scipy"
)
app = modal.App("nnphysics-exp05-preprocess", image=image)
volume = modal.Volume.from_name("tcc", create_if_missing=True)

VOLUME_PATH = "/data"
PREPROCESS_DIR = "preprocess/exp_05"


### ============= ### ###  Solucao de Referencia (Espectral + RK4)  ### ###  ============= ###

def _spectral_reference(N: int, dt: float):
    """Burgers viscoso 1D: pseudo-espectral Fourier + RK4 explicito.

    Periodico em x in [X_MIN, X_MAX). RK4 com passo dt ate t = T_MAX.
    """
    import numpy as np
    L = X_MAX - X_MIN
    x = np.linspace(X_MIN, X_MAX, N, endpoint=False)
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=L / N)
    ik = 1j * k
    k2 = -(k ** 2)

    def rhs(u):
        uhat = np.fft.fft(u)
        ux = np.real(np.fft.ifft(ik * uhat))
        uxx = np.real(np.fft.ifft(k2 * uhat))
        return -u * ux + NU * uxx

    nt = int(round((T_MAX - T_MIN) / dt)) + 1
    t_grid = np.linspace(T_MIN, T_MAX, nt)
    u = np.empty((nt, N), dtype=np.float64)
    u[0] = -np.sin(np.pi * x)
    cur = u[0].copy()
    for n in range(nt - 1):
        k1 = rhs(cur)
        k2_ = rhs(cur + 0.5 * dt * k1)
        k3 = rhs(cur + 0.5 * dt * k2_)
        k4 = rhs(cur + dt * k3)
        cur = cur + (dt / 6.0) * (k1 + 2 * k2_ + 2 * k3 + k4)
        u[n + 1] = cur
    return x, t_grid, u


def _build_reference_grid():
    """Interpola a referencia espectral para a malha de avaliacao (x_eval, t_eval)."""
    import numpy as np
    from scipy.interpolate import RegularGridInterpolator

    x_sp, t_sp, u_sp = _spectral_reference(N_SPECTRAL, DT_SPECTRAL)
    # Estende para incluir o ponto x = X_MAX (periodico)
    x_ext = np.concatenate([x_sp, [X_MAX]])
    u_ext = np.concatenate([u_sp, u_sp[:, 0:1]], axis=1)
    interp = RegularGridInterpolator(
        (t_sp, x_ext), u_ext, bounds_error=False, fill_value=0.0
    )
    x_eval = np.linspace(X_MIN, X_MAX, NX_EVAL)
    t_eval = np.linspace(T_MIN, T_MAX, NT_EVAL)
    XE, TE = np.meshgrid(x_eval, t_eval, indexing="ij")
    pts = np.stack([TE.ravel(), XE.ravel()], axis=1)
    u_ref = interp(pts).reshape(NX_EVAL, NT_EVAL)
    return x_eval, t_eval, u_ref


### ============= ### ###  Amostragem Latin-Hypercube 2D  ### ###  ============= ###

def latin_hypercube(n_points, x_min, x_max, t_min, t_max, seed):
    """Latin-Hypercube estratificado em [0,1)^2 reescalado para o dominio."""
    import numpy as np
    rng = np.random.default_rng(seed)
    cut = np.linspace(0.0, 1.0, n_points + 1)
    a, b = cut[:n_points], cut[1:]
    u = rng.random((n_points, 2))
    pts = a[:, None] + u * (b - a)[:, None]
    for j in range(2):
        rng.shuffle(pts[:, j])
    pts[:, 0] = pts[:, 0] * (x_max - x_min) + x_min
    pts[:, 1] = pts[:, 1] * (t_max - t_min) + t_min
    return pts


### ============= ### ###  Geracao dos Dados no Modal  ### ###  ============= ###

@app.function(volumes={VOLUME_PATH: volume}, timeout=15 * 60)
def gerar_dados():
    import numpy as np
    import torch as tc

    tc.manual_seed(SEED)
    np.random.seed(SEED)

    ### ============= ### ###  Solucao de Referencia  ### ###  ============= ###

    print("[preprocess] integrando referencia espectral + RK4...", flush=True)
    x_eval, t_eval, u_ref = _build_reference_grid()
    print(f"[preprocess] referencia em malha {NX_EVAL}x{NT_EVAL}", flush=True)

    ### ============= ### ###  Pontos de Colocacao (interiores)  ### ###  ============= ###

    treino_colocacao = latin_hypercube(N_COL, X_MIN, X_MAX, T_MIN, T_MAX, seed=SEED)

    ### ============= ### ###  Condicao Inicial e Contorno  ### ###  ============= ###
    # Pontos da condicao inicial u(x, 0) = -sin(pi x)
    x_ci = np.linspace(X_MIN, X_MAX, N_CI).reshape(-1, 1)
    t_ci = np.zeros_like(x_ci)
    xt_ci = np.concatenate([x_ci, t_ci], axis=1)
    u_ci = -np.sin(np.pi * x_ci)

    # Pontos da condicao de contorno u(+-1, t) = 0 (metade em cada borda)
    n_bc_meio = N_BC // 2
    t_bc = np.linspace(T_MIN, T_MAX, n_bc_meio).reshape(-1, 1)
    xt_bc_esq = np.concatenate([np.full_like(t_bc, X_MIN), t_bc], axis=1)
    xt_bc_dir = np.concatenate([np.full_like(t_bc, X_MAX), t_bc], axis=1)
    xt_bc = np.concatenate([xt_bc_esq, xt_bc_dir], axis=0)

    ### ============= ### ###  Persistencia no Volume  ### ###  ============= ###

    out_dir = Path(VOLUME_PATH) / PREPROCESS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_dir / "dataset.npz",
        x_eval=x_eval.astype(np.float64),
        t_eval=t_eval.astype(np.float64),
        u_ref=u_ref.astype(np.float64),
        treino_colocacao=treino_colocacao.astype(np.float64),
        xt_ci=xt_ci.astype(np.float64),
        u_ci=u_ci.astype(np.float64),
        xt_bc=xt_bc.astype(np.float64),
    )

    metadata = {
        "seed": SEED,
        "x_min": X_MIN, "x_max": X_MAX,
        "t_min": T_MIN, "t_max": T_MAX,
        "nu": NU,
        "n_col": N_COL,
        "n_ci": N_CI,
        "n_bc": N_BC,
        "nx_eval": NX_EVAL,
        "nt_eval": NT_EVAL,
        "n_spectral": N_SPECTRAL,
        "dt_spectral": DT_SPECTRAL,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    volume.commit()

    print(f"[preprocess] {N_COL} pontos LHS salvos em {PREPROCESS_DIR}/", flush=True)
    print(f"[preprocess] CI: {N_CI} pontos | BC: {N_BC} pontos", flush=True)
    return metadata


@app.local_entrypoint()
def main():
    meta = gerar_dados.remote()
    print(f"[preprocess] concluido: {meta}", flush=True)
