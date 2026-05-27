#### Importando Bibliotecas
import json
from pathlib import Path

import modal

### ============= ### ###  Hiperparametros  ### ###  ============= ###

# Dominio fisico (quadrado unitario)
X_MIN, X_MAX = 0.0, 1.0
Z_MIN, Z_MAX = 0.0, 1.0

# Disco paramagnetico
DISC_CX, DISC_CZ = 0.5, 0.5
DISC_R = 0.2
MU_OUT, MU_DISC = 1.0, 3.0
SIGMOID_ALPHA = 500.0   # nitidez da transicao mu_r entre exterior e disco

# Resolucao da malha FD para a referencia numerica + grade de avaliacao
NGRID = 100

# Seed (precisa ser determinista entre 1_preprocess e 2_train)
SEED = 21

### ============= ### ###  Modal App  ### ###  ============= ###

app = modal.App(
    "nnphysics-exp07-preprocess",
    image=modal.Image.debian_slim(python_version="3.11").pip_install(
        "numpy", "scipy"
    ),
)
volume = modal.Volume.from_name("tcc", create_if_missing=True)
VOLUME_PATH = "/data"
OUT_DIR = "/data/preprocess/exp_07"


### ============= ### ###  Referencia FD via potencial escalar  ### ###  ============= ###

# Resolve-se Laplace generalizado para o potencial escalar magnetico phi:
#     div( mu grad(phi) ) = 0,    phi = -z no bordo,
# com mu(x,z) sigmoidal entre MU_OUT=1 (exterior) e MU_DISC=3 (disco). A
# imposicao phi = -z no bordo equivale a H = -grad(phi) = (0, 0, 1) ali.
# Discretizacao FD de 5 pontos com media harmonica de mu nas faces. Em
# seguida calcula-se H_x = -d_x(phi) e H_z = -d_z(phi) por diferencas
# centrais (laterais nas bordas).
def fd_reference(ngrid: int):
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    nx = nz = ngrid
    x = np.linspace(X_MIN, X_MAX, nx)
    z = np.linspace(Z_MIN, Z_MAX, nz)
    dx = x[1] - x[0]
    dz = z[1] - z[0]
    Xg, Zg = np.meshgrid(x, z, indexing="xy")

    def k_of(ix, iz):
        return iz * nx + ix

    N = nx * nz
    R = np.sqrt((Xg - DISC_CX) ** 2 + (Zg - DISC_CZ) ** 2)
    s = 1.0 / (1.0 + np.exp(-SIGMOID_ALPHA * (DISC_R - R)))
    mu = MU_OUT + (MU_DISC - MU_OUT) * s
    mu_xface = 2.0 * mu[:, :-1] * mu[:, 1:] / (mu[:, :-1] + mu[:, 1:])
    mu_zface = 2.0 * mu[:-1, :] * mu[1:, :] / (mu[:-1, :] + mu[1:, :])

    rows, cols, vals = [], [], []
    rhs = np.zeros(N)

    def add(r, c, v):
        rows.append(r); cols.append(c); vals.append(v)

    for iz in range(nz):
        for ix in range(nx):
            k = k_of(ix, iz)
            if ix == 0 or ix == nx - 1 or iz == 0 or iz == nz - 1:
                add(k, k, 1.0)
                rhs[k] = -z[iz]
                continue
            mu_e = mu_xface[iz, ix]
            mu_w = mu_xface[iz, ix - 1]
            mu_n = mu_zface[iz, ix]
            mu_s = mu_zface[iz - 1, ix]
            add(k, k_of(ix + 1, iz), mu_e / (dx * dx))
            add(k, k_of(ix - 1, iz), mu_w / (dx * dx))
            add(k, k_of(ix, iz + 1), mu_n / (dz * dz))
            add(k, k_of(ix, iz - 1), mu_s / (dz * dz))
            add(k, k, -(mu_e + mu_w) / (dx * dx) - (mu_n + mu_s) / (dz * dz))
            rhs[k] = 0.0

    A = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
    phi = spla.spsolve(A, rhs).reshape(nz, nx)
    Hx = np.zeros_like(phi)
    Hz = np.zeros_like(phi)
    Hx[:, 1:-1] = -(phi[:, 2:] - phi[:, :-2]) / (2 * dx)
    Hx[:, 0] = -(phi[:, 1] - phi[:, 0]) / dx
    Hx[:, -1] = -(phi[:, -1] - phi[:, -2]) / dx
    Hz[1:-1, :] = -(phi[2:, :] - phi[:-2, :]) / (2 * dz)
    Hz[0, :] = -(phi[1, :] - phi[0, :]) / dz
    Hz[-1, :] = -(phi[-1, :] - phi[-2, :]) / dz
    return Xg, Zg, mu, Hx, Hz


### ============= ### ###  Hipercubo Latino 2D  ### ###  ============= ###

def latin_hypercube(n_points, x_min, x_max, z_min, z_max, seed):
    import numpy as np
    rng = np.random.default_rng(seed)
    intervals = np.linspace(0.0, 1.0, n_points + 1)
    points = np.zeros((n_points, 2))
    for i in range(2):
        u = intervals[:-1] + (intervals[1] - intervals[0]) * rng.random(n_points)
        perm = rng.permutation(n_points)
        points[:, i] = u[perm]
    points[:, 0] = points[:, 0] * (x_max - x_min) + x_min
    points[:, 1] = points[:, 1] * (z_max - z_min) + z_min
    return points


### ============= ### ###  Funcao remota  ### ###  ============= ###

# Pontos interiores para o residuo da EDP
N_INT = 8000


@app.function(volumes={VOLUME_PATH: volume}, timeout=15 * 60)
def build_dataset():
    import numpy as np

    np.random.seed(SEED)

    # Referencia FD sobre malha NGRID x NGRID
    Xg, Zg, mu_r_grid, Hx_ref, Hz_ref = fd_reference(NGRID)

    # Pontos interiores via Hipercubo Latino (residuo)
    treino = latin_hypercube(N_INT, X_MIN, X_MAX, Z_MIN, Z_MAX, seed=SEED)

    # Metadados
    metadata = {
        "X_MIN": X_MIN, "X_MAX": X_MAX,
        "Z_MIN": Z_MIN, "Z_MAX": Z_MAX,
        "DISC_CX": DISC_CX, "DISC_CZ": DISC_CZ, "DISC_R": DISC_R,
        "MU_OUT": MU_OUT, "MU_DISC": MU_DISC,
        "SIGMOID_ALPHA": SIGMOID_ALPHA,
        "NGRID": NGRID, "N_INT": N_INT, "SEED": SEED,
    }

    # Salvando no volume
    out = Path(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / "referencia.npz",
        x=Xg, z=Zg, mu_r=mu_r_grid,
        Hx_ref=Hx_ref, Hz_ref=Hz_ref,
    )
    np.savez(out / "treino.npz", treino=treino)
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    volume.commit()

    print(f"[preprocess] Hipercubo Latino: {treino.shape[0]} pontos interiores")
    print(f"[preprocess] Referencia FD: grade {NGRID} x {NGRID}")
    print(f"[preprocess] |Hx_ref| max = {abs(Hx_ref).max():.3e}, "
          f"|Hz_ref| max = {abs(Hz_ref).max():.3e}")
    print(f"[preprocess] Saida em {OUT_DIR}")
    return metadata


@app.local_entrypoint()
def main():
    meta = build_dataset.remote()
    print("[preprocess] metadata:", json.dumps(meta, indent=2))
