#### Importando Bibliotecas
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

### ============= ### ###  Caminhos  ### ###  ============= ###

PARENT = Path(__file__).resolve().parent
CHECKPOINTS = PARENT / "tmp_checkpoints"        # baixado via `modal volume get`
OUT_PNG = PARENT / "burgers_v25.png"

PINN_LABEL = "pinn_6x64"
MIX_LABEL  = "mix_3x3_sof"


### ============= ### ###  Utilitarios  ### ###  ============= ###

def _localize(path: Path) -> Path:
    """Procura o arquivo em tmp_checkpoints/exp_05/ ou tmp_checkpoints/ direto."""
    candidatos = [
        CHECKPOINTS / "exp_05" / path,
        CHECKPOINTS / path,
        CHECKPOINTS / "checkpoints" / "exp_05" / path,
    ]
    for c in candidatos:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Nao encontrado: {path}. Procurei em {[str(c) for c in candidatos]}"
    )


def _fmt_l2(l2: float) -> str:
    """Formata L2 em notacao cientifica PT-BR (ex.: 1,10e-02)."""
    s = f"{l2:.2e}"                              # ex.: '1.10e-02'
    mant, exp = s.split("e")
    mant_pt = mant.replace(".", ",")
    exp_int = int(exp)
    return rf"{mant_pt} \times 10^{{{exp_int}}}"


### ============= ### ###  Carregando Resultados  ### ###  ============= ###

pinn_json = json.loads(_localize(Path(f"{PINN_LABEL}.json")).read_text())
mix_json  = json.loads(_localize(Path(f"{MIX_LABEL}.json")).read_text())

pinn_pred = np.load(_localize(Path(f"{PINN_LABEL}_pred.npz")))
mix_pred  = np.load(_localize(Path(f"{MIX_LABEL}_pred.npz")))

x_eval = pinn_pred["x_eval"]
t_eval = pinn_pred["t_eval"]
u_ref  = pinn_pred["u_ref"]
u_pinn = pinn_pred["u_pred"]
u_mix  = mix_pred["u_pred"]

l2_pinn = float(pinn_json["l2_final"])
l2_mix  = float(mix_json["l2_final"])

print(f"[analyze] PINN  6x64   L2 = {l2_pinn:.3e}   (n_params = {pinn_json['n_params']})")
print(f"[analyze] Mix   3x3 sof L2 = {l2_mix:.3e}   (n_params = {mix_json['n_params']})")

### ============= ### ###  Plot: 3 heatmaps  ### ###  ============= ###
# Layout: referencia | PINN | MixFunn_sof  (todos com colorbar individual)

# Eixos: t (horizontal) x (vertical), como na figura do monografia
T, X = np.meshgrid(t_eval, x_eval, indexing="xy")  # X tem shape (len(x), len(t))
extent = [t_eval[0], t_eval[-1], x_eval[0], x_eval[-1]]

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
cmap = "RdBu_r"
vmin, vmax = -1.0, 1.0

titulos = [
    r"Solução de referência (espectral)",
    rf"PINN $6 \times 64$ ($L^2 = {_fmt_l2(l2_pinn)}$)",
    rf"MixFunn$_{{\rm sof}}$ $3 \times 3$ ($L^2 = {_fmt_l2(l2_mix)}$)",
]
campos = [u_ref, u_pinn, u_mix]

for ax, campo, titulo in zip(axes, campos, titulos):
    im = ax.imshow(
        campo, origin="lower", aspect="auto", cmap=cmap,
        vmin=vmin, vmax=vmax, extent=extent,
    )
    ax.set_title(titulo)
    ax.set_xlabel("$t$")
    ax.set_ylabel("$x$")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("$u$")

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
print(f"[analyze] figura salva em {OUT_PNG}")
