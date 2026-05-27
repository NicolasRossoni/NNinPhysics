#### Importando Bibliotecas
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

### ============= ### ###  Hiperparametros  ### ###  ============= ###

PARENT = Path(__file__).resolve().parent
DEFAULT_CHK_DIR = PARENT / "tmp_checkpoints" / "exp_07"

LABEL_PINN = "pinn_8x96_nsup"
LABEL_MIX = "mix_1x2_sof_nsup"

OUT_FIG = PARENT / "baldan_v25.png"


### ============= ### ###  L2 relativo por componente  ### ###  ============= ###

def l2_rel(pred: np.ndarray, ref: np.ndarray) -> float:
    num = float(((pred - ref) ** 2).sum())
    den = float((ref ** 2).sum()) + 1e-20
    return (num / den) ** 0.5


### ============= ### ###  Figura: 2 linhas (H_x, H_z) x 3 colunas  ### ###  ============= ###

def make_figure(data_pinn: dict, data_mix: dict, out_path: Path,
                l2_pinn_hx: float, l2_mix_hx: float):
    Xg = data_pinn["x"]; Zg = data_pinn["z"]
    Hx_ref = data_pinn["Hx_ref"]; Hz_ref = data_pinn["Hz_ref"]
    Hx_pinn = data_pinn["Hx_pred"]; Hz_pinn = data_pinn["Hz_pred"]
    Hx_mix = data_mix["Hx_pred"]; Hz_mix = data_mix["Hz_pred"]

    # Escala comum por componente (referencia define vmin/vmax)
    vmax_x = float(max(abs(Hx_ref).max(), abs(Hx_pinn).max(), abs(Hx_mix).max()))
    vmax_z = float(max(abs(Hz_ref).max(), abs(Hz_pinn).max(), abs(Hz_mix).max()))

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)

    cols = [
        ("Solução de referência (FD)", Hx_ref, Hz_ref),
        (f"PINN 8x96 lift ($L^2_{{H_x}} = {l2_pinn_hx:.2e}$)", Hx_pinn, Hz_pinn),
        (f"MixFunn$_{{\\rm sof}}$ 1x2 ($L^2_{{H_x}} = {l2_mix_hx:.2e}$)",
         Hx_mix, Hz_mix),
    ]

    for j, (title, hx, hz) in enumerate(cols):
        ax = axes[0, j]
        im = ax.pcolormesh(Xg, Zg, hx, cmap="viridis",
                           vmin=-vmax_x, vmax=vmax_x, shading="auto")
        ax.set_title(title)
        ax.set_xlabel("x"); ax.set_ylabel("z")
        ax.set_aspect("equal")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("$H_x$")

        ax = axes[1, j]
        im = ax.pcolormesh(Xg, Zg, hz, cmap="viridis",
                           vmin=-vmax_z, vmax=vmax_z, shading="auto")
        ax.set_xlabel("x"); ax.set_ylabel("z")
        ax.set_aspect("equal")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("$H_z$")

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


### ============= ### ###  Entry point  ### ###  ============= ###

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chk-dir",
        type=Path,
        default=DEFAULT_CHK_DIR,
        help="Diretorio contendo os arquivos .json/.npz baixados do volume Modal.",
    )
    args = parser.parse_args()
    chk_dir = args.chk_dir

    # Permite passar o pai (tmp_checkpoints) ou o subdir (tmp_checkpoints/exp_07)
    if not (chk_dir / f"{LABEL_PINN}.npz").exists():
        for cand in [chk_dir / "exp_07", chk_dir / "checkpoints" / "exp_07"]:
            if (cand / f"{LABEL_PINN}.npz").exists():
                chk_dir = cand
                break

    print(f"[analyze] usando chk_dir={chk_dir}")

    # Carrega registros JSON com metricas finais
    with open(chk_dir / f"{LABEL_PINN}.json") as fh:
        rec_pinn = json.load(fh)
    with open(chk_dir / f"{LABEL_MIX}.json") as fh:
        rec_mix = json.load(fh)

    print(f"[analyze] PINN  L2_Hx = {rec_pinn['l2_hx']:.3e}  "
          f"L2_Hz = {rec_pinn['l2_hz']:.3e}  n_params = {rec_pinn['n_params']}")
    print(f"[analyze] Mix   L2_Hx = {rec_mix['l2_hx']:.3e}  "
          f"L2_Hz = {rec_mix['l2_hz']:.3e}  n_params = {rec_mix['n_params']}")

    # Carrega campos
    data_pinn = dict(np.load(chk_dir / f"{LABEL_PINN}.npz"))
    data_mix = dict(np.load(chk_dir / f"{LABEL_MIX}.npz"))

    # Recalcula L2 a partir dos arrays (sanity check independente do registro)
    l2_pinn_hx = l2_rel(data_pinn["Hx_pred"], data_pinn["Hx_ref"])
    l2_pinn_hz = l2_rel(data_pinn["Hz_pred"], data_pinn["Hz_ref"])
    l2_mix_hx = l2_rel(data_mix["Hx_pred"], data_mix["Hx_ref"])
    l2_mix_hz = l2_rel(data_mix["Hz_pred"], data_mix["Hz_ref"])

    print(f"[analyze] (recalculado) PINN  L2_Hx = {l2_pinn_hx:.3e}  "
          f"L2_Hz = {l2_pinn_hz:.3e}")
    print(f"[analyze] (recalculado) Mix   L2_Hx = {l2_mix_hx:.3e}  "
          f"L2_Hz = {l2_mix_hz:.3e}")

    make_figure(data_pinn, data_mix, OUT_FIG, l2_pinn_hx, l2_mix_hx)
    print(f"[analyze] figura salva em {OUT_FIG}")

    # Saida JSON resumida
    summary = {
        "pinn_8x96_nsup": {
            "l2_hx": l2_pinn_hx, "l2_hz": l2_pinn_hz,
            "n_params": rec_pinn["n_params"],
            "wall_clock": rec_pinn["wall_clock"],
        },
        "mix_1x2_sof_nsup": {
            "l2_hx": l2_mix_hx, "l2_hz": l2_mix_hz,
            "n_params": rec_mix["n_params"],
            "wall_clock": rec_mix["wall_clock"],
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
