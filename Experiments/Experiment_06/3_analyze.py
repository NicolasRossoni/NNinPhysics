#### Importando Bibliotecas
import json
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

### ============= ### ###  Localizacao dos Checkpoints  ### ###  ============= ###

HERE = Path(__file__).resolve().parent
CHECKPOINTS_LOCAL = HERE / "tmp_checkpoints" / "by_label"
FIG_OUT = HERE / "schrod_v25.png"

LABEL_PINN = "pinn_8x100_nsup"
LABEL_MIX = "mix_3x6_sof_nsup"

T_CUTS = [0.25, 0.5, 0.65, 0.8]


### ============= ### ###  Formatacao em Notacao Cientifica  ### ###  ============= ###

def fmt_sci(x, casas=2):
    if x is None or not math.isfinite(x):
        return "---"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10 ** e
    s = f"{m:.{casas}f}".replace(".", ",")
    return f"{s} \\times 10^{{{e}}}"


### ============= ### ###  Leitura dos Resultados  ### ###  ============= ###

def carregar(label):
    p_json = CHECKPOINTS_LOCAL / f"{label}.json"
    p_npz = CHECKPOINTS_LOCAL / f"{label}_pred.npz"
    if not p_json.exists() or not p_npz.exists():
        return None
    meta = json.loads(p_json.read_text())
    data = np.load(p_npz)
    return {
        "meta": meta,
        "u_pred": data["u_pred"],
        "v_pred": data["v_pred"],
        "abs_psi_pred": data["abs_psi_pred"],
        "u_ref": data["u_ref"],
        "v_ref": data["v_ref"],
        "abs_psi_ref": data["abs_psi_ref"],
        "x": data["x"],
        "t": data["t"],
    }


### ============= ### ###  Geracao da Figura  ### ###  ============= ###

def gerar_figura(pinn, mix):
    """3 heatmaps (ref / PINN / MixFunn) + 4 cortes em t."""
    x = pinn["x"]
    t = pinn["t"]
    abs_ref = pinn["abs_psi_ref"]            # (NX, NT)
    abs_pinn = pinn["abs_psi_pred"]
    abs_mix = mix["abs_psi_pred"]

    l2_pinn = pinn["meta"]["l2_val"]
    l2_mix = mix["meta"]["l2_val"]

    vmin = 0.0
    vmax = float(max(abs_ref.max(), abs_pinn.max(), abs_mix.max()))

    fig = plt.figure(figsize=(13, 8))
    gs = gridspec.GridSpec(
        2, 4, height_ratios=[1.0, 0.85],
        hspace=0.45, wspace=0.32,
        left=0.06, right=0.97, top=0.94, bottom=0.08,
    )

    # Top: 3 heatmaps em colunas 0..2, colorbar na coluna 3
    extent = [t.min(), t.max(), x.min(), x.max()]
    titulos = [
        "Solução de referência (SSF)",
        f"PINN $8\\times 100$ ($L^2 = {fmt_sci(l2_pinn)}$)",
        f"MixFunn$_{{\\rm sof}}$ $3\\times 6$ ($L^2 = {fmt_sci(l2_mix)}$)",
    ]
    dados = [abs_ref, abs_pinn, abs_mix]
    eixos_heat = []
    im_ref = None
    for i, (titulo, dado) in enumerate(zip(titulos, dados)):
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(
            dado, origin="lower", aspect="auto",
            extent=extent, vmin=vmin, vmax=vmax, cmap="viridis",
        )
        if i == 0:
            im_ref = im
        ax.set_xlabel("$t$")
        ax.set_ylabel("$x$")
        ax.set_title(titulo, fontsize=10)
        # Linhas dos cortes (magenta tracejada)
        for tc_val in T_CUTS:
            ax.axvline(tc_val, color="magenta", lw=0.7, ls="--", alpha=0.7)
        eixos_heat.append(ax)
    # Tag "cortes" no primeiro heatmap (canto superior direito)
    eixos_heat[0].text(
        0.97, 0.95, "cortes", transform=eixos_heat[0].transAxes,
        fontsize=8, color="black", ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.25", fc="white",
                  ec="magenta", lw=0.8, alpha=0.9),
    )

    # Colorbar na quarta celula superior
    ax_cb_holder = fig.add_subplot(gs[0, 3])
    ax_cb_holder.set_visible(False)
    bbox = ax_cb_holder.get_position()
    cax = fig.add_axes([bbox.x0 + 0.01, bbox.y0, 0.015, bbox.height])
    fig.colorbar(im_ref, cax=cax, label="$|\\psi|$")

    # Bottom: 4 cortes em t
    for k, t_cut in enumerate(T_CUTS):
        ax = fig.add_subplot(gs[1, k])
        it = int(np.argmin(np.abs(t - t_cut)))
        ax.plot(x, abs_ref[:, it], color="C0", lw=2.0, label="Analítica")
        ax.plot(x, abs_pinn[:, it], color="orange", lw=1.4, ls="--", label="PINN")
        ax.plot(x, abs_mix[:, it], color="green", lw=1.4, ls=":",
                label="MixFunn$_{\\rm sof}$")
        ax.set_title(f"$t = {t_cut:g}$", fontsize=10)
        ax.set_xlabel("$x$")
        if k == 0:
            ax.set_ylabel("$|\\psi|$")
            ax.legend(loc="upper right", fontsize=8, frameon=True)
        ax.grid(alpha=0.3)
        ax.set_xlim(x.min(), x.max())

    fig.savefig(FIG_OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[analyze] figura salva em {FIG_OUT}")


### ============= ### ###  Impressao do Resultado  ### ###  ============= ###

def main():
    if not CHECKPOINTS_LOCAL.exists():
        raise FileNotFoundError(
            f"Diretorio {CHECKPOINTS_LOCAL} nao existe. Baixe primeiro com:\n"
            f"  modal volume get tcc /checkpoints/exp_06 ./tmp_checkpoints"
        )

    pinn = carregar(LABEL_PINN)
    mix = carregar(LABEL_MIX)
    if pinn is None:
        raise FileNotFoundError(f"checkpoint ausente: {LABEL_PINN}")
    if mix is None:
        raise FileNotFoundError(f"checkpoint ausente: {LABEL_MIX}")

    print()
    print("Figura 9 — Schrödinger não-linear não-supervisionada")
    print("=" * 70)
    print(f"{'Configuracao':<22} | {'L2|psi|':<10} | {'n_params':<8} | wall (s)")
    print("-" * 70)
    for nome, r in [("PINN 8x100",  pinn), ("MixFunn-sof 3x6", mix)]:
        m = r["meta"]
        print(f"{nome:<22} | {m['l2_val']:.3e} | {m['n_params']:<8} | "
              f"{m['wall_clock']:.0f}")
    print("=" * 70)
    print()

    gerar_figura(pinn, mix)


if __name__ == "__main__":
    main()
