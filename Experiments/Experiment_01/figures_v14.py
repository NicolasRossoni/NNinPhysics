"""Figuras Kovasznay v14: campos extrap (Fig 4) + loss decomp (Fig 5)."""
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RD = HERE / "results_v14"
OUT = HERE.parent.parent.parent / "reports" / "latex" / "tcc" / "figures"

X_MIN, X_TR, X_MAX = -0.5, 1.0, 2.5
Y_MIN, Y_MAX = -0.5, 1.5


def main():
    pinn = json.loads((RD / "pinn_4x32_nsup_s21.json").read_text())
    mix = json.loads((RD / "mix_3x1_nsup_s21.json").read_text())

    gp = pinn["extrap_x_grid"]; gm = mix["extrap_x_grid"]
    x_full = np.array(gp["x_lin"]); y_full = np.array(gp["y_lin"])
    u_an = np.array(gp["u_analytic"])
    u_pinn = np.array(gp["u_pred"])
    u_mix = np.array(gm["u_pred"])

    fig = plt.figure(figsize=(13, 12))
    gs = fig.add_gridspec(4, 4, height_ratios=[1, 1, 1, 1.2], hspace=0.4, wspace=0.3)
    titles = ["Analítica", "PINN $4{\\times}32$", "MixFunn $3{\\times}1$"]
    datas = [u_an, u_pinn, u_mix]
    vmin = min(d.min() for d in datas); vmax = max(d.max() for d in datas)
    x_cuts = [0.0, 0.75, 1.25, 2.0]

    for i, (data, title) in enumerate(zip(datas, titles)):
        ax = fig.add_subplot(gs[i, :])
        im = ax.pcolormesh(x_full, y_full, data, cmap="viridis", vmin=vmin, vmax=vmax, shading="auto")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title(title)
        ax.axvline(X_TR, color="white", linestyle=":", linewidth=1.4, label="limite treino $x=1$")
        for k, xc in enumerate(x_cuts):
            ax.axvline(xc, color="red", linestyle="--", linewidth=0.8, alpha=0.7,
                        label="cortes" if k == 0 else None)
        if i == 0:
            ax.legend(fontsize=8, loc="upper left", framealpha=0.7)
        plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="u")

    for j, xc in enumerate(x_cuts):
        ax = fig.add_subplot(gs[3, j])
        ix = int(np.argmin(np.abs(x_full - xc)))
        ax.plot(y_full, u_an[:, ix], "k-", linewidth=1.6, label="Analítica")
        ax.plot(y_full, u_pinn[:, ix], "b--", linewidth=1.2, label="PINN")
        ax.plot(y_full, u_mix[:, ix], color="orange", linestyle="--", linewidth=1.2, label="MixFunn")
        ax.set_xlabel("y")
        if j == 0:
            ax.set_ylabel("u")
        zone = "treino" if xc <= X_TR else "extrap"
        ax.set_title(f"corte $x={xc:g}$ ({zone})", fontsize=10)
        ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="best", framealpha=0.7)

    plt.suptitle("Kovasznay --- campos $u$ com BC Coons (sem leak), $x$ estendido até $2{,}5$", fontsize=11)
    out_path = OUT / "kov_v12_fields_extrap.png"
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"saved {out_path}")

    # Figure 2: loss decomp
    sup = json.loads((RD / "pinn_4x32_sup_s21.json").read_text())
    nsup = pinn
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.semilogy(sup["epochs_log"], sup["l2_val_curve"], "b-", label="PINN supervisionada", linewidth=1.5)
    ax1.semilogy(nsup["epochs_log"], nsup["l2_val_curve"], "r-", label="PINN não-supervisionada", linewidth=1.5)
    ax1.set_xlabel("Iterações"); ax1.set_ylabel("$L^2$ relativo na validação")
    ax1.set_title("(a) Convergência: sup vs não-sup (PINN $4{\\times}32$, seed 21, BC Coons)")
    ax1.grid(alpha=0.4); ax1.legend()
    ax2.semilogy(nsup["epochs_log"], nsup["loss_momento"], color="purple",
                  label="$L_{Momento}$ (Eq.1 + Eq.2)", linewidth=1.5)
    ax2.semilogy(nsup["epochs_log"], nsup["loss_massa"], color="darkgreen",
                  label="$L_{Massa}$ (Eq.3)", linewidth=1.5)
    ax2.set_xlabel("Iterações"); ax2.set_ylabel("Loss componente")
    ax2.set_title("(b) PINN não-sup: loss decomposta em conservações")
    ax2.grid(alpha=0.4); ax2.legend()
    plt.tight_layout()
    out2 = OUT / "kov_v12_loss_decomp.png"
    plt.savefig(out2, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"saved {out2}")


if __name__ == "__main__":
    main()
