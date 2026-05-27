#### Importando Bibliotecas
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.spatial import cKDTree

### ============= ### ###  Hiperparametros  ### ###  ============= ###

# Diretorio local com o que foi baixado do volume Modal:
#   modal volume get tcc /checkpoints/exp_04 ./tmp_checkpoints
# Dentro dele esperamos by_label/<LABEL>.json e <LABEL>_pred.npz
HERE = Path(__file__).resolve().parent
CHECKPOINTS = HERE / "tmp_checkpoints" / "by_label"

# Tambem precisamos da nuvem de pontos ANEUMO (ground truth).
# Esperamos um arquivo case_AN4_m002.npz baixado de
#   modal volume get tcc /preprocess/exp_04/case_AN4_m002.npz ./tmp_checkpoints/
CASE_NPZ = HERE / "tmp_checkpoints" / "case_AN4_m002.npz"

# Mapa: rotulo na tabela do monograph -> label salvo pelo 2_train.py.
TABLE_ROWS = [
    ("PINN 5x64",        "0%",   1.80e-2, "pinn_5x64_unsup"),
    ("PINN 8x128",       "0%",   1.78e-2, "pinn_8x128_unsup"),
    ("MixFunn 3x1",      "0%",   1.92e-2, "mix_3x1_unsup"),
    ("MixFunn-sof 3x1",  "0%",   1.83e-2, "mix_3x1_sof_unsup"),
    ("PINN 5x64",        "25%",  1.33e-2, "pinn_5x64_semi_25"),
    ("PINN 5x64",        "50%",  1.33e-2, "pinn_5x64_semi_50"),
    ("MixFunn-sof 2x2",  "50%",  1.38e-2, "mix_2x2_sof_semi_50"),
    ("PINN 5x64",        "100%", 1.23e-3, "pinn_5x64_sup_full"),
    ("MixFunn-sof 2x2",  "100%", 3.87e-3, "mix_2x2_sof_sup_full"),
]

# Configs usadas nas figuras (Figura 6 = aneur_panel_3d, Figura 7 = aneur_panel_plane)
FIG_PANELS = [
    ("GT (ANEUMO)",                None),
    (r"PINN $5\times64$",           "pinn_5x64_sup_full"),
    (r"MixFunn$_{\rm sof}$ $2\times2$", "mix_2x2_sof_sup_full"),
    (r"PINN $8\times128$",          "pinn_8x128_unsup"),
]

### ============= ### ###  Carregando dados  ### ###  ============= ###

def load_record(label):
    j = CHECKPOINTS / f"{label}.json"
    n = CHECKPOINTS / f"{label}_pred.npz"
    rec = json.loads(j.read_text())
    pred = np.load(n)
    return rec, pred


def main():
    if not CASE_NPZ.exists():
        raise SystemExit(
            f"Arquivo {CASE_NPZ} nao encontrado. Baixe com:\n"
            f"  modal volume get tcc /preprocess/exp_04/case_AN4_m002.npz {CASE_NPZ.parent}/"
        )
    if not CHECKPOINTS.exists():
        raise SystemExit(
            f"Pasta {CHECKPOINTS} nao encontrada. Baixe com:\n"
            f"  modal volume get tcc /checkpoints/exp_04 {CHECKPOINTS.parent}"
        )

    # Ground truth
    d = np.load(CASE_NPZ)
    xyz = d["xyz"]; u_gt = d["u"]; v_gt = d["v"]; w_gt = d["w"]; p_gt = d["p"]
    speed_gt = np.sqrt(u_gt**2 + v_gt**2 + w_gt**2)
    bbox_min = xyz.min(0); bbox_max = xyz.max(0)
    center = (bbox_min + bbox_max) / 2
    print(f"GT: N={len(xyz)}, |u| range=[0,{speed_gt.max():.3f}] m/s, "
          f"p range=[{p_gt.min():.2f},{p_gt.max():.2f}] Pa")

    ### ============= ### ###  Tabela 5  ### ###  ============= ###

    print("\nTabela 5 — Variacoes de PINN e MixFunn no ANEUMO")
    print(f"{'Configuracao':22s} {'Sup':>5s}  {'MSE gerado':>12s}  {'MSE PDF':>10s}  {'Delta':>9s}")
    print("-" * 70)
    table_results = []
    for arch, sup, mse_ref, label in TABLE_ROWS:
        try:
            rec, _ = load_record(label)
            mse_gen = rec["mse_uvw"]
            delta = (mse_gen - mse_ref) / mse_ref * 100.0
            table_results.append((arch, sup, mse_gen, mse_ref, delta, True))
            print(f"{arch:22s} {sup:>5s}  {mse_gen:12.4e}  {mse_ref:10.2e}  {delta:+8.1f}%")
        except FileNotFoundError:
            table_results.append((arch, sup, None, mse_ref, None, False))
            print(f"{arch:22s} {sup:>5s}  {'MISSING':>12s}  {mse_ref:10.2e}  {'--':>9s}")

    ### ============= ### ###  Sub-amostragem do scatter 3D  ### ###  ============= ###

    rng = np.random.default_rng(0)
    sub = rng.choice(len(xyz), size=min(10000, len(xyz)), replace=False)

    # Limites de cor uniformes para todos os paineis: 99o percentil dos campos GT.
    speed_vmax = float(np.percentile(speed_gt, 99))
    p_vmax = float(np.percentile(np.abs(p_gt), 99))

    ### ============= ### ###  Figura 6: aneur_panel_3d.png  ### ###  ============= ###
    # Linha superior: 4 painéis 3D coloridos por |u|.
    # Linha inferior: 4 painéis 3D coloridos por p.
    fig = plt.figure(figsize=(20, 9))
    for col, (titulo, label) in enumerate(FIG_PANELS):
        if label is None:
            sp = speed_gt; pr = p_gt
        else:
            try:
                _, pred = load_record(label)
                sp = np.sqrt(pred["u_pred"]**2 + pred["v_pred"]**2 + pred["w_pred"]**2)
                pr = pred["p_pred"]
            except FileNotFoundError:
                sp = np.zeros_like(speed_gt); pr = np.zeros_like(p_gt)

        ax = fig.add_subplot(2, 4, col + 1, projection="3d")
        sc = ax.scatter(xyz[sub, 0], xyz[sub, 1], xyz[sub, 2],
                        c=sp[sub], cmap="viridis", s=2, alpha=0.7,
                        vmin=0, vmax=speed_vmax)
        ax.set_xlabel("X (mm)", fontsize=8); ax.set_ylabel("Y (mm)", fontsize=8)
        ax.set_zlabel("Z (mm)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title(rf"{titulo}: $|\mathbf{{u}}|$", fontsize=11)
        if col == 3:
            cb = fig.colorbar(sc, ax=ax, shrink=0.65, pad=0.10)
            cb.set_label("|u| (m/s)", fontsize=9)

        ax2 = fig.add_subplot(2, 4, col + 5, projection="3d")
        sc2 = ax2.scatter(xyz[sub, 0], xyz[sub, 1], xyz[sub, 2],
                          c=pr[sub], cmap="RdBu_r", s=2, alpha=0.7,
                          vmin=-p_vmax, vmax=p_vmax)
        ax2.set_xlabel("X (mm)", fontsize=8); ax2.set_ylabel("Y (mm)", fontsize=8)
        ax2.set_zlabel("Z (mm)", fontsize=8)
        ax2.tick_params(labelsize=7)
        ax2.set_title(rf"{titulo}: $p$", fontsize=11)
        if col == 3:
            cb2 = fig.colorbar(sc2, ax=ax2, shrink=0.65, pad=0.10)
            cb2.set_label("p (Pa)", fontsize=9)

    plt.tight_layout()
    out_3d = HERE / "aneur_panel_3d.png"
    plt.savefig(out_3d, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"\nSalvo: {out_3d}")

    ### ============= ### ###  Figura 7: aneur_panel_plane.png  ### ###  ============= ###
    # Plano central z ~ center[2] (~ 39.75 mm). Interpolacao k-NN do point cloud.

    z_plane = float(center[2])
    tree = cKDTree(xyz)

    def interp_field(field, n_grid=110, pad=1.0):
        x_r = np.linspace(bbox_min[0] - pad, bbox_max[0] + pad, n_grid)
        y_r = np.linspace(bbox_min[1] - pad, bbox_max[1] + pad, n_grid)
        Xg, Yg = np.meshgrid(x_r, y_r, indexing="xy")
        pts = np.stack([Xg.ravel(), Yg.ravel(), np.full(Xg.size, z_plane)], axis=1)
        ds, idxs = tree.query(pts, k=8)
        w = 1.0 / (ds + 1e-6); w /= w.sum(axis=1, keepdims=True)
        f = (field[idxs] * w).sum(axis=1).reshape(n_grid, n_grid)
        f[(ds[:, 0] > 1.0).reshape(n_grid, n_grid)] = np.nan
        return x_r, y_r, f

    x_r, y_r, speed_g_gt = interp_field(speed_gt)
    _, _, p_g_gt = interp_field(p_gt)
    extent_plane = [x_r.min(), x_r.max(), y_r.min(), y_r.max()]
    # Mesma escala de cor do painel 3D para facilitar a leitura conjunta.
    speed_vmax_p = speed_vmax
    p_vmax_p = p_vmax

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for col, (titulo, label) in enumerate(FIG_PANELS):
        if label is None:
            sp = speed_gt; pr = p_gt
        else:
            try:
                _, pred = load_record(label)
                sp = np.sqrt(pred["u_pred"]**2 + pred["v_pred"]**2 + pred["w_pred"]**2)
                pr = pred["p_pred"]
            except FileNotFoundError:
                sp = np.zeros_like(speed_gt); pr = np.zeros_like(p_gt)

        _, _, sp_g = interp_field(sp)
        _, _, pr_g = interp_field(pr)

        ax = axes[0, col]
        im = ax.imshow(sp_g.T, origin="lower", extent=extent_plane, cmap="viridis",
                       vmin=0, vmax=speed_vmax_p, aspect="equal")
        ax.set_title(rf"{titulo}: $|\mathbf{{u}}|$", fontsize=11)
        ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
        if col == 3:
            cb = fig.colorbar(im, ax=ax, shrink=0.85)
            cb.set_label("|u| (m/s)", fontsize=9)

        ax2 = axes[1, col]
        im2 = ax2.imshow(pr_g.T, origin="lower", extent=extent_plane, cmap="RdBu_r",
                         vmin=-p_vmax_p, vmax=p_vmax_p, aspect="equal")
        ax2.set_title(rf"{titulo}: $p$", fontsize=11)
        ax2.set_xlabel("X (mm)"); ax2.set_ylabel("Y (mm)")
        if col == 3:
            cb2 = fig.colorbar(im2, ax=ax2, shrink=0.85)
            cb2.set_label("p (Pa)", fontsize=9)

    plt.tight_layout()
    out_plane = HERE / "aneur_panel_plane.png"
    plt.savefig(out_plane, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Salvo: {out_plane}")

    return table_results


if __name__ == "__main__":
    main()
