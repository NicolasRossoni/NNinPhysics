"""Aggregate per-seed training metrics into the tables shown in the monograph."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
RESULTS = Path("/tmp/ns_kovasznay_results.json")
FIG_DIR = HERE.parents[2] / "reports" / "latex" / "tcc" / "figures"
TABLES_OUT = Path("/tmp/kov_tables.tex")

RE = 40.0
LAM = RE / 2.0 - math.sqrt((RE / 2.0) ** 2 + 4.0 * math.pi ** 2)
X_MIN, X_MAX = -0.5, 1.0
Y_MIN, Y_MAX = -0.5, 1.5
X_MIN_TR, X_MAX_TR = -0.25, 0.75
Y_MIN_TR, Y_MAX_TR = -0.25, 1.25


def kovasznay_np(X, Y):
    u = 1.0 - np.exp(LAM * X) * np.cos(2.0 * np.pi * Y)
    v = (LAM / (2.0 * np.pi)) * np.exp(LAM * X) * np.sin(2.0 * np.pi * Y)
    p = 0.5 * (1.0 - np.exp(2.0 * LAM * X))
    return u, v, p


def sci(x, casas=1):
    if x is None or not np.isfinite(x):
        return "---"
    if x == 0:
        return "0"
    e = int(np.floor(np.log10(abs(x))))
    m = x / 10 ** e
    s = f"{m:.{casas}f}".replace(".", "{,}")
    return f"{s} \\times 10^{{{e}}}"


def load_results():
    if RESULTS.exists():
        return json.loads(RESULTS.read_text())
    # Tentar fallback de incremental local (alguma copia em /tmp)
    fb = Path("/tmp/ns_kovasznay_by_label")
    if fb.exists():
        out = []
        for f in sorted(fb.glob("*.json")):
            out.append(json.loads(f.read_text()))
        return out
    raise FileNotFoundError(f"Sem resultados em {RESULTS}; rode run.py primeiro.")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    results = load_results()
    by_label = {r["label"]: r for r in results}
    print(f"loaded {len(results)} results")

    # ============================================================
    # 1) Sweep matrices (3x3 cada)
    # ============================================================
    pinn_NL = [4, 6, 8]
    pinn_NN = [16, 32, 64]
    pinn_mat = np.full((3, 3), np.nan)
    pinn_labels = np.empty((3, 3), dtype=object)
    for i, nL in enumerate(pinn_NL):
        for j, nn_w in enumerate(pinn_NN):
            lab = f"arch_pinn_{nL}x{nn_w}"
            if lab in by_label:
                pinn_mat[i, j] = by_label[lab]["l2_val"]
                pinn_labels[i, j] = lab

    mix_NL = [2, 3, 4]
    mix_NH = [4, 6, 8]
    mix_mat = np.full((3, 3), np.nan)
    mix_labels = np.empty((3, 3), dtype=object)
    for i, nL in enumerate(mix_NL):
        for j, nh in enumerate(mix_NH):
            lab = f"arch_mix_{nL}x{nh}"
            if lab in by_label:
                mix_mat[i, j] = by_label[lab]["l2_val"]
                mix_labels[i, j] = lab

    def top_k(mat, labels, k=3):
        flat = mat.flatten()
        labs = labels.flatten()
        ok = np.isfinite(flat)
        idx_ok = np.where(ok)[0]
        if len(idx_ok) == 0:
            return []
        sorted_idx = idx_ok[np.argsort(flat[idx_ok])]
        return [labs[i] for i in sorted_idx[:k]]

    pinn_top3 = top_k(pinn_mat, pinn_labels, 3)
    mix_top3 = top_k(mix_mat, mix_labels, 3)
    pinn_best = pinn_top3[0] if pinn_top3 else None
    mix_best = mix_top3[0] if mix_top3 else None
    if pinn_best:
        print(f"PINN top3: {pinn_top3}  best L2={by_label[pinn_best]['l2_val']:.3e}")
    if mix_best:
        print(f"Mix  top3: {mix_top3}   best L2={by_label[mix_best]['l2_val']:.3e}")

    # ============================================================
    # Fig: 3 melhores PINN (rasters de |u|: predicao e erro)
    # ============================================================
    def fig_top3(top3, name_out, titulo):
        if not top3:
            return
        fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.4))
        nx = by_label[top3[0]]["grid_nx"]
        ny = by_label[top3[0]]["grid_ny"]
        xe = np.linspace(X_MIN, X_MAX, nx)
        ye = np.linspace(Y_MIN, Y_MAX, ny)
        XE, YE = np.meshgrid(xe, ye, indexing="xy")
        u_ref, v_ref, _ = kovasznay_np(XE, YE)
        mag_ref = np.sqrt(u_ref ** 2 + v_ref ** 2)
        vmin = float(mag_ref.min()); vmax = float(mag_ref.max())
        for k, lab in enumerate(top3):
            r = by_label[lab]
            u = np.asarray(r["u_grid"]); v = np.asarray(r["v_grid"])
            mag = np.sqrt(u ** 2 + v ** 2)
            err = np.abs(mag - mag_ref)
            ax = axes[0, k]
            im = ax.imshow(mag, origin="lower", extent=[X_MIN, X_MAX, Y_MIN, Y_MAX],
                            cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_title(f"{lab}\n$L^2={r['l2_val']:.2e}$", fontsize=9)
            ax.set_xlabel("$x$"); ax.set_ylabel("$y$" if k == 0 else "")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax2 = axes[1, k]
            im2 = ax2.imshow(err, origin="lower", extent=[X_MIN, X_MAX, Y_MIN, Y_MAX],
                              cmap="magma", aspect="auto")
            ax2.set_title(f"erro absoluto $|u|$", fontsize=9)
            ax2.set_xlabel("$x$"); ax2.set_ylabel("$y$" if k == 0 else "")
            plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        fig.suptitle(titulo, fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(FIG_DIR / name_out, dpi=150)
        plt.close(fig)
        print(f"ok {name_out}")

    fig_top3(pinn_top3, "kov_arch_3redes_pinn.png", "PINN — 3 melhores arquiteturas")
    fig_top3(mix_top3, "kov_arch_3redes_mix.png", "MixFunn — 3 melhores arquiteturas")

    # ============================================================
    # Fig: kov_compare (analitica | PINN | Mix), heatmap + cortes
    # ============================================================
    if pinn_best and mix_best:
        rp = by_label[pinn_best]; rm = by_label[mix_best]
        nx = rp["grid_nx"]; ny = rp["grid_ny"]
        xe = np.linspace(X_MIN, X_MAX, nx)
        ye = np.linspace(Y_MIN, Y_MAX, ny)
        XE, YE = np.meshgrid(xe, ye, indexing="xy")
        u_ref, v_ref, _ = kovasznay_np(XE, YE)
        mag_ref = np.sqrt(u_ref ** 2 + v_ref ** 2)
        u_p = np.asarray(rp["u_grid"]); v_p = np.asarray(rp["v_grid"])
        mag_p = np.sqrt(u_p ** 2 + v_p ** 2)
        u_m = np.asarray(rm["u_grid"]); v_m = np.asarray(rm["v_grid"])
        mag_m = np.sqrt(u_m ** 2 + v_m ** 2)

        fig = plt.figure(figsize=(10.0, 6.4))
        gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.85])
        vmin = float(min(mag_ref.min(), mag_p.min(), mag_m.min()))
        vmax = float(max(mag_ref.max(), mag_p.max(), mag_m.max()))
        for k, (mag, ttl) in enumerate([
            (mag_ref, "analitica"),
            (mag_p, f"PINN(${rp['n_layers']}\\times{rp['width']}$)"),
            (mag_m, f"MixFunn(${rm['n_layers']}\\times{rm['width']}$)"),
        ]):
            ax = fig.add_subplot(gs[0, k])
            im = ax.imshow(mag, origin="lower", extent=[X_MIN, X_MAX, Y_MIN, Y_MAX],
                            cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_title(ttl)
            ax.set_xlabel("$x$"); ax.set_ylabel("$y$" if k == 0 else "")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax2 = fig.add_subplot(gs[1, :])
        cores = ["tab:red", "tab:green", "tab:orange", "tab:purple"]
        for i, x_fix in enumerate([-0.25, 0.0, 0.5, 0.9]):
            j_x = int(np.argmin(np.abs(xe - x_fix)))
            ax2.plot(ye, mag_ref[:, j_x], color=cores[i], lw=1.4,
                     label=f"$x={x_fix:.2f}$")
            ax2.plot(ye, mag_p[:, j_x], color=cores[i], lw=0.9, linestyle="--")
            ax2.plot(ye, mag_m[:, j_x], color=cores[i], lw=0.9, linestyle=":")
        ax2.set_xlabel("$y$"); ax2.set_ylabel(r"$\sqrt{u^2+v^2}$")
        ax2.grid(alpha=0.3); ax2.legend(loc="upper right", fontsize=7, ncol=2)
        ax2.set_title("Cortes em $x$ fixo: analitica (--), PINN (--), MixFunn (:)")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "kov_compare.png", dpi=150)
        plt.close(fig)
        print("ok kov_compare.png")

    # ============================================================
    # Fig: 4 curvas de loss sup/nsup
    # ============================================================
    def pick(prefix, seeds=(21, 22, 23)):
        d = {}
        for s in seeds:
            lab = f"{prefix}_s{s}"
            if lab in by_label:
                d[s] = by_label[lab]
        return d

    sup_pinn = pick("supnsup_pinn_sup")
    nsup_pinn = pick("supnsup_pinn_nsup")
    sup_mix = pick("supnsup_mix_sup")
    nsup_mix = pick("supnsup_mix_nsup")

    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    estilos = [
        ("Sup PINN",     sup_pinn,  "tab:blue",   "-"),
        ("Nao-Sup PINN", nsup_pinn, "tab:blue",   "--"),
        ("Sup MixFunn",     sup_mix,   "tab:orange", "-"),
        ("Nao-Sup MixFunn", nsup_mix,  "tab:orange", "--"),
    ]
    last_r = None
    for nome, store, cor, ls in estilos:
        if not store:
            continue
        best_seed = min(store, key=lambda s: store[s]["l2_val"])
        r = store[best_seed]
        ax.plot(r["epochs_log"], r["loss_curve_val"], color=cor,
                linestyle=ls, lw=1.4, label=nome)
        last_r = r
    if last_r:
        ax.set_xlim(0, max(last_r["epochs_log"]))
    ax.set_xlabel("epoca"); ax.set_ylabel(r"$L^2_{\rm val}$")
    ax.set_ylim(0, 1.5)
    ax.grid(alpha=0.3); ax.legend(loc="upper right", fontsize=9)
    ax.set_title("Re=40, lr_PINN=1e-3, lr_Mix=1e-2, StepLR x0,5 a cada 10k")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "kov_loss_4curves.png", dpi=150)
    plt.close(fig)
    print("ok kov_loss_4curves.png")

    # ============================================================
    # Fig: loss por componente (NSup PINN best seed)
    # ============================================================
    if nsup_pinn:
        best_seed = min(nsup_pinn, key=lambda s: nsup_pinn[s]["l2_val"])
        r = nsup_pinn[best_seed]
        fig, ax = plt.subplots(figsize=(7.6, 4.0))
        ax.plot(r["epochs_log"], r["loss_pde"], color="tab:purple", lw=1.4,
                label="residuo NS")
        ax.plot(r["epochs_log"], r["loss_div"], color="tab:green",  lw=1.4,
                label="divergencia")
        ax.set_xlabel("epoca"); ax.set_ylabel("componente da loss")
        ax.set_xlim(0, max(r["epochs_log"]))
        all_v = list(r["loss_pde"]) + list(r["loss_div"])
        if all_v:
            ax.set_ylim(0, max(all_v) * 1.05)
        ax.grid(alpha=0.3); ax.legend(loc="upper right", fontsize=9)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "kov_loss_per_component.png", dpi=150)
        plt.close(fig)
        print("ok kov_loss_per_component.png")

    # ============================================================
    # Fig: sup vs unsup, 4 paineis (barras L2 por seed)
    # ============================================================
    fig, axes = plt.subplots(1, 4, figsize=(12.0, 3.2), sharey=True)
    rows = [
        ("Sup PINN",     sup_pinn,  "tab:blue"),
        ("Nao-Sup PINN", nsup_pinn, "tab:blue"),
        ("Sup MixFunn",     sup_mix,   "tab:orange"),
        ("Nao-Sup MixFunn", nsup_mix,  "tab:orange"),
    ]
    for ax, (nome, store, cor) in zip(axes, rows):
        seeds = sorted(store.keys()) if store else []
        vals = [store[s]["l2_val"] for s in seeds]
        ax.bar([str(s) for s in seeds], vals, color=cor, alpha=0.8)
        ax.set_title(nome, fontsize=9)
        ax.set_xlabel("seed")
        ax.grid(alpha=0.3, axis="y")
        if vals:
            ax.set_yscale("log")
    axes[0].set_ylabel(r"$L^2_{\rm val}$")
    fig.suptitle("Sup vs Nao-Sup, 3 seeds (log $L^2$)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG_DIR / "kov_sup_vs_unsup.png", dpi=150)
    plt.close(fig)
    print("ok kov_sup_vs_unsup.png")

    # ============================================================
    # Fig: interp/extrap (treino em subdominio interno)
    # ============================================================
    inp = by_label.get("interp_pinn"); inm = by_label.get("interp_mix")
    if inp and inm:
        fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.4))
        nx = inp["grid_nx"]; ny = inp["grid_ny"]
        xe = np.linspace(X_MIN, X_MAX, nx)
        ye = np.linspace(Y_MIN, Y_MAX, ny)
        XE, YE = np.meshgrid(xe, ye, indexing="xy")
        u_ref, v_ref, _ = kovasznay_np(XE, YE)
        mag_ref = np.sqrt(u_ref ** 2 + v_ref ** 2)
        for row, (r, nome) in enumerate([(inp, "PINN"), (inm, "MixFunn")]):
            u = np.asarray(r["u_grid"]); v = np.asarray(r["v_grid"])
            mag = np.sqrt(u ** 2 + v ** 2)
            err = np.abs(mag - mag_ref)
            for k, (data, ttl, cmap) in enumerate([
                (mag_ref, "analitica", "viridis"),
                (mag,     f"{nome}",   "viridis"),
                (err,     "erro |u|",  "magma"),
            ]):
                ax = axes[row, k]
                im = ax.imshow(data, origin="lower",
                                extent=[X_MIN, X_MAX, Y_MIN, Y_MAX],
                                cmap=cmap, aspect="auto")
                # Retangulo treino
                ax.plot([X_MIN_TR, X_MAX_TR, X_MAX_TR, X_MIN_TR, X_MIN_TR],
                        [Y_MIN_TR, Y_MIN_TR, Y_MAX_TR, Y_MAX_TR, Y_MIN_TR],
                        "w-", lw=1.0)
                ax.set_title(ttl if row == 0 else f"{nome}: {ttl}", fontsize=9)
                ax.set_xlabel("$x$" if row == 1 else "")
                ax.set_ylabel("$y$" if k == 0 else "")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle("Treino em subdominio interno (caixa branca); validacao em todo o dominio",
                     fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(FIG_DIR / "kov_interp_extrap.png", dpi=150)
        plt.close(fig)
        print("ok kov_interp_extrap.png")

    # ============================================================
    # Tabelas em /tmp/kov_tables.tex
    # ============================================================
    lines = []

    lines.append("% --- arch_pinn 3x3 ---")
    for i, nL in enumerate(pinn_NL):
        row = f"    ${nL}$"
        for j, w in enumerate(pinn_NN):
            v = pinn_mat[i, j]
            cell = sci(v) if np.isfinite(v) else "---"
            if pinn_labels[i, j] == pinn_best:
                cell = "\\mathbf{" + cell + "}"
            row += f" & ${cell}$"
        row += " \\\\"
        lines.append(row)
    lines.append("")

    lines.append("% --- arch_mix 3x3 ---")
    for i, nL in enumerate(mix_NL):
        row = f"    ${nL}$"
        for j, nh in enumerate(mix_NH):
            v = mix_mat[i, j]
            cell = sci(v) if np.isfinite(v) else "---"
            if mix_labels[i, j] == mix_best:
                cell = "\\mathbf{" + cell + "}"
            row += f" & ${cell}$"
        row += " \\\\"
        lines.append(row)
    lines.append("")

    lines.append("% --- sup vs nsup (3 seeds) ---")
    seeds = [21, 22, 23]
    for nome, store in [("Sup PINN", sup_pinn),
                         ("Nao-Sup PINN", nsup_pinn),
                         ("Sup MixFunn", sup_mix),
                         ("Nao-Sup MixFunn", nsup_mix)]:
        if not store:
            lines.append(f"    {nome} & --- & --- & --- & --- \\\\")
            continue
        vals = [store[s]["l2_val"] for s in seeds if s in store]
        std = float(np.std(vals)) if len(vals) > 1 else 0.0
        cells = [sci(store[s]["l2_val"]) if s in store else "---" for s in seeds]
        lines.append(f"    {nome} & " + " & ".join(f"${c}$" for c in cells)
                     + f" & ${sci(std)}$ \\\\")
    lines.append("")

    lines.append("% --- regularizacao (5 linhas + baseline) ---")
    reg_rows = [
        ("baseline",                          f"arch_pinn_{8}x{64}",   f"arch_mix_{3}x{6}"),
        ("dropout $p=0{,}05$",                "reg_pinn_dropout_0.05", "reg_mix_dropout_0.05"),
        ("dropout $p=0{,}10$",                "reg_pinn_dropout_0.1",  "reg_mix_dropout_0.1"),
        ("subsample $70\\%$",                  "reg_pinn_sub_0.7",      "reg_mix_sub_0.7"),
        ("subsample $50\\%$",                  "reg_pinn_sub_0.5",      "reg_mix_sub_0.5"),
        ("dropout$+$subsample",               "reg_pinn_combo",        "reg_mix_combo"),
    ]
    for nome, lab_p, lab_m in reg_rows:
        vp = by_label.get(lab_p, {}).get("l2_val", float("nan"))
        vm = by_label.get(lab_m, {}).get("l2_val", float("nan"))
        cp = sci(vp); cm = sci(vm)
        if "baseline" in nome:
            cp = "\\mathbf{" + cp + "}"
            cm = "\\mathbf{" + cm + "}"
        lines.append(f"    {nome} & ${cp}$ & ${cm}$ \\\\")
    lines.append("")

    lines.append("% --- interp/extrap ---")
    for nome, lab in [("PINN", "interp_pinn"), ("MixFunn", "interp_mix")]:
        r = by_label.get(lab)
        if not r:
            lines.append(f"    {nome} & --- & --- & --- & --- \\\\")
            continue
        lines.append(
            f"    {nome} & ${sci(r['l2_interp_u'])}$ & ${sci(r['l2_extrap_u'])}$"
            f" & ${sci(r['l2_interp_v'])}$ & ${sci(r['l2_extrap_v'])}$ \\\\"
        )
    lines.append("")

    lines.append("% --- WSS (parede inferior) ---")
    if pinn_best and mix_best:
        lines.append(
            f"    $L^2(\\tau_w)$ & ${sci(by_label[pinn_best]['l2_wss'])}$"
            f" & ${sci(by_label[mix_best]['l2_wss'])}$ \\\\"
        )
    lines.append("")

    lines.append("% --- eficiencia ---")
    if pinn_best and mix_best:
        rp = by_label[pinn_best]; rm = by_label[mix_best]
        lines.append(f"    Tempo de treino (s) & ${sci(rp['wall_clock'])}$ & ${sci(rm['wall_clock'])}$ \\\\")
        lines.append(f"    \\# parametros & ${sci(rp['n_params'])}$ & ${sci(rm['n_params'])}$ \\\\")
        lines.append(f"    $L^2$ global & ${sci(rp['l2_val'])}$ & ${sci(rm['l2_val'])}$ \\\\")

    TABLES_OUT.write_text("\n".join(lines))
    print(f"ok wrote tables -> {TABLES_OUT}")

    print("\n=== SUMMARY NS Kovasznay (final) ===")
    if pinn_best:
        rp = by_label[pinn_best]
        print(f"PINN best:    {pinn_best}  L2={rp['l2_val']:.3e} "
              f"WSS={rp['l2_wss']:.3e} wall={rp['wall_clock']:.1f}s")
    if mix_best:
        rm = by_label[mix_best]
        print(f"MixFunn best: {mix_best}  L2={rm['l2_val']:.3e} "
              f"WSS={rm['l2_wss']:.3e} wall={rm['wall_clock']:.1f}s")
    inp = by_label.get("interp_pinn"); inm = by_label.get("interp_mix")
    if inp:
        print(f"PINN interp/extrap: interp_u={inp['l2_interp_u']:.2e} extrap_u={inp['l2_extrap_u']:.2e}")
    if inm:
        print(f"Mix  interp/extrap: interp_u={inm['l2_interp_u']:.2e} extrap_u={inm['l2_extrap_u']:.2e}")


if __name__ == "__main__":
    main()
