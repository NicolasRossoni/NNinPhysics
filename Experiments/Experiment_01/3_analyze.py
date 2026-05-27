#### Importando Bibliotecas
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

### ============= ### ###  Hiperparametros  ### ###  ============= ###

# Diretorio com os arquivos baixados via `modal volume get tcc /checkpoints/exp_01 ./tmp_checkpoints`
HERE = Path(__file__).resolve().parent
CHECKPOINTS = HERE / "tmp_checkpoints"
# Caso o modal volume get tenha criado um subdiretorio adicional, tentamos descer
if (CHECKPOINTS / "exp_01").exists():
    CHECKPOINTS = CHECKPOINTS / "exp_01"

# Saida das figuras
OUT_FIG_FIELDS = HERE / "kov_v12_fields_extrap.png"
OUT_FIG_LOSS = HERE / "kov_v12_loss_decomp.png"

# Dominio (precisa casar com 2_train.py)
X_MIN, X_TR, X_MAX = -0.5, 1.0, 2.5
Y_MIN, Y_MAX = -0.5, 1.5

# Seeds da Tabela 1
SEEDS_T1 = [21, 22, 23]

# Sweeps da Tabela 2
PINN_NL = [4, 6, 8]
PINN_NN = [16, 32, 64]
MIX_NL = [1, 2, 3]
MIX_NH = [1, 2, 3]


### ============= ### ###  Carregando resultados  ### ###  ============= ###

def load_json(label):
    path = CHECKPOINTS / f"{label}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def load_npz(label):
    path = CHECKPOINTS / f"{label}.npz"
    if not path.exists():
        return None
    return np.load(path)


### ============= ### ###  Formatacao cientifica  ### ###  ============= ###

def sci(x, casas=1):
    if x is None or not np.isfinite(x):
        return "---"
    if x == 0:
        return "0"
    e = int(np.floor(np.log10(abs(x))))
    m = x / 10 ** e
    return f"{m:.{casas}f}e{e:+d}"


### ============= ### ###  Tabela 1: sup vs nsup  ### ###  ============= ###

def print_table_1():
    print("\n=== Tabela 1: sup vs nsup (3 seeds) ===")
    print(f"{'config':<32}" + "".join(f"  s={s:<8}" for s in SEEDS_T1) + "  sigma")
    rows = [
        ("Sup PINN (4x32)", "pinn_4x32_sup"),
        ("Nsup PINN (4x32)", "pinn_4x32_nsup"),
        ("Sup MixFunn (3x1)", "mix_3x1_sup"),
        ("Nsup MixFunn (3x1)", "mix_3x1_nsup"),
    ]
    table = {}
    for nome, prefix in rows:
        vals = []
        for s in SEEDS_T1:
            r = load_json(f"{prefix}_s{s}")
            vals.append(r["l2_val"] if r else float("nan"))
        sigma = float(np.std(vals)) if len(vals) > 1 else 0.0
        table[nome] = (vals, sigma)
        cells = "  ".join(sci(v) for v in vals)
        print(f"{nome:<32}  {cells}  sigma={sci(sigma)}")
    return table


### ============= ### ###  Tabela 2: sweep arquitetural  ### ###  ============= ###

def print_table_2():
    print("\n=== Tabela 2a: PINN sweep (N x n) ===")
    print(f"{'N\\n':<6}" + "".join(f"  n={n:<8}" for n in PINN_NN))
    pinn_mat = np.full((len(PINN_NL), len(PINN_NN)), np.nan)
    for i, nL in enumerate(PINN_NL):
        row_cells = []
        for j, nW in enumerate(PINN_NN):
            r = load_json(f"arch_pinn_{nL}x{nW}")
            if r:
                pinn_mat[i, j] = r["l2_val"]
            row_cells.append(sci(pinn_mat[i, j]))
        print(f"N={nL:<4}" + "  ".join(f"  {c:<10}" for c in row_cells))

    print("\n=== Tabela 2b: MixFunn sweep (N x n, sof=False) ===")
    print(f"{'N\\n':<6}" + "".join(f"  n={n:<8}" for n in MIX_NH))
    mix_mat = np.full((len(MIX_NL), len(MIX_NH)), np.nan)
    for i, nL in enumerate(MIX_NL):
        row_cells = []
        for j, nh in enumerate(MIX_NH):
            r = load_json(f"arch_mix_{nL}x{nh}")
            if r:
                mix_mat[i, j] = r["l2_val"]
            row_cells.append(sci(mix_mat[i, j]))
        print(f"N={nL:<4}" + "  ".join(f"  {c:<10}" for c in row_cells))

    sof = load_json("mix_1x1_sof_true")
    if sof:
        print(f"\nMixFunn 1x1 sof=True: L2 = {sci(sof['l2_val'])}")

    return pinn_mat, mix_mat, sof


### ============= ### ###  Figura 1: campos com extrapolacao  ### ###  ============= ###

def plot_fields_extrap():
    pinn = load_json("pinn_4x32_nsup_s21")
    mix = load_json("mix_3x1_nsup_s21")
    pinn_npz = load_npz("pinn_4x32_nsup_s21")
    mix_npz = load_npz("mix_3x1_nsup_s21")
    if pinn is None or mix is None or pinn_npz is None or mix_npz is None:
        print("[plot_fields_extrap] arquivos ausentes; pulando figura")
        return

    x_full = pinn_npz["x_lin"]; y_full = pinn_npz["y_lin"]
    u_an = pinn_npz["u_analytic"]
    u_pinn = pinn_npz["u_pred"]
    u_mix = mix_npz["u_pred"]

    fig = plt.figure(figsize=(13, 12))
    gs = fig.add_gridspec(4, 4, height_ratios=[1, 1, 1, 1.2], hspace=0.4, wspace=0.3)
    titles = ["Analitica", "PINN $4{\\times}32$", "MixFunn $3{\\times}1$"]
    datas = [u_an, u_pinn, u_mix]
    vmin = min(d.min() for d in datas); vmax = max(d.max() for d in datas)
    x_cuts = [0.0, 0.75, 1.25, 2.0]

    for i, (data, title) in enumerate(zip(datas, titles)):
        ax = fig.add_subplot(gs[i, :])
        im = ax.pcolormesh(x_full, y_full, data, cmap="viridis",
                           vmin=vmin, vmax=vmax, shading="auto")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title(title)
        ax.axvline(X_TR, color="white", linestyle=":", linewidth=1.4,
                   label="limite treino $x=1$")
        for k, xc in enumerate(x_cuts):
            ax.axvline(xc, color="red", linestyle="--", linewidth=0.8, alpha=0.7,
                       label="cortes" if k == 0 else None)
        if i == 0:
            ax.legend(fontsize=8, loc="upper left", framealpha=0.7)
        plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="u")

    for j, xc in enumerate(x_cuts):
        ax = fig.add_subplot(gs[3, j])
        ix = int(np.argmin(np.abs(x_full - xc)))
        ax.plot(y_full, u_an[:, ix], "k-", linewidth=1.6, label="Analitica")
        ax.plot(y_full, u_pinn[:, ix], "b--", linewidth=1.2, label="PINN")
        ax.plot(y_full, u_mix[:, ix], color="orange", linestyle="--",
                linewidth=1.2, label="MixFunn")
        ax.set_xlabel("y")
        if j == 0:
            ax.set_ylabel("u")
        zone = "treino" if xc <= X_TR else "extrap"
        ax.set_title(f"corte $x={xc:g}$ ({zone})", fontsize=10)
        ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="best", framealpha=0.7)

    plt.suptitle("Kovasznay --- campos $u$ com BC Coons, $x$ estendido ate $2{,}5$",
                 fontsize=11)
    plt.savefig(OUT_FIG_FIELDS, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"saved {OUT_FIG_FIELDS}")


### ============= ### ###  Figura 2: loss decomp  ### ###  ============= ###

def plot_loss_decomp():
    sup = load_json("pinn_4x32_sup_s21")
    nsup = load_json("pinn_4x32_nsup_s21")
    if sup is None or nsup is None:
        print("[plot_loss_decomp] arquivos ausentes; pulando figura")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.semilogy(sup["epochs_log"], sup["l2_val_curve"], "b-",
                 label="PINN supervisionada", linewidth=1.5)
    ax1.semilogy(nsup["epochs_log"], nsup["l2_val_curve"], "r-",
                 label="PINN nao-supervisionada", linewidth=1.5)
    ax1.set_xlabel("Iteracoes"); ax1.set_ylabel("$L^2$ relativo na validacao")
    ax1.set_title("(a) Convergencia: sup vs nao-sup (PINN $4{\\times}32$, seed 21)")
    ax1.grid(alpha=0.4); ax1.legend()

    ax2.semilogy(nsup["epochs_log"], nsup["loss_momento"], color="purple",
                 label=r"$\mathcal{L}_{Momento}$ (Eq.1 + Eq.2)", linewidth=1.5)
    ax2.semilogy(nsup["epochs_log"], nsup["loss_massa"], color="darkgreen",
                 label=r"$\mathcal{L}_{Massa}$ (Eq.3)", linewidth=1.5)
    ax2.set_xlabel("Iteracoes"); ax2.set_ylabel("Loss componente")
    ax2.set_title("(b) PINN nao-sup: loss decomposta em conservacoes")
    ax2.grid(alpha=0.4); ax2.legend()

    plt.tight_layout()
    plt.savefig(OUT_FIG_LOSS, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"saved {OUT_FIG_LOSS}")


### ============= ### ###  Main  ### ###  ============= ###

if __name__ == "__main__":
    print(f"[analyze] lendo checkpoints em {CHECKPOINTS}")
    if not CHECKPOINTS.exists():
        raise SystemExit(
            f"Diretorio {CHECKPOINTS} nao existe. Rode antes:\n"
            f"  modal volume get tcc /checkpoints/exp_01 ./tmp_checkpoints"
        )
    print_table_1()
    print_table_2()
    plot_fields_extrap()
    plot_loss_decomp()
    print("\n[analyze] OK")
