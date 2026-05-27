#### Importando Bibliotecas
import json
import math
from pathlib import Path

### ============= ### ###  Localizacao dos Checkpoints  ### ###  ============= ###

HERE = Path(__file__).resolve().parent
CHECKPOINTS_LOCAL = HERE / "tmp_checkpoints" / "by_label"

### ============= ### ###  Formatacao em Notacao Cientifica  ### ###  ============= ###

def fmt_sci(x, casas=1):
    if x is None or not math.isfinite(x):
        return "---"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10 ** e
    return f"{m:.{casas}f}e{e:+d}"

### ============= ### ###  Leitura dos Resultados  ### ###  ============= ###

def carregar(label):
    p = CHECKPOINTS_LOCAL / f"{label}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())

### ============= ### ###  Impressao da Tabela 3  ### ###  ============= ###

def main():
    if not CHECKPOINTS_LOCAL.exists():
        raise FileNotFoundError(
            f"Diretorio {CHECKPOINTS_LOCAL} nao existe. Baixe primeiro com:\n"
            f"  modal volume get tcc /checkpoints/exp_02 ./tmp_checkpoints"
        )

    linhas = [
        ("baseline",          "pinn_4x32_baseline", "mix_3x1_baseline", 1.2e-3, 8.4e-2),
        ("dropout 10%",       "pinn_4x32_drop10",   "mix_3x1_drop10",   8.9e-2, 9.7e-2),
        ("dropout 15%",       "pinn_4x32_drop15",   "mix_3x1_drop15",   9.0e-2, 9.7e-2),
        ("sub-amostra 70%",   "pinn_4x32_sub70",    "mix_3x1_sub70",    1.6e-3, 1.7e-2),
        ("sub-amostra 50%",   "pinn_4x32_sub50",    "mix_3x1_sub50",    1.9e-3, 4.7e-2),
        ("combo",             "pinn_4x32_combo",    "mix_3x1_combo",    9.0e-2, 9.7e-2),
    ]

    print()
    print("Tabela 3 — Regularizacao em Kovasznay")
    print("=" * 86)
    print(f"{'Configuracao':<18} | {'PINN 4x32':<22} | {'MixFunn 3x1':<22} | walltime (s)")
    print(f"{'':<18} | {'gerado':<10} {'PDF':<11} | {'gerado':<10} {'PDF':<11} | PINN  /  Mix")
    print("-" * 86)

    for nome, lab_p, lab_m, ref_p, ref_m in linhas:
        rp = carregar(lab_p); rm = carregar(lab_m)
        gp = rp["l2_val"] if rp else None
        gm = rm["l2_val"] if rm else None
        wp = rp["wall_clock"] if rp else None
        wm = rm["wall_clock"] if rm else None
        print(
            f"{nome:<18} | "
            f"{fmt_sci(gp):<10} {fmt_sci(ref_p):<11} | "
            f"{fmt_sci(gm):<10} {fmt_sci(ref_m):<11} | "
            f"{(f'{wp:5.0f}' if wp else '  ---'):>5} / {(f'{wm:5.0f}' if wm else '  ---'):>5}"
        )

    print("=" * 86)
    print()
    print("Comparacao com Tabela 3 do monograph (PDF):")
    print()
    for nome, lab_p, lab_m, ref_p, ref_m in linhas:
        rp = carregar(lab_p); rm = carregar(lab_m)
        for kind, r, ref in [("PINN 4x32", rp, ref_p), ("MixFunn 3x1", rm, ref_m)]:
            if r is None:
                print(f"  {kind:<12} {nome:<18}: ausente")
                continue
            g = r["l2_val"]
            delta_pct = (g - ref) / ref * 100.0
            ord_g = int(math.floor(math.log10(abs(g)))) if g > 0 else None
            ord_r = int(math.floor(math.log10(abs(ref)))) if ref > 0 else None
            mesma_ordem = (ord_g == ord_r)
            tag = "OK" if mesma_ordem else "DIFERE"
            print(
                f"  {kind:<12} {nome:<18}: gerado={fmt_sci(g)} "
                f"PDF={fmt_sci(ref)} delta={delta_pct:+7.1f}%  [{tag}]"
            )
    print()


if __name__ == "__main__":
    main()
