#### Importando Bibliotecas
import json
import sys
from pathlib import Path

import numpy as np
import torch as tc

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mixfunn import Mix2Funn, BASE_FUNCTION_NAMES, Q


### ======================================== ###
###          Configuracao do script           ###
### ======================================== ###

CKPT_DIR = HERE / "tmp_checkpoints"  # destino do `modal volume get tcc /checkpoints/exp_03`
T_FINAL_MIX = 0.05
ITER = 15000
SEED = 21


def reconstruir_rede(record: dict, ckpt_dir: Path) -> Mix2Funn:
    """Reconstroi a Mix2Funn com a mesma topologia do treino, carrega state_dict."""
    tc.manual_seed(SEED)
    net = Mix2Funn(
        n_in=2, n_out=3,
        n_layers=record["n_layers"], n_hidden=record["n_hidden"],
        use_softmax=True, T_init=5.0, T_final=T_FINAL_MIX,
        n_anneal_epochs=ITER,
        second_order_function=record["sof"],
        dropout=0.0, init_alpha_std=0.1,
    )
    label = record["label"]
    state = tc.load(ckpt_dir / f"{label}.statedict.pt", map_location="cpu", weights_only=True)
    # _prune_mask vira como buffer extra apenas se houve prune_alpha aplicado e
    # mantido; aceitamos strict=False para tolerar ausencia.
    net.load_state_dict(state, strict=False)
    net.eval()
    return net


def imprimir_tabela_4(records: list[dict]) -> None:
    """Imprime a Tabela 4 (pruning loss) lado a lado."""

    # PDF reference (Tab. 4)
    pdf_3x1 = {0.00: 8.4e-2, 0.30: 8.9e-2, 0.50: 2.0e-1, 0.70: 6.0e-1, 0.90: 1.3e-1}
    pdf_sof = {0.00: 1.9e-2, 0.30: 1.9e-2, 0.50: 1.2e-1, 0.70: 2.1e-1, 0.90: 1.8e-1}

    def linha(r: float, pr_3x1: dict, pr_sof: dict) -> str:
        l2_3 = pr_3x1["l2"]
        l2_s = pr_sof["l2"]
        d3 = abs(l2_3 - pdf_3x1[r]) / pdf_3x1[r] * 100.0
        ds = abs(l2_s - pdf_sof[r]) / pdf_sof[r] * 100.0
        return (f"  r={r:.2f}  |  Mix 3x1: gerado={l2_3:.2e}  PDF={pdf_3x1[r]:.1e}  Δ={d3:5.1f}%"
                f"  ||  Mix-sof 1x1: gerado={l2_s:.2e}  PDF={pdf_sof[r]:.1e}  Δ={ds:5.1f}%")

    rec_3x1 = next(r for r in records if r["label"] == "mix_3x1")
    rec_sof = next(r for r in records if r["label"] == "mix_sof_1x1")

    print("=" * 100)
    print("Tabela 4 — Loss apos pruning das MixFunn no escoamento de Kovasznay")
    print("=" * 100)
    print(f"Mix 3x1     : n_params={rec_3x1['n_params']}  n_alpha={rec_3x1['n_alpha']}")
    print(f"Mix-sof 1x1 : n_params={rec_sof['n_params']}  n_alpha={rec_sof['n_alpha']}")
    print()
    for r in [0.00, 0.30, 0.50, 0.70, 0.90]:
        pr_3 = next(p for p in rec_3x1["prune_results"] if abs(p["ratio"] - r) < 1e-9)
        pr_s = next(p for p in rec_sof["prune_results"] if abs(p["ratio"] - r) < 1e-9)
        print(linha(r, pr_3, pr_s))
    print()


def imprimir_funcoes_atomicas_dominantes(record: dict, ckpt_dir: Path) -> None:
    """Inspeciona os alphas da rede pruned (somente para Mix 3x1, sof=False)
    e imprime, por camada e por neuronio, qual funcao da base concentrou
    maior peso na softmax. Util para extrair forma analitica.
    """
    if record["sof"]:
        return  # apenas para a 3x1 sem produtos cruzados

    print("=" * 100)
    print("Funcoes atomicas dominantes — Mix 3x1 (sof=False) pos-treino")
    print("=" * 100)

    net = reconstruir_rede(record, ckpt_dir)

    for li, layer in enumerate(net.layers):
        alpha = layer.alpha.detach().cpu().numpy()  # [n_out, Q]
        # softmax com temperatura final
        w = np.exp(alpha / T_FINAL_MIX)
        w = w / w.sum(axis=1, keepdims=True)
        n_out_l = alpha.shape[0]
        print(f"--- Camada {li + 1} (n_out={n_out_l}) ---")
        for k in range(n_out_l):
            row = w[k]
            top_idx = int(np.argmax(row))
            top_w = float(row[top_idx])
            top_name = BASE_FUNCTION_NAMES[top_idx]
            ranked = [(BASE_FUNCTION_NAMES[i], float(row[i])) for i in range(Q)]
            ranked.sort(key=lambda x: -x[1])
            top3_str = ", ".join(f"{n}={p:.2f}" for n, p in ranked[:3])
            print(f"  saida {k + 1}: dominante={top_name} (w={top_w:.3f})  | top3: {top3_str}")
    print()


def localizar_arquivo(nome: str) -> Path | None:
    """Procura `nome` em CKPT_DIR e em alguns subdiretorios usuais."""
    for cand in [CKPT_DIR / nome,
                 CKPT_DIR / "exp_03" / nome,
                 CKPT_DIR / "checkpoints" / "exp_03" / nome]:
        if cand.exists():
            return cand
    return None


def main() -> None:
    if not CKPT_DIR.exists():
        sys.exit(f"Diretorio {CKPT_DIR} nao existe. Rode antes:\n"
                 f"  modal volume get tcc /checkpoints/exp_03 {CKPT_DIR.name}")

    records: list[dict] = []
    arquivo_referencia: Path | None = None
    for label in ["mix_3x1", "mix_sof_1x1"]:
        cand = localizar_arquivo(f"{label}.json")
        if cand is None:
            sys.exit(f"Nao encontrei {label}.json em {CKPT_DIR}")
        records.append(json.loads(cand.read_text()))
        arquivo_referencia = cand.parent  # diretorio onde estao tambem os .statedict.pt

    assert arquivo_referencia is not None
    imprimir_tabela_4(records)
    rec_3x1 = next(r for r in records if r["label"] == "mix_3x1")
    imprimir_funcoes_atomicas_dominantes(rec_3x1, arquivo_referencia)


if __name__ == "__main__":
    main()
