"""Inspect the trained Mix 3x1 state dict and print the dominant atomic
basis functions per neuron, formatted for inclusion in the monograph.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch as tc

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mixfunn import Mix2Funn, BASE_FUNCTION_NAMES, Q

STATE_PT = HERE / "results_v13" / "prune_mix_3x1_coons.statedict.pt"
RESULTS_JSON = HERE / "results_v13" / "prune_mix_3x1_coons.json"


def analyze():
    # Rebuild model exactly as trained (2 in -> 3 out, n_layers=3, n_hidden=1, sof=False)
    net = Mix2Funn(
        n_in=2, n_out=3, n_layers=3, n_hidden=1,
        use_softmax=True, T_init=5.0, T_final=0.05,
        n_anneal_epochs=15000,
        second_order_function=False, dropout=0.0, init_alpha_std=0.1,
    )
    state = tc.load(STATE_PT, map_location="cpu", weights_only=True)
    net.load_state_dict(state, strict=False)  # _prune_mask is buffer not in fresh model
    net.eval()

    info = json.loads(RESULTS_JSON.read_text()) if RESULTS_JSON.exists() else {}
    final_T = 0.05
    print(f"=== Mix 3x1 sof=False, BC Coons, post-treino ===")
    print(f"L2_val final: {info.get('l2_val', 'n/a')}")
    print(f"n_params: {info.get('n_params', 'n/a')}\n")

    bullet_lines = []
    for li, layer in enumerate(net.layers):
        # alpha shape: [n_out, Q]
        alpha = layer.alpha.detach().cpu().numpy()
        # softmax aplicada com temperature
        w = np.exp(alpha / final_T)
        w = w / w.sum(axis=1, keepdims=True)
        n_out_l = alpha.shape[0]
        print(f"--- Layer {li+1} (n_out={n_out_l}) ---")
        layer_summary = []
        for k in range(n_out_l):
            row = w[k]  # shape [Q]
            top_idx = int(np.argmax(row))
            top_w = float(row[top_idx])
            top_name = BASE_FUNCTION_NAMES[top_idx]
            # Lista pesos relativos
            ranked = [(BASE_FUNCTION_NAMES[i], float(row[i])) for i in range(Q)]
            ranked.sort(key=lambda x: -x[1])
            top3 = ranked[:3]
            top3_str = ", ".join(f"{n}={p:.2f}" for n, p in top3)
            print(f"  saida {k+1}: dominante={top_name} ({top_w:.3f})  | top3: {top3_str}")
            layer_summary.append((top_name, top_w, ranked))
        bullet_lines.append((li + 1, layer_summary))

    # Construir bullet para LaTeX
    parts = []
    for li, summary in bullet_lines:
        ts = []
        for k, (name, w, ranked) in enumerate(summary):
            ts.append(f"saida {k+1}: {name} ($w$={w:.2f})")
        parts.append(f"\\textbf{{camada {li}}} --- " + ", ".join(ts))
    latex_bullet = "; ".join(parts)
    print("\n=== LaTeX bullet ===")
    print(latex_bullet)

    # Salva pra patch_latex_v13 ler
    out = HERE / "results_v13" / "analyze_summary.json"
    out.write_text(json.dumps({
        "bullet": latex_bullet,
        "layers": [
            {"layer": li, "summary": [(n, w, r) for (n, w, r) in s]}
            for li, s in bullet_lines
        ],
    }, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    analyze()
