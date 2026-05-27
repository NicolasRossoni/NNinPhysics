# Experimento 7 — Magnetostática 2D com disco paramagnético

> Produz a **Figura 10** de `monograph.pdf`.

## Problema

$$\left\{\begin{array}{l}
\nabla \times \mathbf{H} = 0, \\
\nabla \cdot (\mu\,\mathbf{H}) = 0, \\
\mathbf{H}|_{\partial\Omega} = (0, 0, 1),
\end{array}\right.
\qquad
\mu_r = \left\{\begin{array}{l}
3, \ \|(x,z) - (0{,}5;\, 0{,}5)\| < 0{,}2, \\
1, \ \text{caso contrário}.
\end{array}\right.$$

Condição de contorno imposta exatamente por uma função de *lift* $\mathbf{H}(x) = f(x)\,d_0(x) + V(x)$. Treino não-supervisionado. A referência numérica é gerada por diferenças finitas sobre o potencial escalar magnético.

## Arquivos

- `run.py` — treina, em paralelo no Modal, PINN $8\times96$ e MixFunn-sof $1\times2$ (com base atômica reduzida a quatro funções).
- `run_refine.py` — refinamento adicional com L-BFGS e um *schedule* mais agressivo de temperatura da softmax.
- `mixfunn.py` — camada Mix2Funn.

## Reprodução

```bash
modal run run.py
modal run run_refine.py
modal volume get tcc /final/baldan_v23 ./results
```

Tempo: ~30 min de wall-time em T4 (`run` + `run_refine` combinados). Custo: ~$0{,}30.
