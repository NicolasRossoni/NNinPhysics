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

Condição de contorno imposta exatamente por uma função de *lift* $\mathbf{H}(x) = f(x)\,d_0(x) + V(x)$, com $V = (0,1)$ e $d_0 = 1 - \exp(-10\,d_{\partial\Omega})$. Treino não-supervisionado (apenas resíduo da EDP). A referência numérica é obtida por diferenças finitas sobre o potencial escalar magnético, resolvendo $\nabla\cdot(\mu\,\nabla\phi) = 0$ com $\phi = -z$ no bordo.

Resíduos:
- $f_1 = \partial_z H_x - \partial_x H_z$ (rotacional em $y$);
- $f_2 = \partial_x(\mu H_x) + \partial_z(\mu H_z)$ (divergência de $\mathbf{B}$);
- $\mathcal{L} = \mathrm{mean}(f_1^2) + \mathrm{mean}(f_2^2)$.

## Arquivos

- `1_preprocess.py` — gera a referência FD para o potencial escalar magnético ($\mu_r = 3$ no disco), o hipercubo latino de pontos interiores e os metadados. Salva em `tcc:/preprocess/exp_07/`.
- `2_train.py` — treina, em paralelo no Modal, **PINN 8×96 com lift hard-BC** e **MixFunn-sof 3×3 com a base atômica canônica completa de sete funções** ($\sin$, $\cos$, $e^{-x}$, $e^{x}$, $\sqrt{\cdot}$, $\log$, identidade) e produtos cruzados de segunda ordem ativados. Cada configuração roda Adam (~10k iterações) seguido de L-BFGS (~300 iterações externas) na mesma função. Salva em `tcc:/checkpoints/exp_07/`.
- `3_analyze.py` — lê os checkpoints baixados, computa o $L^2$ relativo por componente e produz `baldan_v25.png` (duas linhas, $H_x$ e $H_z$; três colunas, FD / PINN / MixFunn).
- `mixfunn.py` — camada Mix2Funn (mesmo arquivo dos demais experimentos).

## Reprodução

```bash
modal run 1_preprocess.py
modal run 2_train.py
modal volume get tcc /checkpoints/exp_07 ./tmp_checkpoints
python 3_analyze.py
```

Tempo: ~25–35 min de wall-time em T4 (dois containers em paralelo, Adam + L-BFGS). Custo: ~$0{,}30.
