# Experimento 6 — Equação de Schrödinger não-linear

> Produz a **Figura 9** de `monograph.pdf`.

## Problema

$$\left\{\begin{array}{l}
i\,\psi_t + \tfrac{1}{2}\,\psi_{xx} + |\psi|^2\,\psi = 0, \\
\psi(x, 0) = 2\,\mathrm{sech}(x), \\
\psi(\pm 5, t) = 0,
\end{array}\right. \qquad t \in [0,\, \pi/2].$$

Rede de duas saídas $(\Re\psi,\, \Im\psi)$; treino não-supervisionado (resíduo da EDP + penalidade da condição inicial + penalidade das condições de contorno). A referência numérica é obtida por integração split-step Fourier de Strang ($N=512$ modos, $\Delta t = 5 \times 10^{-5}$).

## Arquivos

- `1_preprocess.py` — calcula a solução de referência split-step Fourier no grid $200 \times 100$ e amostra os pontos de colocação via Hipercubo Latino, salva em `tcc:/preprocess/exp_06/`.
- `2_train.py` — entrypoint Modal `nnphysics-exp06`. Despacha em paralelo as duas configurações finais (PINN $8\times 100$ não-supervisionada com $25{,}000$ iterações e MixFunn$_{\rm sof}$ $3\times 6$ não-supervisionada com $10{,}000$ iterações), Adam com `StepLR` ($\gamma=0{,}5$ a cada $25\%$ do treino). Salva checkpoints em `tcc:/checkpoints/exp_06/by_label/`.
- `3_analyze.py` — lê os checkpoints baixados localmente, imprime a tabela de $L^2$ comparando contra a referência e gera `schrod_v25.png` (três mapas de calor de $|\psi|$ na linha superior e quatro cortes em $t$ na inferior).
- `mixfunn.py` — camada Mix2Funn (empilhável, com `second_order_function=True` e annealing da temperatura da softmax).

## Reprodução

```bash
modal run 1_preprocess.py
modal run 2_train.py
modal volume get tcc /checkpoints/exp_06 ./tmp_checkpoints
python 3_analyze.py
```

Tempo: ~25 min de wall-time em T4 (dois containers paralelos). Custo: ~$0{,}20.
