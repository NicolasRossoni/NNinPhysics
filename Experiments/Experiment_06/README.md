# Experimento 6 — Equação de Schrödinger não-linear

> Produz a **Figura 9** de `monograph.pdf`.

## Problema

$$\left\{\begin{array}{l}
i\,\psi_t + \tfrac{1}{2}\,\psi_{xx} + |\psi|^2\,\psi = 0, \\
\psi(x, 0) = 2\,\mathrm{sech}(x), \\
\psi(\pm 5, t) = 0,
\end{array}\right. \qquad t \in [0,\, \pi/2].$$

Rede de duas saídas $(\Re\psi,\, \Im\psi)$; treino não-supervisionado. A referência numérica é obtida por integração split-step Fourier.

## Arquivos

A configuração do monograph foi escolhida em uma sprint exploratória de quatro lotes sequenciais — cada lote leu o resultado do anterior antes de definir suas configurações:

- `batch1.py` — varredura inicial de PINN em diferentes (profundidade, largura).
- `batch2.py` — extensão de iterações para as PINNs que ainda decresciam.
- `batch3.py` — primeiras configurações de MixFunn-sof e uma PINN com iterações equivalentes para comparação.
- `batch4.py` — PINN no número de iterações faltante para o pareamento direto com a melhor MixFunn-sof.
- `mixfunn.py` — camada Mix2Funn.

## Reprodução

```bash
modal run batch1.py
modal run batch2.py
modal run batch3.py
modal run batch4.py
modal volume get tcc /final/schrod_v22 ./results
```

Tempo: ~30 min de wall-time em T4 (lotes em sequência; cada lote despacha jobs em paralelo). Custo: ~$0{,}30.

> Apenas `batch3.py` e `batch4.py` são necessários para reproduzir as duas configurações finais que aparecem na Figura 9; os dois primeiros lotes fazem parte do diário da exploração.
