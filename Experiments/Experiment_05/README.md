# Experimento 5 — Equação de Burgers viscosa

> Produz a **Figura 8** de `monograph.pdf`.

## Problema

$$\begin{cases}
u_t + u\,u_x = \nu\,u_{xx}, \\
u(x, 0) = -\sin(\pi x), \\
u(\pm 1, t) = 0,
\end{cases}
\qquad \nu = 0{,}01/\pi, \quad x \in [-1, 1], \quad t \in [0, 1].$$

Treino estritamente não-supervisionado (resíduo da EDP mais condições inicial e de contorno; sem termo de regressão). A solução de referência é gerada por pseudo-espectral Fourier ($N = 256$) com integração temporal Runge–Kutta de quarta ordem ($\Delta t = 10^{-4}$) e depois interpolada para a grade de avaliação $200 \times 100$.

## Arquivos

- `1_preprocess.py` — gera a referência espectral + RK4 e a amostragem Latin-Hypercube dos pontos de colocação. Salva tudo num `.npz` único em `tcc:/preprocess/exp_05/`.
- `2_train.py` — treina PINN $6\times64$ e MixFunn-sof $3\times3$ em paralelo no Modal. Checkpoints em `tcc:/checkpoints/exp_05/`.
- `3_analyze.py` — baixa os checkpoints, calcula $L^2$ final e gera `burgers_v25.png` (três mapas de calor: referência | PINN | MixFunn$_{\rm sof}$).
- `mixfunn.py` — camada Mix2Funn (não editar).

## Reprodução

```bash
modal run 1_preprocess.py
modal run 2_train.py
modal volume get tcc /checkpoints/exp_05 ./tmp_checkpoints
python 3_analyze.py
```

Tempo: ~20 min de wall-time em T4 (duas redes em paralelo).
