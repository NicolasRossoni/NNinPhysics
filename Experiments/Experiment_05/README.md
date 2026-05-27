# Experimento 5 — Equação de Burgers viscosa

> Produz a **Figura 8** de `monograph.pdf`.

## Problema

$$\left\{\begin{array}{l}
u_t + u\,u_x = \nu\,u_{xx}, \\
u(x, 0) = -\sin(\pi x), \\
u(\pm 1, t) = 0,
\end{array}\right.
\qquad \nu = 0{,}01/\pi, \quad x \in [-1, 1], \quad t \in [0, 1].$$

Treino estritamente não-supervisionado (resíduo da EDP mais condições inicial e de contorno; sem termo de regressão). A referência numérica usada na figura é gerada por Runge–Kutta de quarta ordem em uma grade de diferenças finitas.

## Arquivos

- `run.py` — despacha PINN $6\times64$ e MixFunn-sof $3\times6$ no Modal.
- `mixfunn.py` — camada Mix2Funn.

## Reprodução

```bash
modal run run.py
modal volume get tcc /final/burgers_v22 ./results
```

Tempo: ~15 min de wall-time em T4. Custo: ~$0{,}15.
