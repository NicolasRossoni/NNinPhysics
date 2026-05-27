# Experimento 1 — Escoamento de Kovasznay

> Produz a **Tabela 1**, a **Tabela 2**, a **Figura 4** e a **Figura 5** de `monograph.pdf`.

## Problema

Navier–Stokes incompressível 2D estacionário, com solução analítica fechada de Kovasznay:

$$\left\{\begin{array}{l}
u\,u_x + v\,u_y = -p_x + \mathrm{Re}^{-1}\,(u_{xx} + u_{yy}), \\
u\,v_x + v\,v_y = -p_y + \mathrm{Re}^{-1}\,(v_{xx} + v_{yy}), \\
u_x + v_y = 0,
\end{array}\right.
\qquad \Omega = [-0{,}5;\,1] \times [-0{,}5;\,1{,}5], \quad \mathrm{Re} = 40.$$

Condição de contorno imposta exatamente via interpolação transfinita de Coons.

## Arquivos

- `run.py` — despacha em paralelo no Modal todas as configurações da Tabela 1 e da Tabela 2 (sup / não-sup × 3 seeds, varredura de profundidade e largura para PINN e MixFunn, além da MixFunn-sof $1\times1$ usada na análise da Eq. 11 do monograph).
- `mixfunn.py` — camada Mix2Funn (empilhável, com `second_order_function` opcional e annealing da temperatura da softmax).
- `aggregate.py` — consolida as métricas salvas no volume Modal nas tabelas do monograph.
- `figures.py` — gera a Figura 4 (campos interp/extrap) e a Figura 5 (decomposição da loss em Momento e Massa).

## Reprodução

```bash
modal run run.py
modal volume get tcc /final/kov_v14 ./results
python aggregate.py
python figures.py
```

Tempo: ~30–45 min de wall-time em T4 (containers paralelos). Custo: ~$0{,}50.
