# Experimento 1 — Escoamento de Kovasznay

> Produz a **Tabela 1**, a **Tabela 2**, a **Figura 4** e a **Figura 5** de `monograph.pdf`.

## Problema

Navier–Stokes incompressível 2D estacionário, com solução analítica fechada de Kovasznay:

$$\begin{cases}
u\,u_x + v\,u_y = -p_x + \mathrm{Re}^{-1}\,(u_{xx} + u_{yy}), \\
u\,v_x + v\,v_y = -p_y + \mathrm{Re}^{-1}\,(v_{xx} + v_{yy}), \\
u_x + v_y = 0,
\end{cases}
\qquad \Omega = [-0{,}5;\,1] \times [-0{,}5;\,1{,}5], \quad \mathrm{Re} = 40.$$

Solução analítica:

$$u(x,y) = 1 - e^{\lambda x}\cos(2\pi y), \quad
v(x,y) = \tfrac{\lambda}{2\pi}\,e^{\lambda x}\sin(2\pi y), \quad
p(x,y) = \tfrac{1}{2}(1 - e^{2\lambda x}),$$

com $\lambda = \mathrm{Re}/2 - \sqrt{(\mathrm{Re}/2)^2 + 4\pi^2} \approx -0{,}964$. Condição de contorno imposta exatamente via interpolação transfinita de Coons.

## Arquivos

- `1_preprocess.py` — monta os pontos de colocação (Hipercubo Latino) e as grades de validação / avaliação / extrapolação; faz upload em `tcc:/preprocess/exp_01/`.
- `2_train.py` — treina em paralelo no Modal T4 as 31 configurações (Tabela 1: PINN $4\times32$ e MixFunn $3\times1$, sup e nsup, 3 seeds; Tabela 2: varredura $N\times n$ para PINN e MixFunn; MixFunn $1\times1$ com `sof=True`). Salva métricas + checkpoints em `tcc:/checkpoints/exp_01/`.
- `3_analyze.py` — lê os checkpoints baixados em `./tmp_checkpoints/`, imprime Tabela 1 e Tabela 2 no console e gera `kov_v12_fields_extrap.png` e `kov_v12_loss_decomp.png`.
- `mixfunn.py` — camada Mix2Funn (empilhável, com `second_order_function` opcional e annealing da temperatura da softmax).

## Reprodução

```bash
modal run 1_preprocess.py
modal run 2_train.py
modal volume get tcc /checkpoints/exp_01 ./tmp_checkpoints
python 3_analyze.py
```

## Tempo

31 jobs em paralelo no Modal T4. `ITER_T1 = 15000` (Tabela 1) e `ITER_T2 = 10000` (sweep). Wall-clock ~ 15–25 min por job.
