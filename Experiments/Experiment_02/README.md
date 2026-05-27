# Experimento 2 — Regularizacao em Kovasznay

> Produz a **Tabela 3** de `monograph.pdf`.

## Problema

Mesma EDP do [Experimento 1](../Experiment_01/README.md). Escoamento de Kovasznay (Navier-Stokes incompressivel 2D estacionario, Re = 40), com solucao analitica fechada:

$$
\begin{cases}
u\,u_x + v\,u_y = -p_x + \mathrm{Re}^{-1}\,(u_{xx} + u_{yy}), \\
u\,v_x + v\,v_y = -p_y + \mathrm{Re}^{-1}\,(v_{xx} + v_{yy}), \\
u_x + v_y = 0,
\end{cases}
\qquad \Omega = [-0{,}5;\,1] \times [-0{,}5;\,1{,}5].
$$

Condicao de contorno imposta exatamente via interpolacao transfinita de Coons. Variam-se dois mecanismos de regularizacao em duas redes (PINN $4\times32$ e MixFunn $3\times1$), totalizando **12 configuracoes**:

| Configuracao        | Dropout | Sub-amostra |
|---------------------|--------:|------------:|
| baseline            |       0 |        100% |
| dropout 10%         |     10% |        100% |
| dropout 15%         |     15% |        100% |
| sub-amostra 70%     |       0 |         70% |
| sub-amostra 50%     |       0 |         50% |
| combo               |     10% |         50% |

Semente unica: `SEED = 21`.

## Arquivos

- `1_preprocess.py` — gera a nuvem Latin-Hypercube de 4000 pontos de colocacao e a malha de validacao de referencia; envia para o volume `tcc:/preprocess/exp_02/`.
- `2_train.py` — entrypoint Modal (`@app.function train_one` + `@app.local_entrypoint main`); despacha as 12 configuracoes em paralelo (T4) e grava checkpoints + metricas em `tcc:/checkpoints/exp_02/by_label/`.
- `3_analyze.py` — le os checkpoints baixados localmente e imprime a Tabela 3 comparada contra os valores do PDF.
- `mixfunn.py` — camada Mix2Funn (softmax + annealing de temperatura).

## Reproducao

```bash
modal run 1_preprocess.py
modal run 2_train.py
modal volume get tcc /checkpoints/exp_02 ./tmp_checkpoints
python 3_analyze.py
```

Se o `modal` nao estiver no PATH: `~/.pyenv/versions/3.13.3/envs/venv/bin/modal`.

Tempo: ~25 min de wall-time em T4 (12 containers em paralelo). Custo estimado: ~$0,30.
