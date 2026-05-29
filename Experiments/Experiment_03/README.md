# Experimento 3 — Pruning de MixFunn em Kovasznay

> Produz a **Tabela 4** de `monograph.pdf`.

## Problema

Equacoes incompressiveis de Navier-Stokes 2D estacionarias, com solucao
analitica de Kovasznay:

```
u(x,y) = 1 − e^{λ x} cos(2π y)
v(x,y) = (λ / 2π) e^{λ x} sin(2π y)
p(x,y) = (1 − e^{2 λ x}) / 2
λ = Re/2 − sqrt((Re/2)^2 + 4π^2),    Re = 40
```

Dominio: `(x, y) ∈ [−0.5, 1.0] × [−0.5, 1.5]`. Condicao de contorno
imposta de forma forte por interpolacao transfinita de Coons. Treino
nao-supervisionado, semente unica `21`, 15 000 iteracoes Adam com
annealing linear da temperatura `T` do softmax (`5.0 → 0.05`).

Aplica-se *pruning* por magnitude aos pesos $\alpha_k$ do softmax em
duas variantes da MixFunn e em cinco razoes cada:

- **MixFunn 3x1** (35 pesos $\alpha$, `second_order_function=False`):
  razoes $r \in \{0,\,0{,}30,\,0{,}50,\,0{,}70,\,0{,}90\}$.
- **MixFunn-sof 1x1** (105 pesos $\alpha$, `second_order_function=True`):
  mesmas cinco razoes.

Para cada razao zera-se os $r \cdot N$ pesos de menor magnitude, mede-se
$L^2_{\rm val}$ no grid 80x100, e restauram-se os $\alpha$ antes da
proxima razao (avaliacao independente por razao).

## Arquivos

- `1_preprocess.py` — gera o grid de colocacao por Hipercubo Latino e
  os campos analiticos no grid de validacao. Saida em
  `tcc:/preprocess/exp_03/`.
- `2_train.py` — treina as duas MixFunn, aplica *pruning* em cada
  razao, mede $L^2_{\rm val}$ e salva os checkpoints em
  `tcc:/checkpoints/exp_03/`.
- `3_analyze.py` — baixa os checkpoints localmente, imprime a Tabela 4
  comparando com o PDF, e lista as funcoes atomicas dominantes por
  neuronio da MixFunn 3x1.
- `mixfunn.py` — camada Mix2Funn; inclui `prune_alpha` com mascara
  persistente.

## Reproducao

```bash
modal run 1_preprocess.py
modal run 2_train.py
mkdir tmp_checkpoints
for f in mix_3x1.json mix_3x1.statedict.pt mix_sof_1x1.json mix_sof_1x1.statedict.pt; do
    modal volume get tcc /checkpoints/exp_03/$f ./tmp_checkpoints/$f
done
python 3_analyze.py
```

Tempo: ~25 min de *wall-time* em T4.
