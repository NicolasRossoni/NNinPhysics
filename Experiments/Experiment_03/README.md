# Experimento 3 — Pruning de MixFunn em Kovasznay

> Produz a **Tabela 4** de `monograph.pdf`.

## Problema

Mesma EDP do [Experimento 1](../Experiment_01/README.md). Aplica-se *pruning* por magnitude aos pesos $\alpha_k$ da softmax, em duas variantes da rede e em cinco razões cada:

- **MixFunn $3\times1$** (35 pesos): razões $r \in \{0;\,0{,}30;\,0{,}50;\,0{,}70;\,0{,}90\}$.
- **MixFunn-sof $1\times1$** (105 pesos): mesmas cinco razões.

## Arquivos

- `run.py` — treina os baselines e aplica *pruning* em cada razão.
- `mixfunn.py` — camada Mix2Funn; inclui o `prune_alpha` com máscara persistente.
- `analyze_mix_params.py` — inspeção post-hoc do `state_dict` para identificar as funções atômicas dominantes por neurônio.

## Reprodução

```bash
modal run run.py
modal volume get tcc /final/kov_v13 ./results
python analyze_mix_params.py
```

Tempo: ~25 min de wall-time em T4. Custo: ~$0{,}30.
