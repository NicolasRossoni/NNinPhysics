# Experimento 2 — Regularização em Kovasznay

> Produz a **Tabela 3** de `monograph.pdf`.

## Problema

Mesma EDP do [Experimento 1](../Experiment_01/README.md). Varia-se *dropout* e sub-amostragem de pontos de colocação na PINN $4\times32$ e na MixFunn $3\times1$ (12 configurações no total):

| Configuração | Dropout | Sub-amostra |
|---|---|---|
| baseline | 0 | 100% |
| dropout 10% | 10% | 100% |
| dropout 15% | 15% | 100% |
| sub-amostra 70% | 0 | 70% |
| sub-amostra 50% | 0 | 50% |
| combo | 10% | 50% |

## Arquivos

- `run_extras.py` — despacha as 12 configurações em paralelo no Modal.
- `mixfunn.py` — camada Mix2Funn.

## Reprodução

```bash
modal run run_extras.py
modal volume get tcc /final/kov_v19_extras ./results
```

Tempo: ~20 min de wall-time em T4. Custo: ~$0{,}20.
