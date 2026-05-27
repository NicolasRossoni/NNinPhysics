# Experiment 02 — Kovasznay flow: regularization sweep

Produces **Table 3** of `monograph.pdf`.

## Physical problem

Same Kovasznay setup as Experiment 01. The goal here is to measure how two classic regularizers — **dropout** and **collocation-point sub-sampling** — affect the validation loss of the two reference architectures: PINN $4 \times 32$ and MixFunn $3 \times 1$.

## Configurations trained

Six configurations per architecture (12 runs total):

| Configuration         | Dropout | Sub-sample |
| --------------------- | ------- | ---------- |
| baseline              | 0       | 100%       |
| dropout 10%           | 10%     | 100%       |
| dropout 15%           | 15%     | 100%       |
| sub-sample 70%        | 0       | 70%        |
| sub-sample 50%        | 0       | 50%        |
| combo                 | 10%     | 50%        |

All other hyperparameters (learning rate, annealing schedule, optimizer, iterations) match the Experiment 01 baseline.

## Reproduce

```bash
modal run run_extras.py
modal volume get tcc /final/kov_v19_extras ./results
```

Total runtime ≈ 20 min wall on T4 (12 containers in parallel). Cost ≈ $0.20 of Modal credit.

## Files

- `run_extras.py` — Modal entrypoint dispatching every configuration above.
- `mixfunn.py` — MixFunn implementation (identical to Experiment 01).

## Result summary

In the PINN, dropout actually **worsens** the loss by two orders of magnitude (regularization is harmful when the network is already underfit); sub-sampling is essentially neutral. In the MixFunn, sub-sampling 70% delivers the best regime (one order of magnitude better than baseline), while dropout is again harmful. See §3.1 of the monograph for the full discussion.
