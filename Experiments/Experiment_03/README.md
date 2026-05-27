# Experiment 03 — Kovasznay flow: MixFunn pruning

Produces **Table 4** of `monograph.pdf`.

## Physical problem

Same Kovasznay setup as Experiment 01. The goal here is to evaluate **magnitude-based pruning** of the MixFunn softmax weights $\alpha_k$: after a baseline training, weights with magnitude below a percentile threshold are zeroed and the network is fine-tuned, measuring how much of the expressivity each ratio sacrifices.

## Configurations trained

Two MixFunn variants, each at five pruning ratios:

- **MixFunn 3×1** (35 trainable weights): ratios $r \in \{0, 0.30, 0.50, 0.70, 0.90\}$.
- **MixFunn-sof 1×1** (105 trainable weights): same five ratios.

The sof variant has the second-order cross-product term enabled, which is what allows a single-layer network to represent the $1 - e^{\lambda x} \cos(2\pi y)$ product structure of the Kovasznay solution (Equation 11 in the monograph).

## Reproduce

```bash
modal run run_v13.py                               # trains baseline + 5 pruning ratios
modal volume get tcc /final/kov_v13 ./results
python analyze_mix_params.py                       # post-hoc inspection of state_dict
```

Total runtime ≈ 25 min wall on T4. Cost ≈ $0.30 of Modal credit.

## Files

- `run_v13.py` — Modal entrypoint, dispatches one container per (variant, ratio) pair.
- `mixfunn.py` — MixFunn implementation; includes the `prune_alpha` routine that masks the lowest-magnitude weights and re-tunes.
- `analyze_mix_params.py` — post-hoc analysis: identifies dominant atomic basis functions per neuron, formats the surviving symbolic combination.

## Result summary

About 30% of the parameters are essentially redundant in both variants: loss stays in the same order of magnitude up to $r = 0.30$. From $r = 0.50$ onward the loss degrades monotonically, but never enough that the surviving weights form a single dominant analytical expression in this experimental budget — closing that gap would require more aggressive annealing or a smaller atomic basis, both of which risk introducing lookahead bias.
