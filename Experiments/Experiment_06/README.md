# Experiment 06 — Nonlinear Schrödinger equation

Produces **Figure 9** of `monograph.pdf`.

## Physical problem

One-dimensional cubic nonlinear Schrödinger equation,

$$
i\,\psi_t + \tfrac{1}{2}\,\psi_{xx} + |\psi|^2\,\psi = 0, \qquad
\psi(x, 0) = 2\,\mathrm{sech}(x), \qquad \psi(\pm 5, t) = 0,
$$

on $t \in [0, \pi/2]$. The $2\,\mathrm{sech}(x)$ initial condition seeds a bright soliton that propagates while preserving its envelope shape.

## Configurations trained

- **PINN 5×100**, two-output ($\Re\psi$, $\Im\psi$) with the residual evaluated on both components.
- **MixFunn-sof 3×6**, same two-output topology.

Both networks train unsupervised on the residual + IC + BC penalties. The reference solution shown in Figure 9 is computed offline by split-step Fourier integration.

The Schrödinger sprint also explored alternative MixFunn bases (e.g., $\{ \sin, \cos \}$ with $K = 2$) and a Raissi-style Adam + L-BFGS pipeline — none beat the $3 \times 6$ MixFunn-sof baseline reported here.

## Reproduce

```bash
modal run batch1.py                                 # PINN 5x100
modal run batch2.py                                 # MixFunn-sof 3x6 baseline
modal run batch3.py                                 # variant sweep (alternative bases)
modal run batch4.py                                 # Adam + L-BFGS pipeline
modal volume get tcc /final/schrod_v22 ./results
```

Total runtime ≈ 30 min wall on T4. Cost ≈ $0.30 of Modal credit.

## Files

- `batch1.py` … `batch4.py` — Modal entrypoints; each launches a batch of configurations on the same Modal app.
- `mixfunn.py` — MixFunn implementation.

## Observed behaviour

PINN reproduces the soliton faithfully across the full time horizon. MixFunn-sof underestimates the central peak amplitude but recovers the exponential decay at the boundaries well — see Figure 9 caption and §3.3 of the monograph.

## Reference

Problem statement and reference numerical solution as in RAISSI et al. (2019).
