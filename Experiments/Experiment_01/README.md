# Experiment 01 — Kovasznay flow (deep dive)

Produces **Table 1**, **Table 2**, **Figure 4** and **Figure 5** of `monograph.pdf`, plus the MixFunn-sof analysis paragraph in §3.1.

## Physical problem

Steady incompressible Navier–Stokes in 2D with the analytical Kovasznay solution. The closed-form fields on $(x, y) \in [-0.5, 1] \times [-0.5, 1.5]$ at $\mathrm{Re} = 40$ are

$$
u = 1 - e^{\lambda x} \cos(2\pi y), \qquad
v = \tfrac{\lambda}{2\pi} e^{\lambda x} \sin(2\pi y), \qquad
p = \tfrac{1}{2}(1 - e^{2\lambda x}),
$$

with $\lambda = \mathrm{Re}/2 - \sqrt{(\mathrm{Re}/2)^2 + 4\pi^2} \approx -0.964$. Boundary conditions are enforced exactly via transfinite Coons interpolation, so the residual loss never sees a leak of the analytical solution at the interior.

## Configurations trained

- PINN **4×32** baseline (Table 1, Figure 4, Figure 5).
- MixFunn **3×1** baseline (Table 1, Figure 4).
- PINN architecture sweep on a $3 \times 3$ grid in (depth, width): $N \in \{4, 6, 8\}$, $n \in \{16, 32, 64\}$ (Table 2 left).
- MixFunn architecture sweep on a $3 \times 3$ grid in (depth, width): $N \in \{1, 2, 3\}$, $n \in \{1, 2, 3\}$ (Table 2 right).
- MixFunn-sof **1×1** (Equation 11 in the monograph), to verify that the $5.6 \times 10^{-2}$ ceiling of single-layer MixFunns is a structural product-expressivity issue (text in §3.1).

All runs use Adam, learning rate $10^{-3}$ for PINN and $10^{-2}$ for MixFunn, with a step scheduler and softmax temperature annealing on the MixFunn side.

## Reproduce

```bash
modal run run_v14.py                                # launch every config in parallel
modal volume get tcc /final/kov_v14 ./results       # fetch artifacts
python aggregate.py                                 # consolidate metrics
python figures_v14.py                               # generate figures (PNG + LaTeX patch)
```

Total runtime ≈ 30–45 min wall on T4 (containers run in parallel). Cost ≈ $0.50 of Modal credit.

## Files

- `run_v14.py` — Modal entrypoint: dispatches every configuration listed above.
- `mixfunn.py` — MixFunn / Mix2Funn implementation (Farias 2025), supports `sof=True/False`, softmax annealing, alpha pruning.
- `figures_v14.py` — produces Figure 4 (interp/extrap fields) and Figure 5 (loss decomposition into momentum vs mass).
- `aggregate.py` — aggregates per-seed losses into the format used by Tables 1 and 2.

## Reference

Closed-form fields and physical parameters from KOVASZNAY, L. I. G. (1948). PINN methodology from RAISSI et al. (2019). MixFunn architecture from FARIAS et al. (2025).
