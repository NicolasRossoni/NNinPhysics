# Experiment 07 — 2D magnetostatics with a paramagnetic disc

Produces **Figure 10** of `monograph.pdf`.

## Physical problem

Sourceless Maxwell equations in 2D, on the unit square, with a discontinuous permeability:

$$
\nabla \times \mathbf{H} = 0, \qquad
\nabla \cdot (\mu\,\mathbf{H}) = 0, \qquad
\mathbf{H}|_{\partial\Omega} = (0, 0, 1),
$$

with $\mu_r = 3$ inside a disc of radius $0.2$ centered at $(0.5, 0.5)$, and $\mu_r = 1$ elsewhere. The disc deforms the otherwise uniform field, and the discontinuity at $\partial(\mathrm{disc})$ must be tracked through the divergence-free constraint on $\mu \mathbf{H}$.

## Configurations trained

- **PINN 8×96** with a lift function that enforces $\mathbf{H}|_{\partial\Omega}$ exactly without a boundary penalty.
- **MixFunn-sof 1×2** with a 4-function atomic basis $\{ \sin, \cos, z, z^2 \}$ tuned to the symmetry of the dipolar response.

Both networks train unsupervised on the residual of the two Maxwell constraints. The reference solution shown in Figure 10 is computed offline by a finite-difference solver on a scalar magnetic potential (problem from NOHRA & DUFOUR, 2024).

## Reproduce

```bash
modal run run.py                                    # initial training
modal run run_refine.py                             # extended fine-tuning
modal volume get tcc /final/baldan_v23 ./results
```

Total runtime ≈ 30 min wall on T4 (run + refine combined). Cost ≈ $0.30 of Modal credit.

## Files

- `run.py` — Modal entrypoint for the baseline training of PINN and MixFunn-sof.
- `run_refine.py` — second-stage fine-tuning with a tighter learning-rate schedule.
- `mixfunn.py` — MixFunn implementation.

## Observed behaviour

PINN recovers the perturbed field pattern with high fidelity. MixFunn-sof reaches a slightly higher loss but captures the qualitative dipolar structure near the disc — see Figure 10 caption and §3.3 of the monograph.

## Reference

Problem statement, parameters, and reference numerical solution from NOHRA & DUFOUR (2024).
