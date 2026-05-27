# Experiment 05 — Viscous Burgers equation

Produces **Figure 8** of `monograph.pdf`.

## Physical problem

One-dimensional viscous Burgers equation,

$$
u_t + u\,u_x = \nu\,u_{xx}, \qquad
u(x, 0) = -\sin(\pi x), \qquad u(\pm 1, t) = 0,
$$

with $\nu = 0.01 / \pi$, on $x \in [-1, 1]$ and $t \in [0, 1]$. The initial sinusoid steepens into a viscous-shock-like profile near $x = 0$ as $t \to 1$.

## Configurations trained

- **PINN 6×64** with $\tanh$ activations.
- **MixFunn-sof 3×6** with the standard analytic basis $\{ \sin, \cos, e^z, \log|z|, z, z^2, 1 \}$ and second-order cross-products enabled.

Both networks are trained **fully unsupervised** using the PDE residual plus the initial- and boundary-condition penalties — no reference solution enters the loss. The reference solution shown in Figure 8 (for comparison only) is computed offline with a fourth-order Runge–Kutta time-stepper on a finite-difference spatial grid.

## Reproduce

```bash
modal run run.py                                # trains PINN + MixFunn-sof in parallel
modal volume get tcc /final/burgers_v22 ./results
```

Total runtime ≈ 15 min wall on T4. Cost ≈ $0.15 of Modal credit.

## Files

- `run.py` — Modal entrypoint for both architectures.
- `mixfunn.py` — MixFunn implementation.

## Reference

Problem statement and reference numerical solution as in RAISSI et al. (2019).
