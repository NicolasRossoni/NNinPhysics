# Experiment 04 — 3D intracranial aneurysm flow (ANEUMO)

Produces **Table 5**, **Figure 6** and **Figure 7** of `monograph.pdf`.

## Physical problem

Steady incompressible Navier–Stokes in 3D on a real intracranial aneurysm geometry from the **ANEUMO** dataset (Li et al., 2025), case `AN4` at mass-flow factor $m = 0.002$. The domain is a point cloud (~129k points) reconstructed from a real patient's cerebrovascular tree; reference fields (velocity and pressure) were pre-computed by the dataset authors with a commercial CFD solver.

The governing equations are

$$
(\mathbf{u} \cdot \nabla)\mathbf{u} + \nabla p / \rho - \nu \nabla^2 \mathbf{u} = 0, \qquad
\nabla \cdot \mathbf{u} = 0,
$$

with $\mathbf{u} = 0$ on the vessel wall, the physiological velocity profile from the dataset at the inlet, and $p = 0$ at the outlet.

## Configurations trained

Nine configurations spanning unsupervised, partially-supervised, and fully-supervised regimes:

- PINN 5×64, MixFunn 3×1, MixFunn-sof 3×1 at **0%** supervision (unsupervised).
- PINN 8×128 at **0%** supervision (larger unsupervised attempt).
- PINN 5×64, MixFunn-sof 2×2 at **25%** and **50%** supervision.
- PINN 5×64 and MixFunn-sof 2×2 at **100%** supervision (best-performing, shown in figures).

See §3.2 of the monograph for the discussion of why the unsupervised regime fails at the cerebrovascular Reynolds number with the budget used here, and the pointers to future work (curriculum learning over $\mathrm{Re}$).

## Reproduce

```bash
python parse_aneumo.py           # downloads ANEUMO dataset and extracts case AN4
modal run train_networks.py      # trains every configuration in Table 5
modal volume get tcc /final/aneurisma ./results
python plot_predictions.py       # generates Figure 6 (3D panel) and Figure 7 (central plane)
python validate_aneumo.py        # sanity-checks predictions against reference fields
```

Total runtime ≈ 60–90 min wall on T4 across the nine containers. Cost ≈ $1 of Modal credit.

The ANEUMO dataset is openly available at https://arxiv.org/abs/2505.14717.

## Files

- `parse_aneumo.py` — downloads and parses the ANEUMO point cloud for case `AN4`.
- `train_networks.py` — Modal entrypoint dispatching the nine training jobs.
- `run_aneur_pinn_unsup.py` — standalone unsupervised training script for the PINN.
- `plot_predictions.py` — generates the 3D panel (Figure 6) and the central plane slice (Figure 7).
- `plot_aneumo.py` — visualisations of the raw dataset (geometry + reference field).
- `validate_aneumo.py` — numeric validation against ground truth.

## Reference

Patient data and CFD reference fields from LI et al. (2025); architecture choice and partial-supervision sweep follow the structure of RAISSI et al. (2018, Hidden Fluid Mechanics) adapted to the ANEUMO geometry.
