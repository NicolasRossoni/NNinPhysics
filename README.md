# Neural Networks in Physics

Companion code repository for the undergraduate thesis (TCC) **"Redes neurais aplicadas na física: usando PINN e MixFunn para resolver EDPs"**, Instituto de Física de São Carlos / USP, 2026.

The thesis compares two physics-informed neural network architectures — **PINN** (Raissi et al., 2019) and **MixFunn / Mix2Funn** (Farias et al., 2025) — across a range of partial differential equation (PDE) problems, from a deep dive on the Kovasznay flow (analytical Navier–Stokes 2D) to a real intracranial aneurysm reconstruction, plus three additional benchmarks (Burgers, nonlinear Schrödinger, magnetostatics with discontinuous media).

The full text (in Portuguese) is in `monograph.pdf`.

## Repository layout

```
NNinPhysics/
├── README.md             # this file
├── monograph.pdf         # the thesis text (Portuguese)
└── Experiments/
    ├── Experiment_01/    # Kovasznay deep dive (Tab. 1, 2; Fig. 4, 5)
    ├── Experiment_02/    # Kovasznay regularization (Tab. 3)
    ├── Experiment_03/    # Kovasznay pruning (Tab. 4)
    ├── Experiment_04/    # 3D aneurysm flow on ANEUMO dataset (Tab. 5; Fig. 6, 7)
    ├── Experiment_05/    # Viscous Burgers equation (Fig. 8)
    ├── Experiment_06/    # Nonlinear Schrödinger equation (Fig. 9)
    └── Experiment_07/    # 2D magnetostatics with dielectric disc (Fig. 10)
```

Each experiment folder contains its own `README.md` with the physical problem, the architecture used, the command to reproduce the result, and pointers to the figure or table in `monograph.pdf` that it generates.

## Setup

All experiments run on **Modal** (https://modal.com), a serverless cloud platform with on-demand GPU containers. The local machine only needs to dispatch jobs and post-process artifacts — no local GPU required.

### 1. Modal account

Create a free Modal account at https://modal.com/signup. As of May 2026, the free tier includes:

- **$5 of free credits** for any new account.
- **+$30 of free credits** when you add a payment method. Setting a spending cap of $30 prevents any actual charges while unlocking the larger credit pool.

Total cost of reproducing every experiment in this repository fits well inside the free tier (each run uses a T4 GPU at ~$0.59/hour, and most experiments finish in 10–60 minutes).

### 2. Local environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install modal torch numpy scipy matplotlib
modal token new            # link the local CLI to your account
```

### 3. Modal workspace and volume

Each experiment writes outputs to a persistent Modal volume named `tcc`. Create it once:

```bash
modal volume create tcc
```

The Python scripts reference the active Modal workspace via `modal.App(...)`. By default this uses the workspace returned by `modal profile current`; change with `modal profile activate <name>` if you have multiple workspaces.

### 4. Reproducing an experiment

Each folder includes its own instructions. As a general pattern:

```bash
cd Experiments/Experiment_05
modal run run.py            # launches the GPU job
modal volume get tcc /final/burgers_v22 ./output   # fetch artifacts
python plot.py              # local post-processing (if present)
```

## Methodology summary

- **PINN**: a multilayer perceptron $u_\theta(x, t)$ whose loss is the PDE residual sampled at collocation points plus initial- and boundary-condition penalties. Training is unsupervised — no reference solution is required.
- **MixFunn / Mix2Funn**: a neuron design where the activation is a softmax-weighted combination of a fixed analytic basis $\{ \sin, \cos, e^z, \log|z|, z, z^2, 1 \}$, with optional second-order cross-products of basis functions. After training, the surviving weights yield an interpretable symbolic expression.

Both architectures are implemented exactly as in the original papers, without any extra regularization that would compromise the comparison.

## Citing this work

If you use this code or build upon the experiments, please cite the underlying papers:

1. RAISSI, M.; PERDIKARIS, P.; KARNIADAKIS, G. E. Physics-informed neural networks: a deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, v. 378, p. 686–707, 2019. https://doi.org/10.1016/j.jcp.2018.10.045
2. FARIAS, T. S.; LIMA, G. G.; MAZIERO, J.; VILLAS-BOAS, C. J. MixFunn: a neural network for differential equations with improved generalization and interpretability. *arXiv preprint arXiv:2503.22528*, 2025. https://arxiv.org/abs/2503.22528
3. GOODFELLOW, I.; BENGIO, Y.; COURVILLE, A. *Deep Learning*. Cambridge: MIT Press, 2016. https://www.deeplearningbook.org
4. RAISSI, M.; YAZDANI, A.; KARNIADAKIS, G. E. Hidden fluid mechanics: learning velocity and pressure fields from flow visualizations. *Science*, v. 367, n. 6481, p. 1026–1030, 2020. https://arxiv.org/abs/1808.04327
5. LI, Y. et al. ANEUMO: a high-quality dataset for aneurysm-related cerebral hemodynamics. *arXiv preprint arXiv:2505.14717*, 2025. https://arxiv.org/abs/2505.14717
6. NOHRA, M.; DUFOUR, S. Physics-informed neural networks for the numerical modeling of steady-state and transient electromagnetic problems with discontinuous media. *arXiv preprint arXiv:2406.04380*, 2024. https://arxiv.org/abs/2406.04380
7. MIRANDA, G. C. de; LIMA, G. G. de; FARIAS, T. S. An introduction to neural networks for physicists. *arXiv preprint arXiv:2505.13042*, 2025. https://arxiv.org/abs/2505.13042

## Author

Nicolas Oliveira Rossoni — Instituto de Física de São Carlos / USP. Advisor: Prof. Dr. Celso Jorge Villas-Boas (UFSCar).
