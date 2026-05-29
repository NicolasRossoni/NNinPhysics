# Neural Networks in Physics

Repositório companheiro do Trabalho de Conclusão de Curso **"Redes neurais aplicadas na física: usando PINN e MixFunn para resolver EDPs"**, Instituto de Física de São Carlos / USP, 2026.

O texto completo está em `monograph.pdf`. Cada pasta em `Experiments/` reproduz uma ou mais figuras / tabelas do monograph.

## Layout

```
NNinPhysics/
├── README.md              # este arquivo
├── monograph.pdf          # texto integral do TCC (PT-BR)
└── Experiments/
    ├── Experiment_01/     # Kovasznay (Tab. 1, 2 ; Fig. 4, 5)
    ├── Experiment_02/     # Kovasznay — regularização (Tab. 3)
    ├── Experiment_03/     # Kovasznay — pruning (Tab. 4)
    ├── Experiment_04/     # Aneurisma 3D — ANEUMO (Tab. 5 ; Fig. 6, 7)
    ├── Experiment_05/     # Burgers viscosa (Fig. 8)
    ├── Experiment_06/     # Schrödinger não-linear (Fig. 9)
    └── Experiment_07/     # Magnetostática 2D — Baldan (Fig. 10)
```

Cada experimento traz seu próprio `README.md` (problema, arquivos, comando de reprodução e tempo estimado).

## Pré-requisitos

Todos os treinos rodam em containers efêmeros com GPU NVIDIA T4 na **Modal** (<https://modal.com>); a máquina local só despacha os jobs e busca os artefatos. Nenhuma GPU local é necessária.

### Conta Modal (gratuita)

Cadastre-se em <https://modal.com/signup>. O *tier* gratuito da Modal é suficiente para reproduzir todos os experimentos deste repositório.

### Ambiente local

```bash
python3 -m venv venv
source venv/bin/activate
pip install modal torch numpy scipy matplotlib
modal token new
modal volume create tcc
```

## Padrão de uso

```bash
cd Experiments/Experiment_05
modal run 1_preprocess.py
modal run 2_train.py
modal volume get tcc /checkpoints/exp_05 ./tmp_checkpoints
python 3_analyze.py
```

Cada pasta indica seu próprio caminho no volume Modal e os arquivos de pós-processamento (quando existem).

## Referências

1. RAISSI, M.; PERDIKARIS, P.; KARNIADAKIS, G. E. Physics-informed neural networks: a deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. **Journal of Computational Physics**, v. 378, p. 686–707, 2019. Disponível em: https://doi.org/10.1016/j.jcp.2018.10.045.
2. FARIAS, T. de S.; LIMA, G. G. de; MAZIERO, J.; VILLAS-BOAS, C. J. MixFunn: a neural network for differential equations with improved generalization and interpretability. **arXiv preprint arXiv:2503.22528**, 2025. Disponível em: https://arxiv.org/abs/2503.22528.
3. GOODFELLOW, I.; BENGIO, Y.; COURVILLE, A. **Deep Learning**. Cambridge: MIT Press, 2016. Disponível em: https://www.deeplearningbook.org.
4. RAISSI, M.; YAZDANI, A.; KARNIADAKIS, G. E. Hidden fluid mechanics: learning velocity and pressure fields from flow visualizations. **Science**, v. 367, n. 6481, p. 1026–1030, 2020. Disponível em: https://arxiv.org/abs/1808.04327.
5. LI, Y. et al. ANEUMO: a high-quality dataset for aneurysm-related cerebral hemodynamics. **arXiv preprint arXiv:2505.14717**, 2025. Disponível em: https://arxiv.org/abs/2505.14717.
6. NOHRA, M.; DUFOUR, S. Physics-informed neural networks for the numerical modeling of steady-state and transient electromagnetic problems with discontinuous media. **arXiv preprint arXiv:2406.04380**, 2024. Disponível em: https://arxiv.org/abs/2406.04380.
7. MIRANDA, G. C. de; LIMA, G. G. de; FARIAS, T. de S. An introduction to neural networks for physicists. **arXiv preprint arXiv:2505.13042**, 2025. Disponível em: https://arxiv.org/abs/2505.13042.

## Autoria

Nicolas Oliveira Rossoni — Instituto de Física de São Carlos / USP. Orientador: Prof. Dr. Celso Jorge Villas-Boas (UFSCar).
