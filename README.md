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

Cada experimento traz seu próprio `README.md` (problema, arquivos, comando de reprodução, tempo e custo estimados).

## Pré-requisitos

Todos os treinos rodam em containers efêmeros com GPU NVIDIA T4 na **Modal** (<https://modal.com>); a máquina local só despacha os jobs e busca os artefatos. Nenhuma GPU local é necessária.

### Conta Modal (gratuita)

Cadastre-se em <https://modal.com/signup>. Em maio de 2026, o *tier* gratuito oferecia:

- **$5** em créditos para qualquer conta nova;
- **$25 adicionais** ao registrar um meio de pagamento (totalizando **$30**). Configurar um *spending cap* de $30 evita qualquer cobrança real e mantém todo o crédito disponível.

O orçamento total deste repositório (ver tabela abaixo) cabe folgadamente nesse limite.

### Ambiente local

```bash
python3 -m venv venv
source venv/bin/activate
pip install modal torch numpy scipy matplotlib
modal token new
modal volume create tcc
```

## Orçamento

| Experimento | Tempo de wall-time | Custo aproximado |
|---|---|---|
| Experiment_01 | ~30–45 min (31 jobs) | $3–4 |
| Experiment_02 | ~25 min     | $0,30 |
| Experiment_03 | ~25 min     | $0,80 |
| Experiment_04 | ~60–90 min  | $1,00 |
| Experiment_05 | ~20 min     | $0,35 |
| Experiment_06 | ~25 min     | $0,40 |
| Experiment_07 | ~30 min     | $1,40 |
| **Total**     | **~4 h**    | **~$7–8** |

Reproduzir o repositório inteiro cabe dentro do crédito gratuito da Modal.

## Padrão de uso

```bash
cd Experiments/Experiment_05
modal run 1_preprocess.py
modal run 2_train.py
modal volume get tcc /checkpoints/exp_05 ./tmp_checkpoints
python 3_analyze.py
```

Cada pasta indica seu próprio caminho no volume Modal e os arquivos de pós-processamento (quando existem).

## Créditos

Detalhes de implementação, validação e literatura estão descritos no monograph (`monograph.pdf`, seções de Introdução, Materiais e Métodos, e Referências). A bibliografia canônica usada pelo trabalho aparece ao final do PDF.

## Autoria

Nicolas Oliveira Rossoni — Instituto de Física de São Carlos / USP. Orientador: Prof. Dr. Celso Jorge Villas-Boas (UFSCar).
