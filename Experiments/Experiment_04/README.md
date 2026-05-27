# Experimento 4 — Aneurisma intracraniano 3D (ANEUMO)

> Produz a **Tabela 5**, a **Figura 6** e a **Figura 7** de `monograph.pdf`.

## Problema

Navier–Stokes incompressível 3D estacionário em uma geometria anatômica real (caso `AN4` do *dataset* ANEUMO, fator de vazão $m = 0{,}002$):

$$\left\{\begin{array}{l}
(\mathbf{u} \cdot \nabla)\mathbf{u} + \nabla p / \rho - \nu\,\nabla^2 \mathbf{u} = 0, \\
\nabla \cdot \mathbf{u} = 0, \\
\mathbf{u}|_{\text{parede}} = 0, \quad
\mathbf{u}|_{\text{entrada}} = \mathbf{u}_{\text{fis}}, \quad
p|_{\text{saída}} = 0.
\end{array}\right.$$

Nuvem de pontos com aproximadamente $129\,000$ amostras de um paciente real. Os campos de referência $(\mathbf{u}, p)$ vêm de uma simulação CFD comercial fornecida no próprio *dataset*.

## Arquivos

- `parse_aneumo.py` — baixa o *dataset* ANEUMO, extrai o caso `AN4` e sobe a nuvem de pontos para o volume Modal.
- `train_networks.py` — treina, em paralelo, as nove configurações da Tabela 5 (PINN e MixFunn em três regimes de supervisão: 0%, 25%, 50% e 100%).
- `run_aneur_pinn_unsup.py` — variante stand-alone do treino não-supervisionado da PINN $8\times128$.
- `plot_predictions.py` — gera a Figura 6 (painel 3D) e a Figura 7 (corte no plano $z \approx 39{,}75$ mm).
- `plot_aneumo.py` — visualizações do *dataset* bruto.
- `validate_aneumo.py` — verificação numérica das predições contra a referência.

## Reprodução

```bash
modal run parse_aneumo.py
modal run train_networks.py
modal volume get tcc /final/aneurisma ./results
python plot_predictions.py
```

Tempo: ~60–90 min de wall-time em T4 (containers paralelos). Custo: ~$1{,}00.

ANEUMO: <https://arxiv.org/abs/2505.14717>
