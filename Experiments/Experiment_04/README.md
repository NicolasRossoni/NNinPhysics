# Experimento 4 — Aneurisma intracraniano 3D (ANEUMO)

> Produz a **Tabela 5**, a **Figura 6** e a **Figura 7** de `monograph.pdf`.

## Problema

Navier–Stokes incompressível 3D estacionário em geometria anatômica real
(caso `AN4` do *dataset* ANEUMO, fator de vazão $m = 0{,}002$):

$$\begin{cases}
\rho\,(\mathbf{u} \cdot \nabla)\mathbf{u} + \nabla p - \mu\,\nabla^{2}\mathbf{u} = 0, \\
\nabla \cdot \mathbf{u} = 0, \\
\mathbf{u}|_{\text{parede}} = 0, \quad
\mathbf{u}|_{\text{entrada}} = \mathbf{u}_{\text{fis}}, \quad
p|_{\text{saída}} = 0.
\end{cases}$$

Nuvem de pontos com $\approx 129\,000$ amostras de um paciente real
($\rho = 1060$ kg/m³, $\mu = 3{,}5\times 10^{-3}$ Pa·s, $\mathrm{Re} \sim 230$).
Os campos de referência $(\mathbf{u}, p)$ vêm de simulação CFD comercial
disponibilizada no próprio *dataset*.

## Arquivos

- `1_preprocess.py` — envia a nuvem de pontos parseada (`case_AN4_m002.npz`) para
  o volume Modal em `tcc:/preprocess/exp_04/`.
- `2_train.py` — treina, em paralelo no Modal (T4), as **nove** configurações da
  Tabela 5 (variando PINN/MixFunn, profundidade, largura, *softmax* e taxa de
  supervisão entre 0%, 25%, 50% e 100%). Salva em `tcc:/checkpoints/exp_04/`.
- `3_analyze.py` — local; baixa os *checkpoints*, imprime a Tabela 5 e gera
  `aneur_panel_3d.png` (Figura 6) e `aneur_panel_plane.png` (Figura 7).

## Reprodução

O script `1_preprocess.py` espera o arquivo `case_AN4_m002.npz` em `/tmp/`;
ele é gerado a partir do *dataset* bruto ANEUMO (caso 4, m=0.002) — ver
<https://arxiv.org/abs/2505.14717>.

```bash
modal run 1_preprocess.py
modal run 2_train.py
modal volume get tcc /preprocess/exp_04/case_AN4_m002.npz ./tmp_checkpoints/
modal volume get tcc /checkpoints/exp_04 ./tmp_checkpoints
python 3_analyze.py
```


Saída: `aneur_panel_3d.png` e `aneur_panel_plane.png` no diretório do experimento.
