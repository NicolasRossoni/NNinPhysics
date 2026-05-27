"""Mix2Funn — template canonico reimplementado do zero (paper-fiel).

Fonte canonica: Farias, T. de S.; de Lima, G. G.; Maziero, J.; Villas-Boas, C. J.
"MixFunn: A Neural Network for Differential Equations with Improved Generalization
and Interpretability." arXiv, 2025. Equacoes (3-7).

Decisoes de implementacao 2026-05-12 (vide Claude/4-RAM.md):

- Pre-ativacao quadratica: tril ESTRITO (exclui diagonal x_i^2), alinhando com
  paper eq. 5 ("U has dimension 1 x N(N-1)/2") e com github dos autores.
  Para n_in=1: zero termos quadraticos (apenas linear). Para n_in=2: 1 termo
  cruzado (x_1 x_2). Decisao tomada em 2026-05-12 noite apos primeira rodada
  com tril inclusivo (offset=0) sofrer overflow em ExpP(0.01·|s|) para t in [0, 40]
  no oscilador, com s = t^2 = 1600 -> exp(16) ~ 9e6 desestabilizou Adam.
  Trocar para estrito remove esse termo no n_in=1 e estabiliza o treino.
  Contagens: Mix2Funn(1,1,1)=21 params, Mix2Funn(2,1,1)=35 params.

- second_order_function (produtos funcao-com-funcao): IMPLEMENTADO como flag
  opcional. Quando True, a camada computa outer product das Q saidas das funcoes
  e usa a triangular superior como features adicionais com seus proprios alphas.
  Necessario para representar solucoes que sao PRODUTOS (cos(t)*exp(-0.05t) no
  oscilador amortecido, paper §IV.A.3 expressao 9). Sem isso a rede converge para
  solucao trivial x(t)~0. NAO esta em paper §III mas esta no github dos autores e
  reproduz os 77 params da Tab. I. Documentado como extensao do github.
- Softmax com sinal POSITIVO (paper eq. 7). Github usa `softmax(-α/T)` no codigo.
- Annealing de T implementado (paper §III.C, schedule linear T_init->T_final).
- Pruning de magnitude implementado com MASCARA PERSISTENTE (paper §III.E:
  "mask effectively nullifies the parameters" — mascara registrada como buffer +
  re-aplicada apos cada optimizer.step para sobreviver a gradientes nao-zero).
- n_layers>1 (modo empilhado) e EXTENSAO PROPRIA do TCC, NAO esta no paper —
  cada Mix2FunnLayer reaplica o quadratico+funcoes na saida da anterior.
  O paper so descreve 1 camada Mix2Funn. Documentar como achado se usado.

Constantes pragmaticas adotadas do github (sem suporte explicito no paper):
- 0.01 nos exp para evitar overflow em dominios grandes.
- 0.01 no sqrt para derivada finita em x=0.
- 0.1 no log para evitar log(0).
Cada uma e documentada inline.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================
# Funcoes da base — Q = 7 (paper §III.B)
# =========================================================================

class Sin(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


class Cos(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cos(x)


class ExpN(nn.Module):
    """exp(-0.01 * |x|). Escala 0.01 do github — paper nao especifica."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-0.01 * torch.abs(x))


class ExpP(nn.Module):
    """exp(0.01 * |x|). Escala 0.01 do github — sem ela, overflow para |x| ~ 20."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(0.01 * torch.abs(x))


class Sqrt(nn.Module):
    """sqrt(0.01 + ReLU(x)). Constante 0.01 do github — evita derivada infinita em x=0."""
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(0.01 + self.relu(x))


class Log(nn.Module):
    """log(0.1 + ReLU(x)). Constante 0.1 do github — evita log(0)."""
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(0.1 + self.relu(x))


class Id(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


# Conjunto canonico Q=7 (ordem fixa para reprodutibilidade).
BASE_FUNCTIONS: list[nn.Module] = [Sin(), Cos(), ExpN(), ExpP(), Sqrt(), Log(), Id()]
BASE_FUNCTION_NAMES: list[str] = ["sin", "cos", "expN", "expP", "sqrt", "log", "id"]
Q: int = len(BASE_FUNCTIONS)


# =========================================================================
# Camada Mix2Funn — uma operacao (paper-fiel, eq. 5-7)
# =========================================================================

class Mix2FunnLayer(nn.Module):
    """Uma camada Mix2Funn (paper eq. 3 + 6 + 7).

    Forward:
      1. Pre-ativacao quadratica para cada uma das Q funcoes:
           s_i = b_i + W_i x + U_i vec(tril(x x^T))
         tril ESTRITO (offset=-1, exclui diagonal x_i^2): apenas cruzados
         x_i x_j com i > j. Alinha com paper eq. 5 ("U has dimension N(N-1)/2").
         Para N=1: zero termos quadraticos, so a parte linear de x.
         Para N=2: 1 termo cruzado (x_1 x_2).
      2. f_i(s_i) para cada funcao.
      3. Combinacao linear: a_k = sum_i w_{k,i} f_i(s_i).
         w pode vir de softmax(alpha / T) (paper eq. 7) ou de Parameter livre.

    Args:
        n_in: dimensao de entrada.
        n_out: dimensao de saida.
        use_softmax: se True, w vem de softmax(alpha/T) com T anelado.
                     Se False, w e Parameter livre (sem normalizacao).
        T_init: T inicial (so usado se use_softmax=True).
        second_order_function: se True, adiciona produtos f_i*f_j (i<=j, Q(Q+1)/2 termos)
                               como features adicionais. Necessario para representar
                               solucoes que sao PRODUTOS das funcoes da base. Extensao
                               do github dos autores (nao em paper §III). Default True.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        use_softmax: bool = True,
        T_init: float = 5.0,
        second_order_function: bool = False,
        dropout: float = 0.0,
        init_alpha_std: float = 0.0,
    ):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.use_softmax = use_softmax
        self.second_order_function = second_order_function
        self.dropout_p = dropout
        self.init_alpha_std = init_alpha_std
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else None

        # Pre-ativacao quadratica: tril ESTRITO (offset=-1) — exclui diagonal.
        # Para N=n_in, ha N + N(N-1)/2 features (linear + cruzados x_i x_j, i>j).
        # Paper eq. 5: "U has dimension 1 x N(N-1)/2" — alinhamos com isso.
        # Para N=1: zero termos quadraticos (so o linear t).
        # Para N=2: 1 termo cruzado (x_1 x_2).
        self._n_quad = n_in * (n_in - 1) // 2
        self._n_feat = n_in + self._n_quad

        i_idx, j_idx = torch.tril_indices(n_in, n_in, offset=-1).unbind(0)
        self.register_buffer("tril_i", i_idx)
        self.register_buffer("tril_j", j_idx)

        # Uma camada Linear por funcao: input n_feat -> output n_out.
        # Cada uma com seus proprios pesos (paper §III.B: "each function with its own preact").
        self.projections = nn.ModuleList(
            [nn.Linear(self._n_feat, n_out) for _ in range(Q)]
        )

        # Total de features finais: Q primeiros-order + (se on) Q*(Q+1)/2 produtos f_i*f_j.
        self._n_prod = Q * (Q + 1) // 2 if second_order_function else 0
        self._n_total = Q + self._n_prod

        # Indices da triangular superior INCLUSIVA para outer product das saidas funcoes.
        if second_order_function:
            ti, tj = torch.triu_indices(Q, Q, offset=0).unbind(0)
            self.register_buffer("prod_i", ti)
            self.register_buffer("prod_j", tj)

        # Pesos da combinacao final. Forma [n_out, n_total].
        if use_softmax:
            # alpha sao os logits aprendiveis; w = softmax(alpha / T).
            # init_alpha_std=0 (default): zeros (softmax inicial uniforme).
            # init_alpha_std>0: pequena perturbacao randomica para quebrar simetria.
            if init_alpha_std > 0.0:
                self.alpha = nn.Parameter(torch.randn(n_out, self._n_total) * init_alpha_std)
            else:
                self.alpha = nn.Parameter(torch.zeros(n_out, self._n_total))
            # T comeca em T_init; pode ser atualizado via update_temperature().
            self.register_buffer("T", torch.tensor(float(T_init)))
        else:
            # w livre.
            self.w = nn.Parameter(torch.randn(n_out, self._n_total) * 0.1)

        # Inicializacao Xavier nos Linear (alinha com PINN canonica).
        for lin in self.projections:
            nn.init.xavier_normal_(lin.weight)
            nn.init.zeros_(lin.bias)

    def _quadratic_features(self, x: torch.Tensor) -> torch.Tensor:
        """Concatena [x, vec(tril estrito(x x^T))] para um batch x de shape [B, N].

        Saida: [B, N + N(N-1)/2] = [B, n_feat]. tril ESTRITO (offset=-1).
        """
        # Produto externo batched: [B, N, 1] @ [B, 1, N] = [B, N, N].
        xx = x.unsqueeze(-1) * x.unsqueeze(-2)
        # Extrai triangular inferior estrita (exclui diagonal): [B, N(N-1)/2].
        xx_tril = xx[:, self.tril_i, self.tril_j]
        return torch.cat([x, xx_tril], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, n_in] -> [B, n_out]."""
        feat = self._quadratic_features(x)              # [B, n_feat]
        s = [proj(feat) for proj in self.projections]   # Q tensors [B, n_out]
        f = [BASE_FUNCTIONS[i](s[i]) for i in range(Q)] # Q tensors [B, n_out]
        f_stack = torch.stack(f, dim=-1)                # [B, n_out, Q]

        if self.second_order_function:
            # Outer product das saidas: [B, n_out, Q, Q].
            f_outer = f_stack.unsqueeze(-1) * f_stack.unsqueeze(-2)
            # Triangular superior inclusiva: [B, n_out, Q(Q+1)/2].
            f_prod = f_outer[..., self.prod_i, self.prod_j]
            # Concatena: [B, n_out, Q + Q(Q+1)/2] = [B, n_out, n_total].
            features_full = torch.cat([f_stack, f_prod], dim=-1)
        else:
            features_full = f_stack

        # Dropout opcional (paper §III.D) aplicado nas features antes da combinacao.
        # No-op se dropout=0.0; tambem no-op em eval mode (nn.Dropout cuida disso).
        if self.dropout is not None:
            features_full = self.dropout(features_full)

        if self.use_softmax:
            T = float(self.T)
            w = F.softmax(self.alpha / T, dim=-1)       # [n_out, n_total], soma=1 por linha
        else:
            w = self.w                                  # [n_out, n_total]

        # Combinacao linear: a_k = sum w_{k,m} feat_m.
        a = (features_full * w.unsqueeze(0)).sum(dim=-1)
        return a

    def update_temperature(self, T: float) -> None:
        """Atualiza T do softmax. So tem efeito se use_softmax=True."""
        if self.use_softmax:
            self.T.fill_(float(T))


# =========================================================================
# Rede Mix2Funn — suporta 1 camada (paper-puro) ou N camadas (deep)
# =========================================================================

class Mix2Funn(nn.Module):
    """Rede Mix2Funn empilhavel.

    n_layers=1: paper-puro (default).
    n_layers>1: extensao TCC — empilha camadas Mix2Funn em sequencia.

    Para n_layers>1 com n_in pequeno, a saida da primeira camada (n_out_intermediate)
    vira a entrada da proxima. As camadas intermediarias usam o mesmo n_out_intermediate
    para entrada e saida.

    Args:
        n_in: dimensao de entrada da rede.
        n_out: dimensao de saida final.
        n_layers: numero de camadas Mix2Funn empilhadas (>= 1).
        n_hidden: dimensao das camadas intermediarias (so usado se n_layers > 1).
        use_softmax: ativa softmax + annealing (eq. 7).
        T_init / T_final: range do annealing linear.
        n_anneal_epochs: numero de epocas sobre as quais T decai linearmente de T_init
                         para T_final. None = sem annealing (T fica em T_init).
    """

    def __init__(
        self,
        n_in: int,
        n_out: int = 1,
        n_layers: int = 1,
        n_hidden: int = 4,
        use_softmax: bool = True,
        T_init: float = 5.0,
        T_final: float = 0.1,
        n_anneal_epochs: int | None = None,
        second_order_function: bool = False,
        dropout: float = 0.0,
        init_alpha_std: float = 0.0,
    ):
        super().__init__()
        assert n_layers >= 1, "n_layers deve ser >= 1"
        self.n_in = n_in
        self.n_out = n_out
        self.n_layers = n_layers
        self.use_softmax = use_softmax
        self.T_init = float(T_init)
        self.T_final = float(T_final)
        self.n_anneal_epochs = n_anneal_epochs
        self.second_order_function = second_order_function
        self.dropout_p = dropout
        self.init_alpha_std = init_alpha_std

        sof = second_order_function
        def mk(a, b):
            return Mix2FunnLayer(a, b, use_softmax, T_init, sof, dropout, init_alpha_std)

        layers: list[Mix2FunnLayer] = []
        if n_layers == 1:
            layers.append(mk(n_in, n_out))
        else:
            layers.append(mk(n_in, n_hidden))
            for _ in range(n_layers - 2):
                layers.append(mk(n_hidden, n_hidden))
            layers.append(mk(n_hidden, n_out))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    # ------------------------------------------------------------------ T

    def update_temperature_from_epoch(self, epoch: int) -> float:
        """Annealing linear de T_init -> T_final ao longo de n_anneal_epochs.

        Aplica T em todas as camadas. Retorna o T atual.
        """
        if not self.use_softmax or self.n_anneal_epochs is None:
            return self.T_init
        frac = min(1.0, max(0.0, epoch / max(1, self.n_anneal_epochs)))
        T = self.T_init + frac * (self.T_final - self.T_init)
        for layer in self.layers:
            layer.update_temperature(T)
        return T

    # ----------------------------------------------------------- Pruning

    def prune_alpha(self, ratio: float) -> int:
        """Pruning de magnitude nos pesos `alpha` (logits do softmax) ou em `w`.

        Zera os `ratio * total` parametros com menor magnitude. Aplica-se camada
        a camada. Retorna o numero de pesos zerados.

        Implementacao com MASCARA PERSISTENTE: registra um buffer `_prune_mask` por
        camada e re-aplica a mascara via hook em `register_post_accumulate_grad_hook`
        do parametro, garantindo que gradientes nos pesos zerados permaneçam zerados
        (eq. III.E: "mask effectively nullifies the parameters").

        Apos chamar este metodo, qualquer optimizer.step() respeita o pruning porque
        os gradientes dos pesos zerados sao zerados antes da atualizacao.
        """
        if not (0.0 <= ratio <= 1.0):
            raise ValueError("ratio deve estar em [0, 1]")
        n_zeroed = 0
        for layer in self.layers:
            target = layer.alpha if layer.use_softmax else layer.w
            with torch.no_grad():
                flat = target.abs().flatten()
                if flat.numel() == 0:
                    continue
                k = int(ratio * flat.numel())
                if k == 0:
                    continue
                threshold = flat.kthvalue(k).values
                mask = (target.abs() > threshold).to(target.dtype)
                target.mul_(mask)
                n_zeroed += int((mask == 0).sum().item())

            # Registra mascara como buffer persistente.
            layer.register_buffer("_prune_mask", mask, persistent=True)
            # Hook no parametro para zerar gradientes nos pesos pruned.
            def _make_hook(m):
                def hook(p):
                    if p.grad is not None:
                        p.grad.mul_(m)
                return hook
            target.register_post_accumulate_grad_hook(_make_hook(mask))
        return n_zeroed


# =========================================================================
# Utilitario — contagem de parametros (sanity test)
# =========================================================================

def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =========================================================================
# Quick self-test (rodar `python mixfunn.py`)
# =========================================================================

if __name__ == "__main__":
    torch.manual_seed(0)
    print("Mix2Funn — quick self-test")
    print("-" * 60)

    # Forward para osc (n_in=1, n_out=1, n_layers=1, paper-puro).
    net = Mix2Funn(n_in=1, n_out=1, n_layers=1)
    t = torch.linspace(0.0, math.pi, 5).unsqueeze(-1)  # [5, 1]
    out = net(t)
    print(f"Mix2Funn(1,1,1)  in={tuple(t.shape)} out={tuple(out.shape)}  "
          f"params={n_params(net)}")

    # Forward para Burgers (n_in=2, n_out=1, n_layers=1).
    net2 = Mix2Funn(n_in=2, n_out=1, n_layers=1)
    xy = torch.randn(4, 2)
    out2 = net2(xy)
    print(f"Mix2Funn(2,1,1)  in={tuple(xy.shape)} out={tuple(out2.shape)}  "
          f"params={n_params(net2)}")

    # Forward para deep (n_in=1, 3 layers, hidden=4).
    net3 = Mix2Funn(n_in=1, n_out=1, n_layers=3, n_hidden=4)
    out3 = net3(t)
    print(f"Mix2Funn(1,1,3,h=4)  out={tuple(out3.shape)}  "
          f"params={n_params(net3)}")

    # Annealing.
    net4 = Mix2Funn(n_in=1, n_out=1, n_layers=1, n_anneal_epochs=100)
    T_start = net4.update_temperature_from_epoch(0)
    T_mid = net4.update_temperature_from_epoch(50)
    T_end = net4.update_temperature_from_epoch(100)
    print(f"Annealing  T(0)={T_start:.2f}  T(50)={T_mid:.2f}  T(100)={T_end:.2f}")

    # Pruning.
    n_z = net.prune_alpha(0.5)
    print(f"Pruning 50%  zeroed={n_z}/{net.layers[0].alpha.numel()}")

    print("-" * 60)
    print("OK")
