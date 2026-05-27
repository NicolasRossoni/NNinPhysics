"""Mix2Funn layer used by the experiments in this repository.

A Mix2Funn neuron computes a softmax-weighted combination of a fixed
analytic basis ({sin, cos, exp+|.|, exp-|.|, sqrt, log, identity}) of an
affine pre-activation, with an optional quadratic pre-mixing of the inputs
and optional second-order cross-products of basis outputs. Stacking is
supported via `n_layers > 1`.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================
# Base functions (Q = 7)
# =========================================================================

class Sin(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


class Cos(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cos(x)


class ExpN(nn.Module):
    """exp(-0.01 * |x|)."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-0.01 * torch.abs(x))


class ExpP(nn.Module):
    """exp(0.01 * |x|)."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(0.01 * torch.abs(x))


class Sqrt(nn.Module):
    """sqrt(0.01 + ReLU(x)) — finite derivative at x=0."""
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(0.01 + self.relu(x))


class Log(nn.Module):
    """log(0.1 + ReLU(x)) — defined at x=0."""
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(0.1 + self.relu(x))


class Id(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


BASE_FUNCTIONS: list[nn.Module] = [Sin(), Cos(), ExpN(), ExpP(), Sqrt(), Log(), Id()]
BASE_FUNCTION_NAMES: list[str] = ["sin", "cos", "expN", "expP", "sqrt", "log", "id"]
Q: int = len(BASE_FUNCTIONS)


# =========================================================================
# Single Mix2Funn layer
# =========================================================================

class Mix2FunnLayer(nn.Module):
    """One Mix2Funn layer.

    Forward:
      1. Quadratic pre-activation (strict lower-triangular, no diagonal):
           s_i = b_i + W_i x + U_i vec(stril(x x^T))
         For N=1 inputs the quadratic block is empty (linear only); for
         N=2 it contributes one cross term x_1 x_2.
      2. Apply each base function f_i to its own s_i.
      3. Linear combination with softmax-normalised weights (or, if
         `use_softmax=False`, with free weights).

    Args:
        n_in:                  input dimension.
        n_out:                 output dimension.
        use_softmax:           if True, the combination weights come from
                               softmax(alpha / T) and T can be annealed.
        T_init:                initial softmax temperature.
        second_order_function: if True, the combination is enriched with
                               pairwise products f_i * f_j of basis outputs.
        dropout:               dropout probability applied on the features.
        init_alpha_std:        std of Gaussian init for the alpha logits.
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

        # Strict-lower-triangular indices for the quadratic pre-activation.
        self._n_quad = n_in * (n_in - 1) // 2
        self._n_feat = n_in + self._n_quad

        i_idx, j_idx = torch.tril_indices(n_in, n_in, offset=-1).unbind(0)
        self.register_buffer("tril_i", i_idx)
        self.register_buffer("tril_j", j_idx)

        # One Linear projection per base function.
        self.projections = nn.ModuleList(
            [nn.Linear(self._n_feat, n_out) for _ in range(Q)]
        )

        # Optional second-order cross-products of base outputs.
        self._n_prod = Q * (Q + 1) // 2 if second_order_function else 0
        self._n_total = Q + self._n_prod

        if second_order_function:
            ti, tj = torch.triu_indices(Q, Q, offset=0).unbind(0)
            self.register_buffer("prod_i", ti)
            self.register_buffer("prod_j", tj)

        # Combination weights.
        if use_softmax:
            if init_alpha_std > 0.0:
                self.alpha = nn.Parameter(torch.randn(n_out, self._n_total) * init_alpha_std)
            else:
                self.alpha = nn.Parameter(torch.zeros(n_out, self._n_total))
            self.register_buffer("T", torch.tensor(float(T_init)))
        else:
            self.w = nn.Parameter(torch.randn(n_out, self._n_total) * 0.1)

        # Xavier init on the projections.
        for lin in self.projections:
            nn.init.xavier_normal_(lin.weight)
            nn.init.zeros_(lin.bias)

    def _quadratic_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return [x, vec(stril(x x^T))] for a batch x of shape [B, N]."""
        xx = x.unsqueeze(-1) * x.unsqueeze(-2)
        xx_tril = xx[:, self.tril_i, self.tril_j]
        return torch.cat([x, xx_tril], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, n_in] -> [B, n_out]."""
        feat = self._quadratic_features(x)              # [B, n_feat]
        s = [proj(feat) for proj in self.projections]   # Q tensors [B, n_out]
        f = [BASE_FUNCTIONS[i](s[i]) for i in range(Q)] # Q tensors [B, n_out]
        f_stack = torch.stack(f, dim=-1)                # [B, n_out, Q]

        if self.second_order_function:
            f_outer = f_stack.unsqueeze(-1) * f_stack.unsqueeze(-2)
            f_prod = f_outer[..., self.prod_i, self.prod_j]
            features_full = torch.cat([f_stack, f_prod], dim=-1)
        else:
            features_full = f_stack

        if self.dropout is not None:
            features_full = self.dropout(features_full)

        if self.use_softmax:
            T = float(self.T)
            w = F.softmax(self.alpha / T, dim=-1)
        else:
            w = self.w

        a = (features_full * w.unsqueeze(0)).sum(dim=-1)
        return a

    def update_temperature(self, T: float) -> None:
        """Set the softmax temperature."""
        if self.use_softmax:
            self.T.fill_(float(T))


# =========================================================================
# Mix2Funn network (stackable)
# =========================================================================

class Mix2Funn(nn.Module):
    """Stackable Mix2Funn network.

    Args:
        n_in / n_out:          input / output dimensions of the network.
        n_layers:              number of stacked layers (>= 1).
        n_hidden:              intermediate width (used if n_layers > 1).
        use_softmax:           enables softmax-weighted combination.
        T_init / T_final:      range of the linear temperature annealing.
        n_anneal_epochs:       number of epochs over which T is annealed
                               (None = no annealing, T stays at T_init).
        second_order_function: enables f_i * f_j cross-products in every layer.
        dropout:               dropout probability inside each layer.
        init_alpha_std:        std for the alpha-logits init.
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
        assert n_layers >= 1, "n_layers must be >= 1"
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
        """Linear annealing of T from T_init to T_final over n_anneal_epochs.

        Applies the new T to every layer and returns it.
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
        """Magnitude pruning on the combination weights (alpha or w).

        Zeros the `ratio * total` smallest-magnitude weights layer by layer
        and registers a persistent mask + gradient hook so subsequent
        optimizer steps keep the pruned weights at zero. Returns the total
        number of zeroed weights.
        """
        if not (0.0 <= ratio <= 1.0):
            raise ValueError("ratio must be in [0, 1]")
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

            layer.register_buffer("_prune_mask", mask, persistent=True)
            def _make_hook(m):
                def hook(p):
                    if p.grad is not None:
                        p.grad.mul_(m)
                return hook
            target.register_post_accumulate_grad_hook(_make_hook(mask))
        return n_zeroed


# =========================================================================
# Utility
# =========================================================================

def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =========================================================================
# Quick self-test
# =========================================================================

if __name__ == "__main__":
    torch.manual_seed(0)
    print("Mix2Funn — quick self-test")
    print("-" * 60)

    net = Mix2Funn(n_in=1, n_out=1, n_layers=1)
    t = torch.linspace(0.0, math.pi, 5).unsqueeze(-1)
    out = net(t)
    print(f"Mix2Funn(1,1,1)  in={tuple(t.shape)} out={tuple(out.shape)}  "
          f"params={n_params(net)}")

    net2 = Mix2Funn(n_in=2, n_out=1, n_layers=1)
    xy = torch.randn(4, 2)
    out2 = net2(xy)
    print(f"Mix2Funn(2,1,1)  in={tuple(xy.shape)} out={tuple(out2.shape)}  "
          f"params={n_params(net2)}")

    net3 = Mix2Funn(n_in=1, n_out=1, n_layers=3, n_hidden=4)
    out3 = net3(t)
    print(f"Mix2Funn(1,1,3,h=4)  out={tuple(out3.shape)}  "
          f"params={n_params(net3)}")

    net4 = Mix2Funn(n_in=1, n_out=1, n_layers=1, n_anneal_epochs=100)
    T_start = net4.update_temperature_from_epoch(0)
    T_mid = net4.update_temperature_from_epoch(50)
    T_end = net4.update_temperature_from_epoch(100)
    print(f"Annealing  T(0)={T_start:.2f}  T(50)={T_mid:.2f}  T(100)={T_end:.2f}")

    n_z = net.prune_alpha(0.5)
    print(f"Pruning 50%  zeroed={n_z}/{net.layers[0].alpha.numel()}")

    print("-" * 60)
    print("OK")
