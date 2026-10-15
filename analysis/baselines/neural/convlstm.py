"""Peephole ConvLSTM whose gates are DeepSphere Chebyshev convs.

Shi et al. 2015 cell (input / forget / output / peephole), but every
``W * X`` is a ChebConv on the HEALPix neighbour graph instead of a
planar 3×3. Hidden state lives on the sphere ``(B, npix, hid)``.

The encoder unrolls over history once. ``decode`` applies a centered lead
embedding to the last hidden state (identity at +20 min) and emits one
residual. That is the encoding–forecasting split with irregular operational
leads instead of equal-cadence radar steps.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from analysis.baselines.neural.hpx_conv import ChebConv
from analysis.baselines.neural.lead import CenteredLead


class ConvLSTMCell(nn.Module):
    """One peephole ConvLSTM step on HEALPix tiles.

    ``i_t = σ(W_xi * X + W_hi * H + W_ci ∘ C + b_i)`` with ``*`` = ChebConv
    and ``∘`` = Hadamard peephole.
    """

    def __init__(self, nside: int, in_ch: int, hid: int, K: int = 3) -> None:
        super().__init__()
        self.hid = int(hid)
        self.x_conv = ChebConv(nside, in_ch, 4 * hid, K=K, bias=True)
        self.h_conv = ChebConv(nside, hid, 4 * hid, K=K, bias=False)
        self.w_ci = nn.Parameter(torch.zeros(1, 1, hid))
        self.w_cf = nn.Parameter(torch.zeros(1, 1, hid))
        self.w_co = nn.Parameter(torch.zeros(1, 1, hid))

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if state is None:
            h = torch.zeros(
                x.shape[0], x.shape[1], self.hid, device=x.device, dtype=x.dtype
            )
            c = torch.zeros_like(h)
        else:
            h, c = state
        i, f, g, o = (self.x_conv(x) + self.h_conv(h)).chunk(4, dim=-1)
        i = torch.sigmoid(i + self.w_ci * c)
        f = torch.sigmoid(f + self.w_cf * c)
        c = f * c + i * torch.tanh(g)
        o = torch.sigmoid(o + self.w_co * c)
        h = o * torch.tanh(c)
        return h, c


class ConvLSTMNet(nn.Module):
    """Stacked spherical ConvLSTM, then a ChebConv residual head."""

    def __init__(
        self,
        *,
        nside: int = 32,
        in_ch: int = 2,
        hidden: int = 57,
        n_layers: int = 2,
        K: int = 3,
    ) -> None:
        super().__init__()
        self.nside = int(nside)
        self.hidden = int(hidden)
        self.n_layers = int(n_layers)
        layers: list[ConvLSTMCell] = []
        ch = int(in_ch)
        for _ in range(self.n_layers):
            layers.append(ConvLSTMCell(self.nside, ch, self.hidden, K=K))
            ch = self.hidden
        self.layers = nn.ModuleList(layers)
        self.lead = CenteredLead(self.hidden)
        self.head = ChebConv(self.nside, self.hidden, 1, K=1)

    def encode(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Unroll history. ``x: (B, T, C, npix)`` → last hidden ``(B, npix, hid)``."""
        xt = x.permute(0, 1, 3, 2).contiguous()
        t = xt.shape[1]
        states: list[tuple[torch.Tensor, torch.Tensor] | None] = [
            None for _ in self.layers
        ]
        h: torch.Tensor | None = None
        for k in range(t):
            inp = xt[:, k]
            new_states: list[tuple[torch.Tensor, torch.Tensor]] = []
            for i, cell in enumerate(self.layers):
                h, cst = cell(inp, states[i])
                new_states.append((h, cst))
                inp = h
            states = new_states
        assert h is not None
        return {"h": h}

    def decode(self, ctx: dict, lead_sec=None) -> torch.Tensor:
        """Lead-conditioned residual ``(B, npix)``. Identity at +20 min at init."""
        h = self.lead.add_to(ctx["h"], lead_sec)
        return self.head(h).squeeze(-1)

    def forward(self, x: torch.Tensor, lead_sec=None) -> torch.Tensor:
        """``x: (B, T, C, npix) → residual (B, npix)`` at ``lead_sec`` (default +20 min)."""
        return self.decode(self.encode(x), lead_sec)
