"""SimVP / temporal U-Net with DeepSphere Chebyshev convs.

History is stacked on the channel axis. Spatial mixing is ChebConv on the
HEALPix graph; resolution changes are nested 4-child pool / upsample
(nside hierarchy), not a planar U-Net.

``encode`` runs once; ``decode`` applies a centered lead embedding to the
translator features (identity at +20 min) and emits one residual map. That
is SimVP's T_out path with irregular operational leads instead of equal-dt
video frames.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from analysis.baselines.neural.hpx_conv import ChebConv, nest_pool, nest_upsample
from analysis.baselines.neural.lead import CenteredLead


class ChebAct(nn.Module):
    def __init__(self, nside: int, in_ch: int, out_ch: int, K: int = 3) -> None:
        super().__init__()
        self.conv = ChebConv(nside, in_ch, out_ch, K=K)
        self.norm = nn.LayerNorm(out_ch)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, nside: int, channels: int, K: int = 3) -> None:
        super().__init__()
        self.a = ChebAct(nside, channels, channels, K=K)
        self.b = ChebConv(nside, channels, channels, K=K)
        self.norm = nn.LayerNorm(channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.b(self.a(x))) + x)


class SimVPNet(nn.Module):
    """Encoder → residual translator → decoder on nested HEALPix."""

    def __init__(
        self,
        *,
        nside: int = 32,
        n_frames: int = 8,
        in_ch: int = 2,
        hid_s: int = 32,
        hid_t: int = 64,
        n_s: int = 2,
        n_t: int = 4,
        K: int = 3,
    ) -> None:
        super().__init__()
        if nside % (2 ** n_s) != 0:
            raise ValueError(f"nside={nside} not divisible by 2^{n_s}")
        self.nside = int(nside)
        self.n_frames = int(n_frames)
        self.in_ch = int(in_ch)
        self.hid_t = int(hid_t)
        stacked = self.n_frames * self.in_ch
        nsides: list[int] = [self.nside]
        for _ in range(n_s):
            nsides.append(nsides[-1] // 2)
        self._nsides = nsides
        self.enc = nn.ModuleList()
        ch = stacked
        for i, ns in enumerate(nsides[:-1]):
            out_ch = hid_s if i + 1 < n_s else hid_t
            self.enc.append(ChebAct(ns, ch, out_ch, K=K))
            ch = out_ch
        bot = nsides[-1]
        self.translator = nn.Sequential(
            *[ResidualBlock(bot, hid_t, K=K) for _ in range(n_t)]
        )
        self.lead = CenteredLead(hid_t)
        self.dec = nn.ModuleList()
        ch = hid_t
        for ns in reversed(nsides[:-1]):
            self.dec.append(ChebAct(ns, ch, hid_s, K=K))
            ch = hid_s
        self.head = ChebConv(self.nside, hid_s, 1, K=1)

    def encode(self, x: torch.Tensor) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """``x: (B, T, C, npix)`` → translator features and encoder skips."""
        b, t, c, npix = x.shape
        if t != self.n_frames or c != self.in_ch:
            raise ValueError(
                f"expected (B, {self.n_frames}, {self.in_ch}, npix), got {tuple(x.shape)}"
            )
        z = x.permute(0, 3, 1, 2).reshape(b, npix, t * c)
        skips: list[torch.Tensor] = []
        for layer in self.enc:
            z = layer(z)
            skips.append(z)
            z = nest_pool(z)
        z = self.translator(z)
        return {"z": z, "skips": skips}

    def decode(self, ctx: dict, lead_sec=None) -> torch.Tensor:
        """Lead-conditioned residual ``(B, npix)``. Identity at +20 min at init."""
        z = self.lead.add_to(ctx["z"], lead_sec)
        skips = ctx["skips"]
        for i, layer in enumerate(self.dec):
            z = nest_upsample(z)
            skip = skips[-(i + 1)]
            if z.shape[1] != skip.shape[1]:
                raise RuntimeError(
                    f"SimVP skip mismatch: {tuple(z.shape)} vs {tuple(skip.shape)}"
                )
            z = layer(z + skip)
        return self.head(z).squeeze(-1)

    def forward(self, x: torch.Tensor, lead_sec=None) -> torch.Tensor:
        """``x: (B, T, C, npix) → residual (B, npix)`` at ``lead_sec`` (default +20 min)."""
        return self.decode(self.encode(x), lead_sec)
