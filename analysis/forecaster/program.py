"""Multimodal transmission forecaster.

One hour of DREAM history (24 frames) and the GOES scans available at the
forecast time are encoded once. Each lead is a change in natural zenith
optical depth, gated by the satellite sightline, then converted back to
transmission with the Beer-Lambert airmass factor. The training loss is
airmass-weighted yield MSE.

Inputs:
  sky, sky_mask: (B, T, npix) nested HEALPix, horizon-locked
  sat: (B, T, 18, H, W) GOES cube
  meta, wind_set: time features and derived-motion winds
Output:
  transmission (B, npix) at the requested lead. ``forward_horizons`` decodes
  90 s, 5 min, 20 min, and 30 min from one encoding.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from analysis.eval_harness.dream_data import DREAM_NSIDE
from analysis.healpix_torch import (
    apply_chebyshev_laplacian,
    chebyshev_basis,
    chebyshev_laplacian_csr,
    gaussian_smooth_healpix,
    neighbour_graph,
)
from analysis.forecaster.geom import (
    DECKS_KM,
    LEAD_SEC,
    WIND_H_IQR,
    WIND_H_MED,
    WIND_U_IQR,
    WIND_U_MED,
    WIND_V_IQR,
    WIND_V_MED,
    altaz_feat,
    decode_wind_uvh,
    decode_wind_uvh_decks,
    en_to_grid,
    local_pix_altaz,
    pierce_en,
    sample_sat_at,
    sat_coord_maps,
)
from analysis.our_model.mask_train import LABEL_SMOOTH_SIGMA, masked_airmass_yield_mse
from analysis.our_model.sat_pretrain.dataset import NATIVE_IR_HW

META_DIM = 9
DOY_SLICE = slice(2, 4)
WIND_FEAT = 6
ABLATION_NAMES = frozenset({"sat", "wind", "meta", "dream"})
_ABLATION_ALIASES = {
    "multimodal": frozenset({"sat", "wind"}),
    "goes": frozenset({"sat", "wind"}),
    "camera": frozenset({"sat", "wind", "meta"}),
    "camera-only": frozenset({"sat", "wind", "meta"}),
    "all": frozenset({"sat", "wind", "meta"}),
    "sat-only": frozenset({"dream"}),
    "goes-only": frozenset({"dream"}),
    "sky": frozenset({"dream"}),
}


def parse_ablate(spec: str | None) -> frozenset[str]:
    """``sat``, ``wind``, ``meta``, ``dream``, or aliases ``multimodal`` / ``camera`` / ``sat-only``."""
    if spec is None:
        return frozenset()
    raw = str(spec).strip().lower()
    if raw in {"", "none", "off"}:
        return frozenset()
    if raw in _ABLATION_ALIASES:
        return _ABLATION_ALIASES[raw]
    parts = {p.strip() for p in raw.replace("+", ",").split(",") if p.strip()}
    bad = sorted(parts - ABLATION_NAMES)
    if bad:
        raise ValueError(
            f"unknown --ablate {bad}; use sat, wind, meta, dream, "
            "multimodal, camera, sat-only"
        )
    return frozenset(parts)


# Transmission at or below this value counts as cloud for the false-clear rate.
FALSE_CLEAR_T = 0.63
DT_FLOOR_SEC = 30.0
HISTORY_SPAN_SEC = 3600.0
NATIVE_TIME_DILATIONS = (1, 2, 4, 8)


def frame_dt_sec(
    unix: torch.Tensor | None, t: int, *, like: torch.Tensor
) -> torch.Tensor:
    """Seconds between consecutive history frames. ``(B, T)``, floor 30 s.

    Frame 0 uses the window median Δt (or uniform 3600/(T-1) if ``unix`` is
    missing). Physical time, not frame index.
    """
    b = int(like.shape[0])
    fallback = HISTORY_SPAN_SEC / max(int(t) - 1, 1)
    if unix is None:
        return like.new_full((b, int(t)), fallback)
    u = unix.to(device=like.device, dtype=like.dtype)
    if u.shape[-1] != int(t):
        u = u[..., -int(t) :]
    dt = torch.empty(b, int(t), device=like.device, dtype=like.dtype)
    if int(t) <= 1:
        dt[:, 0] = fallback
        return dt
    step = (u[:, 1:] - u[:, :-1]).abs().clamp_min(DT_FLOOR_SEC)
    dt[:, 1:] = step
    dt[:, 0] = step.median(dim=1).values.clamp_min(DT_FLOOR_SEC)
    return dt


def frame_age_sec(
    unix: torch.Tensor | None, t: int, *, like: torch.Tensor
) -> torch.Tensor:
    """Seconds before the last history frame. ``(B, T)``, last column is 0."""
    if unix is None:
        dt = frame_dt_sec(None, t, like=like)
        steps = torch.arange(
            int(t) - 1, -1, -1, device=like.device, dtype=like.dtype
        )
        return steps.unsqueeze(0) * dt[:, :1]
    u = unix.to(device=like.device, dtype=like.dtype)
    if u.shape[-1] != int(t):
        u = u[..., -int(t) :]
    return (u[:, -1:] - u).clamp_min(0.0)


def last_unique_scan_pair(
    sat_sel: torch.Tensor,
    unix: torch.Tensor | None,
    sat_age_sec: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Last ABI scan, previous distinct scan, and seconds between scan clocks.

    Camera-adjacent T=24 slots often share one C13 file. Tendencies use the
    last two unique tensors and their ABI times, not the last two DREAM frames.
    If ``sat_age_sec`` is missing, Δt is the camera span between first
    appearances of those two unique ids.
    """
    b, t, _, _, _ = sat_sel.shape
    now = sat_sel[:, -1]
    dt_floor = sat_sel.new_full((b,), DT_FLOOR_SEC)
    if t < 2:
        return now, now, dt_floor

    delta = (sat_sel[:, 1:] - sat_sel[:, :-1]).abs().flatten(2).amax(dim=-1)
    is_new = torch.ones(b, t, dtype=torch.bool, device=sat_sel.device)
    is_new[:, 1:] = delta > 1e-6
    uid = is_new.to(dtype=torch.long).cumsum(dim=1) - 1
    last_id = uid[:, -1]
    prev_id = (last_id - 1).clamp(min=0)
    idx = torch.arange(t, device=sat_sel.device).view(1, t).expand(b, -1)
    prev_frame = torch.where(
        uid == prev_id.unsqueeze(1), idx, idx.new_full((), -1)
    ).amax(dim=1)
    first_last = torch.where(
        uid == last_id.unsqueeze(1), idx, idx.new_full((), t)
    ).amin(dim=1)
    first_prev = torch.where(
        uid == prev_id.unsqueeze(1), idx, idx.new_full((), t)
    ).amin(dim=1)
    batch_ix = torch.arange(b, device=sat_sel.device)
    prev = sat_sel[batch_ix, prev_frame.clamp(min=0)]
    one_scan = last_id <= 0
    prev = torch.where(one_scan.view(b, 1, 1, 1), now, prev)

    if unix is None:
        n_gap = (last_id.to(dtype=sat_sel.dtype)).clamp(min=1.0)
        dt = sat_sel.new_full((b,), HISTORY_SPAN_SEC) / n_gap
    else:
        u = unix.to(device=sat_sel.device, dtype=sat_sel.dtype)
        if u.shape[-1] != t:
            u = u[..., -t:]
        if sat_age_sec is not None:
            age = sat_age_sec.to(device=sat_sel.device, dtype=sat_sel.dtype)
            if age.shape[-1] != t:
                age = age[..., -t:]
            clock = u - age
            dt = (clock[:, -1] - clock[batch_ix, prev_frame.clamp(min=0)]).abs()
        else:
            dt = (u[batch_ix, first_last] - u[batch_ix, first_prev]).abs()
    dt = torch.where(one_scan, dt_floor, dt.clamp_min(DT_FLOOR_SEC))
    return now, prev, dt


def compute_loss(t_hat, t_true, valid, sin_alt, **kwargs):
    """Airmass yield MSE on tiles with a ≥ 30°. Extra kwargs are ignored."""
    del kwargs
    return masked_airmass_yield_mse(t_hat, t_true, valid, sin_alt)


SAT_CHANNEL_NAMES = (
    "c13",
    "c07",
    "c15",
    "btd_fog",
    "btd_split",
    "acm",
    "acha",
    "phase",
    "liquid",
    "ice",
    "lst",
    "tpw",
    "dc13",
    "dacm",
    "c08",
    "c09",
    "c10",
    "is_missing",
)
SAT_KEEP = SAT_CHANNEL_NAMES
SPHERE_SAT_NAMES = ("btd_fog", "btd_split", "c10", "acm", "liquid")
N_DECKS = len(DECKS_KM)
N_SPHERE = len(SPHERE_SAT_NAMES)
N_KIN = 2  # d(btd_fog), d(c10); dc13 is already in the cube
ACHA_OFF_M = 62.0
ACHA_SCALE_M = 1639.0
ACHA_MIN_KM = 1.0
DECK_TAU_KM = 1.5
PERSIST_SCALE_INIT = 1.0
HEAD_INIT_SCALE = 1e-3
ACM_SCALE = 0.25
# Soft residual gate on all tiles: sigmoid(+2)≈0.88 open so quiet sky can
# shrink toward persist without a closed prior that throttles incoming.
GATE_BIAS_INIT = 2.0
SAT_MODALITY_DROP = 0.05
WIND_MODALITY_DROP = 0.05
SAT_CNN_WIDTHS = (32, 64, 64)
SAT_CNN_KERNELS = (3, 5, 3)
SAT_CNN_DILATIONS = (1, 2, 4)
DEC_GRAPH_HIDDEN = 32
DEC_K = 2
TAU_CAP = 2.5
T_CLIP = (0.05, 0.999)
HEAD_MIN_ALT_DEG = 20.0
SKY_MEAN_INIT = 0.9905
SKY_STD_INIT = 0.0406


def resolve_sat_keep(names) -> tuple[int, ...]:
    """Stable indices into the loader cube. Never empty."""
    idx: list[int] = []
    seen: set[int] = set()
    for name in names:
        if name not in SAT_CHANNEL_NAMES:
            continue
        i = SAT_CHANNEL_NAMES.index(name)
        if i not in seen:
            seen.add(i)
            idx.append(i)
    if not idx:
        idx = [SAT_CHANNEL_NAMES.index("c13"), SAT_CHANNEL_NAMES.index("acha")]
    return tuple(idx)


def sat_to_14(sat: torch.Tensor) -> torch.Tensor:
    """Pad 2-ch / short cubes to 14. Keep C ≥ 14 so C08–C10 are not cropped."""
    if sat.ndim != 5:
        return sat
    b, t, c, h, w = sat.shape
    if c >= 14:
        return sat
    if c <= 2:
        d1 = sat.new_zeros(b, t, c, h, w)
        d2 = sat.new_zeros(b, t, c, h, w)
        if t > 1:
            d1[:, 1:] = sat[:, 1:] - sat[:, :-1]
        if t > 2:
            d2[:, 2:] = sat[:, 2:] - sat[:, :-2]
        sat = torch.cat([sat, d1, d2], dim=2)
        c = sat.shape[2]
    if c < 14:
        pad = sat.new_zeros(b, t, 14 - c, h, w)
        return torch.cat([sat, pad], dim=2)
    return sat


def select_sat(sat: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """``(B, T, C, H, W)`` → ``(B, T, len(idx), H, W)``. Pads short cubes first."""
    sat = sat_to_14(sat)
    if sat.ndim != 5:
        return sat
    return sat.index_select(2, idx.to(device=sat.device, dtype=torch.long))


class ChebConv(nn.Module):
    """Chebyshev spectral graph convolution of order ``K``."""

    def __init__(self, nside: int, in_ch: int, out_ch: int, K: int = 3) -> None:
        super().__init__()
        edge_index, edge_weight = neighbour_graph(nside)
        self.nside = int(nside)
        self.K = int(K)
        self.register_buffer("src", torch.as_tensor(edge_index[0]))
        self.register_buffer("dst", torch.as_tensor(edge_index[1]))
        self.register_buffer("w", torch.as_tensor(edge_weight, dtype=torch.float32))
        self.lin = nn.Linear(self.K * in_ch, out_ch)

    def _laplacian(self, x: torch.Tensor) -> torch.Tensor:
        lap = chebyshev_laplacian_csr(self.nside, device=x.device, dtype=x.dtype)
        return apply_chebyshev_laplacian(x, lap)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lap = chebyshev_laplacian_csr(
            self.nside, device=x.device, dtype=torch.float32
        )
        return self.lin(torch.cat(chebyshev_basis(x, lap, self.K), dim=-1))


class SphericalCNN(nn.Module):
    """Per-frame sky embedder with residual graph-conv blocks."""

    def __init__(
        self, nside: int, width: int = 12, K: int = 3, in_ch: int = 1
    ) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.in_proj = ChebConv(nside, in_ch + 1, width, K=1)
        self.conv1 = ChebConv(nside, width, width, K)
        self.conv2 = ChebConv(nside, width, width, K)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(0.1)
        self.width = width

    def forward(self, field: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if field.dim() == 3:
            field = field.unsqueeze(-1)
        b, t, n, c = field.shape
        x = torch.cat([field, mask.unsqueeze(-1)], dim=-1)
        x = x.reshape(b * t, n, c + 1)
        h = self.act(self.in_proj(x))
        h = h + self.act(self.norm1(self.conv1(h)))
        h = h + self.drop(self.act(self.norm2(self.conv2(h))))
        return h.reshape(b, t, n, -1)


def _group_norm(ch: int) -> nn.GroupNorm:
    for g in (8, 4, 2):
        if ch % g == 0:
            return nn.GroupNorm(g, ch)
    return nn.GroupNorm(1, ch)


def _same_pad(kernel: int, dilation: int) -> int:
    return int(dilation) * (int(kernel) - 1) // 2


class SatSpatialEncoder(nn.Module):
    """Stride-1 dilated CNN on native ABI (135×149). No downsample."""

    def __init__(
        self,
        in_ch: int,
        widths: tuple[int, ...] = SAT_CNN_WIDTHS,
        kernels: tuple[int, ...] = SAT_CNN_KERNELS,
        dilations: tuple[int, ...] = SAT_CNN_DILATIONS,
    ) -> None:
        super().__init__()
        if not (len(widths) == len(kernels) == len(dilations)):
            raise ValueError("widths, kernels, dilations must match")
        layers: list[nn.Module] = []
        prev = int(in_ch)
        for width, k, d in zip(widths, kernels, dilations, strict=True):
            layers.extend(
                [
                    nn.Conv2d(
                        prev,
                        int(width),
                        int(k),
                        stride=1,
                        padding=_same_pad(k, d),
                        dilation=int(d),
                    ),
                    _group_norm(int(width)),
                    nn.SiLU(),
                ]
            )
            prev = int(width)
        self.features = nn.Sequential(*layers)
        self.out_dim = prev

    def forward(self, sat: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = sat.shape
        z = self.features(sat.reshape(b * t, c, h, w))
        return z.reshape(b, t, z.shape[1], z.shape[2], z.shape[3])


class TimestampEmb(nn.Module):
    """Age from origin and Δt as log scales plus Fourier features.

    Not zero-init: the mixer sees clock time from step 0.
    """

    def __init__(self, d_model: int, n_freq: int = 4) -> None:
        super().__init__()
        self.n_freq = int(n_freq)
        in_dim = 4 + 4 * self.n_freq
        self.proj = nn.Linear(in_dim, int(d_model))
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(
        self, unix: torch.Tensor | None, *, like: torch.Tensor, t: int
    ) -> torch.Tensor:
        dt = frame_dt_sec(unix, t, like=like)
        age = frame_age_sec(unix, t, like=like)
        parts = [
            torch.log1p(age / 60.0),
            torch.log1p(dt / 60.0),
            (age / 3600.0).clamp(max=2.0),
            (dt / 300.0).clamp(max=8.0),
        ]
        age_hr = age / 3600.0
        dt_n = dt / 300.0
        for k in (1 << i for i in range(self.n_freq)):
            ang_a = (2.0 * math.pi * float(k)) * age_hr
            ang_d = (2.0 * math.pi * float(k)) * dt_n
            parts.extend(
                [torch.sin(ang_a), torch.cos(ang_a), torch.sin(ang_d), torch.cos(ang_d)]
            )
        return self.proj(torch.stack(parts, dim=-1))


class MaskedDilatedTemporalMix(nn.Module):
    """Multiscale masked conv over time at every HEALPix tile.

    Dilations 1,2,4,8 cover a T=24 hour at ~2.5 min cadence without treating
    frame index as physical time. Invalid frames are zeroed and the conv is
    density-normalized by the number of valid taps in the kernel.
    """

    def __init__(
        self,
        d_model: int,
        dilations: tuple[int, ...] = NATIVE_TIME_DILATIONS,
        kernel: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.kernel = int(kernel)
        self.dilations = tuple(int(d) for d in dilations)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    int(d_model),
                    int(d_model),
                    self.kernel,
                    padding=d,
                    dilation=d,
                )
                for d in self.dilations
            ]
        )
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(int(d_model))

    def _valid_mask(
        self,
        valid: torch.Tensor | None,
        b: int,
        t: int,
        n: int,
        like: torch.Tensor,
    ) -> torch.Tensor:
        if valid is None:
            return like.new_ones(b * n, 1, t)
        m = valid.to(device=like.device, dtype=like.dtype)
        if m.dim() == 2:
            m = m.unsqueeze(-1).expand(b, t, n)
        return m.permute(0, 2, 1).reshape(b * n, 1, t)

    def mix(
        self, seq: torch.Tensor, valid: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Residual dilated mix. ``(B, T, N, C)`` → mixed ``(B, T, N, C)``."""
        b, t, n, c = seq.shape
        x = seq.permute(0, 2, 3, 1).reshape(b * n, c, t)
        m = self._valid_mask(valid, b, t, n, x)
        x = x * m
        ones = m.new_ones(1, 1, self.kernel)
        for conv, d in zip(self.convs, self.dilations, strict=True):
            y = conv(x)
            count = nn.functional.conv1d(m, ones, padding=d, dilation=d)
            y = y / count.clamp_min(1.0)
            x = (x + self.drop(self.act(y))) * m
        mixed = x.reshape(b, n, c, t).permute(0, 3, 1, 2)
        return self.norm(mixed)

    def forward(
        self, seq: torch.Tensor, valid: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.mix(seq, valid)[:, -1]


class MetadataMLP(nn.Module):
    def __init__(
        self, in_dim: int = META_DIM, out_dim: int = 16, dropout: float = 0.3
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LeadQueryMLP(nn.Module):
    """Small lead embedding. Last layer zero → identity at init for every lead."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        hidden = int(d_model)
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, int(d_model)),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)


class WindSetEncoder(nn.Module):
    """Cross-attention + pooled statistics over a padded DMW point cloud."""

    def __init__(
        self,
        in_dim: int = WIND_FEAT,
        hidden: int = 32,
        out_dim: int = 24,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.query = nn.Parameter(torch.zeros(1, 1, hidden))
        nn.init.normal_(self.query, std=0.02)
        self.attn = nn.MultiheadAttention(
            hidden, n_heads, dropout=dropout, batch_first=True
        )
        self.out = nn.Linear(hidden * 3, out_dim)
        self.out_dim = out_dim

    def forward(self, wind_set: torch.Tensor, wind_mask: torch.Tensor) -> torch.Tensor:
        b, t, n, _ = wind_set.shape
        x = self.proj(wind_set.reshape(b * t, n, -1))
        q = self.query.expand(b * t, -1, -1)
        pad = wind_mask.reshape(b * t, n) <= 0
        all_pad = pad.all(dim=1)
        if bool(all_pad.any()):
            pad = pad.clone()
            pad[all_pad, 0] = False
        z, _ = self.attn(q, x, x, key_padding_mask=pad, need_weights=False)
        z = z.squeeze(1)
        valid = (~pad).unsqueeze(-1).float()
        mean_pool = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        max_pool = x.masked_fill(pad.unsqueeze(-1), -float("inf")).max(dim=1).values
        z = torch.cat([z, mean_pool, max_pool], dim=-1)
        if bool(all_pad.any()):
            z = z.masked_fill(all_pad.unsqueeze(-1), 0)
        return self.out(z).reshape(b, t, -1)


class SphericalDecoder(nn.Module):
    """Per-tile linear skip + short Cheb residual → Δτ₀."""

    def __init__(
        self,
        nside: int,
        skip_dim: int,
        width: int = DEC_GRAPH_HIDDEN,
        raw_dim: int = 0,
        k: int = DEC_K,
    ) -> None:
        super().__init__()
        self.raw_dim = raw_dim
        in_ch = skip_dim + raw_dim
        self.skip = nn.Linear(in_ch, 1)
        self.down = nn.Linear(in_ch, width)
        self.mix = ChebConv(nside, width, width, K=k)
        self.head = ChebConv(nside, width, 1, K=k)
        self.act = nn.SiLU()
        with torch.no_grad():
            self.skip.weight.mul_(HEAD_INIT_SCALE)
            self.skip.bias.zero_()
            self.head.lin.weight.mul_(HEAD_INIT_SCALE)
            self.head.lin.bias.zero_()

    def forward(
        self,
        skip: torch.Tensor,
        raw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if raw is None:
            raw = skip.new_zeros(skip.shape[0], skip.shape[1], self.raw_dim)
        x = torch.cat([skip, raw], dim=-1)
        residual = self.head(self.act(self.mix(self.act(self.down(x)))))
        return self.skip(x).squeeze(-1) + residual.squeeze(-1)


class ResidualGate(nn.Module):
    """Per-tile soft gate from pierce GOES (now + upwind) → [0, 1].

    Multiplies the full dτ₀ residual on every tile. Bias init is open
    (~0.88) so the gate can shut quiet-sky jitter toward persist without
    a closed prior that blocks incoming cloud.
    """

    def __init__(self, in_dim: int, bias_init: float = GATE_BIAS_INIT) -> None:
        super().__init__()
        self.proj = nn.Linear(int(in_dim), 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.constant_(self.proj.bias, float(bias_init))

    def forward(self, sat_feats: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.proj(sat_feats)).squeeze(-1)


class LeadAttnPool(nn.Module):
    """Per-tile attention pool of history with a lead-conditioned query."""

    def __init__(self, d_model: int, n_heads: int = 4) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            int(d_model), int(n_heads), dropout=0.0, batch_first=True
        )
        self.norm = nn.LayerNorm(int(d_model))

    def forward(
        self,
        hist: torch.Tensor,
        query: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, t, n, c = hist.shape
        h = hist.permute(0, 2, 1, 3).reshape(b * n, t, c)
        q = query.reshape(b * n, 1, c)
        kpm = None
        if key_padding_mask is not None:
            keep = key_padding_mask.to(device=hist.device)
            if keep.dim() == 2:
                keep = keep.unsqueeze(-1).expand(b, t, n)
            # MHA: True = ignore. Always keep the last frame so a row is never empty.
            kpm = (~keep.bool()).permute(0, 2, 1).reshape(b * n, t)
            kpm[:, -1] = False
        z, _ = self.attn(q, h, h, key_padding_mask=kpm, need_weights=False)
        return self.norm(z).reshape(b, n, c)


class SkyMapForecaster(nn.Module):
    """T=24 camera + unique-scan GOES → +20 min T̂. Persist residual in τ₀."""

    def __init__(
        self,
        *,
        nside: int = DREAM_NSIDE,
        n_frames: int = 24,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        sky_width: int = 16,
        sat_width: int = 32,
        sat_dim: int = 32,
        meta_dim: int = 16,
        wind_dim: int = 24,
        dropout: float = 0.3,
        modality_drop: float = 0.3,
        frame_drop: float = 0.2,
        unique_scan_sat: bool = True,
        wind_age_mean: float = 0.0,
        wind_age_std: float = 1.0,
        wind_u_med: float = WIND_U_MED,
        wind_u_iqr: float = WIND_U_IQR,
        wind_v_med: float = WIND_V_MED,
        wind_v_iqr: float = WIND_V_IQR,
        wind_h_med: float = WIND_H_MED,
        wind_h_iqr: float = WIND_H_IQR,
    ) -> None:
        super().__init__()
        n_attn_heads = max(1, int(n_heads))
        del n_heads, sat_dim, n_layers, unique_scan_sat
        self.nside = nside
        self.npix = 12 * nside * nside
        self.n_frames = n_frames
        self.d_model = d_model
        self.lead_ref_sec = float(LEAD_SEC)
        self.register_buffer("wind_age_mean", torch.tensor(float(wind_age_mean)))
        self.register_buffer("wind_age_std", torch.tensor(float(wind_age_std)))
        self.register_buffer("wind_u_med", torch.tensor(float(wind_u_med)))
        self.register_buffer("wind_u_iqr", torch.tensor(float(wind_u_iqr)))
        self.register_buffer("wind_v_med", torch.tensor(float(wind_v_med)))
        self.register_buffer("wind_v_iqr", torch.tensor(float(wind_v_iqr)))
        self.register_buffer("wind_h_med", torch.tensor(float(wind_h_med)))
        self.register_buffer("wind_h_iqr", torch.tensor(float(wind_h_iqr)))
        alt, az = local_pix_altaz(nside)
        self.register_buffer("pix_alt", torch.from_numpy(np.asarray(alt)))
        self.register_buffer("pix_az", torch.from_numpy(np.asarray(az)))
        self.register_buffer("sky_mean", torch.tensor(float(SKY_MEAN_INIT)))
        self.register_buffer("sky_std", torch.tensor(float(SKY_STD_INIT)))
        self.horizon_lock = True
        self.persist_in_residual = True
        self.output_transmission = True
        self.false_clear_t = float(FALSE_CLEAR_T)
        self.persist_scale = nn.Parameter(torch.tensor(float(PERSIST_SCALE_INIT)))
        self.modality_drop = min(float(modality_drop), 0.2)
        self.frame_drop = min(float(frame_drop), 0.1)
        dropout = min(float(dropout), 0.3)

        sat_idx = resolve_sat_keep(SAT_KEEP)
        self.register_buffer("sat_idx", torch.as_tensor(sat_idx, dtype=torch.long))
        sphere_pos = [
            sat_idx.index(SAT_CHANNEL_NAMES.index(nm))
            for nm in SPHERE_SAT_NAMES
            if SAT_CHANNEL_NAMES.index(nm) in sat_idx
        ]
        if not sphere_pos:
            sphere_pos = list(range(min(N_SPHERE, len(sat_idx))))
        self.register_buffer(
            "sphere_pos", torch.as_tensor(sphere_pos, dtype=torch.long)
        )
        self.n_sphere = len(sphere_pos)
        self.acm_sphere_idx = (
            list(SPHERE_SAT_NAMES).index("acm")
            if "acm" in SPHERE_SAT_NAMES
            else -1
        )
        acha_i = SAT_CHANNEL_NAMES.index("acha")
        acha_pos = sat_idx.index(acha_i) if acha_i in sat_idx else -1
        self.register_buffer(
            "acha_pos", torch.as_tensor(acha_pos, dtype=torch.long)
        )
        self.dacm_idx = SAT_CHANNEL_NAMES.index("dacm")
        self.fog_idx = SAT_CHANNEL_NAMES.index("btd_fog")
        self.c10_idx = SAT_CHANNEL_NAMES.index("c10")
        missing_i = SAT_CHANNEL_NAMES.index("is_missing")
        self.missing_pos = sat_idx.index(missing_i) if missing_i in sat_idx else -1
        self.deck_km = nn.Parameter(
            torch.tensor(DECKS_KM, dtype=torch.float32), requires_grad=False
        )

        self.sky_enc = SphericalCNN(nside, width=sky_width, in_ch=7)
        widths = (int(sat_width), int(sat_width) * 2, int(sat_width) * 2)
        self.sat_enc = SatSpatialEncoder(
            in_ch=len(sat_idx) + 2 + N_KIN, widths=widths
        )
        self.pix_fuse = nn.Linear(sky_width + self.sat_enc.out_dim, d_model)
        self.meta_enc = MetadataMLP(META_DIM, meta_dim, dropout=dropout)
        self.wind_enc = WindSetEncoder(
            in_dim=WIND_FEAT, hidden=32, out_dim=wind_dim, n_heads=4, dropout=dropout
        )

        fused = d_model + wind_dim + meta_dim
        self.fuse = nn.Linear(fused, d_model)
        self.fuse_norm = nn.LayerNorm(d_model)
        self.token_drop = nn.Dropout(dropout)
        self.query = nn.Parameter(torch.zeros(1, 1, 1, d_model))
        nn.init.normal_(self.query, std=0.02)
        self.time_emb = TimestampEmb(d_model)
        self.encoder = MaskedDilatedTemporalMix(d_model, dropout=dropout)
        # skip = last + tanh(g(last)) · context. Zero-init → last-frame skip.
        self.hist_ctx_gate = nn.Linear(d_model, 1)
        nn.init.zeros_(self.hist_ctx_gate.weight)
        nn.init.zeros_(self.hist_ctx_gate.bias)
        # Centered lead adapter: e(h)=MLP(φ(h))−MLP(φ(20 min)). Last layer
        # zero → identity at init; +20 min stays the origin after training.
        self.lead_query = LeadQueryMLP(d_model)
        # last T, ΔT, now + upwind pierce, last-frame CNN upwind
        self.decoder = SphericalDecoder(
            nside,
            skip_dim=d_model,
            width=DEC_GRAPH_HIDDEN,
            raw_dim=2 + 2 * self.n_sphere + self.sat_enc.out_dim,
            k=DEC_K,
        )
        # Gate dτ₀ from sat_now∥sat_up (sphere channels); bias open (~0.88).
        self.residual_gate = ResidualGate(2 * self.n_sphere)
        self.lead_pool = LeadAttnPool(d_model, n_heads=n_attn_heads)
        self.last_sin_alt: torch.Tensor | None = None
        self.last_T: torch.Tensor | None = None
        self.last_gate: torch.Tensor | None = None
        self.last_skip: torch.Tensor | None = None
        self.ablate: frozenset[str] = frozenset()

    def set_ablate(self, names: frozenset[str] | set[str] | str | None) -> None:
        if isinstance(names, (set, frozenset)):
            self.ablate = frozenset(names)
        else:
            self.ablate = parse_ablate(names)

    def extra_repr(self) -> str:
        extra = f" ablate={'+'.join(sorted(self.ablate))}" if self.ablate else ""
        return (
            f"n_pix={self.npix} persist_scale={float(self.persist_scale):.3g} "
            f"T=24 last+gated_hist unique-scan GOES residual gate "
            f"deck-upwind Eulerian-persist centered-lead-query{extra}"
        )

    def _deck_weights(self, acha_km: torch.Tensor) -> torch.Tensor:
        decks = self.deck_km.to(device=acha_km.device, dtype=acha_km.dtype)
        valid = acha_km >= ACHA_MIN_KM
        logits = -(acha_km - decks).abs() / DECK_TAU_KM
        logits = logits.masked_fill(~valid, -1.0e4)
        none = ~valid.any(dim=-1)
        fallback = torch.zeros_like(logits)
        fallback[..., 0] = 1.0
        w = torch.softmax(logits, dim=-1)
        return torch.where(none.unsqueeze(-1), fallback, w)

    def _pierce_grids(
        self, alt: torch.Tensor, az: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        grids: list[torch.Tensor] = []
        bads: list[torch.Tensor] = []
        for hkm in DECKS_KM:
            east, north = pierce_en(alt, az, float(hkm))
            bad = ~torch.isfinite(east) | ~torch.isfinite(north)
            grid = en_to_grid(east.nan_to_num(0.0), north.nan_to_num(0.0))
            grid = torch.where(bad.unsqueeze(-1), torch.full_like(grid, 2.0), grid)
            grids.append(grid)
            bads.append(bad)
        return grids, bads

    def _acha_deck_weights(
        self,
        acha_map: torch.Tensor | None,
        grids: list[torch.Tensor],
        bads: list[torch.Tensor],
        b: int,
        n_pix: int,
        ref: torch.Tensor,
    ) -> torch.Tensor:
        if acha_map is None:
            return ref.new_full((b, n_pix, N_DECKS), 1.0 / N_DECKS)
        acha_ch: list[torch.Tensor] = []
        for grid, bad in zip(grids, bads, strict=True):
            acha_s = sample_sat_at(acha_map, grid).squeeze(1)
            acha_s = torch.where(bad, torch.zeros_like(acha_s), acha_s)
            acha_ch.append((acha_s * ACHA_SCALE_M + ACHA_OFF_M) / 1000.0)
        return self._deck_weights(torch.stack(acha_ch, dim=-1))

    def _sample_mixed(
        self,
        maps: torch.Tensor,
        grids: list[torch.Tensor],
        bads: list[torch.Tensor],
        w: torch.Tensor,
    ) -> torch.Tensor:
        """``maps`` ``(B, C, H, W)`` → ACHA-mixed pierce ``(B, N, C)``."""
        chunks = [sample_sat_at(maps, grid).transpose(1, 2) for grid in grids]
        stacked = torch.stack(chunks, dim=2)
        return (stacked * w.unsqueeze(-1)).sum(dim=2)

    def _encode_pierce_cnn_t(
        self,
        sat_in: torch.Tensor,
        grids: list[torch.Tensor],
        bads: list[torch.Tensor],
        deck_w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sat CNN + pierce over DREAM time → ``(B,T,N,D)`` and last-frame ``z``.

        Pack each sample's distinct consecutive GOES scans, encode once,
        then gather pierce tokens back onto DREAM time. GroupNorm is
        sample-local, so packing does not change its result.
        """
        b, t, _, _, _ = sat_in.shape
        delta = (sat_in[:, 1:] - sat_in[:, :-1]).abs().flatten(2).amax(dim=-1)
        is_new = torch.ones(b, t, dtype=torch.bool, device=sat_in.device)
        is_new[:, 1:] = delta > 1e-6
        frames = [sat_in[bi, is_new[bi]] for bi in range(b)]
        counts = [int(x.shape[0]) for x in frames]
        owner = torch.repeat_interleave(
            torch.arange(b, device=sat_in.device),
            torch.as_tensor(counts, device=sat_in.device),
        )
        z_unique = self.sat_enc.features(torch.cat(frames, dim=0))
        p_unique = self._sample_mixed(
            z_unique,
            [grid[owner] for grid in grids],
            [bad[owner] for bad in bads],
            deck_w[owner],
        )
        p_time = p_unique.new_empty(b, t, p_unique.shape[1], p_unique.shape[2])
        z_last = []
        offset = 0
        for bi, count in enumerate(counts):
            gather = is_new[bi].long().cumsum(0) - 1
            p_time[bi] = p_unique[offset : offset + count][gather]
            z_last.append(z_unique[offset + count - 1])
            offset += count
        return p_time, torch.stack(z_last)

    def _scale_acm(self, pierce: torch.Tensor) -> torch.Tensor:
        if self.acm_sphere_idx < 0:
            return pierce
        out = pierce.clone()
        out[..., self.acm_sphere_idx] = out[..., self.acm_sphere_idx] * ACM_SCALE
        return out

    def _wind_decode_kwargs(self) -> dict:
        return {
            "u_med": self.wind_u_med,
            "u_iqr": self.wind_u_iqr,
            "v_med": self.wind_v_med,
            "v_iqr": self.wind_v_iqr,
            "h_med": self.wind_h_med,
            "h_iqr": self.wind_h_iqr,
        }

    def _decode_last_t(
        self, last: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        t = last * self.sky_std + self.sky_mean
        t = t.clamp(1e-4, 1.0)
        if float(LABEL_SMOOTH_SIGMA) > 0.0:
            t = gaussian_smooth_healpix(t, valid > 0, sigma=float(LABEL_SMOOTH_SIGMA))
        return t

    def _dummy_sat(self, ref: torch.Tensor, batch: int, n_frames: int) -> torch.Tensor:
        h, w = NATIVE_IR_HW
        return ref.new_zeros(batch, n_frames, len(SAT_CHANNEL_NAMES), h, w)

    def _zero_ablated(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if "sat" in self.ablate and "sky" in batch:
            sky = batch["sky"]
            batch["sat"] = self._dummy_sat(sky, int(sky.shape[0]), int(sky.shape[1]))
        if "wind" in self.ablate:
            for k in ("wind_set", "wind_mask", "wind_set_q", "wind_mask_q"):
                if k in batch:
                    batch[k] = torch.zeros_like(batch[k])
        if "meta" in self.ablate:
            for k in ("meta", "meta_q"):
                if k in batch:
                    batch[k] = torch.zeros_like(batch[k])
        if "dream" in self.ablate:
            for k in ("sky", "sky_mask"):
                if k in batch:
                    batch[k] = torch.zeros_like(batch[k])
        return batch

    def preprocess_batch(
        self, batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Zero ``dacm``. Do not append fog/liquid/split trend maps."""
        sat = batch.get("sat")
        if sat is not None and sat.ndim == 5 and sat.shape[2] > self.dacm_idx:
            sat = sat.clone()
            sat[:, :, self.dacm_idx] = 0
            batch["sat"] = sat
        return self._zero_ablated(batch)

    def _as_lead(self, lead_sec, *, batch: int, ref: torch.Tensor) -> torch.Tensor:
        """Per-sample lead in seconds. ``None`` → paper ``LEAD_SEC``."""
        if lead_sec is None:
            return ref.new_full((batch,), float(LEAD_SEC))
        lead = torch.as_tensor(lead_sec, device=ref.device, dtype=ref.dtype).reshape(-1)
        if int(lead.numel()) == 1:
            lead = lead.expand(batch)
        if int(lead.numel()) != int(batch):
            raise ValueError(f"lead_sec n={int(lead.numel())} != batch {batch}")
        return lead.clamp(min=1.0)

    def _lead_feat(self, lead: torch.Tensor) -> torch.Tensor:
        """``(log(h/ref), h/ref)``. ``lead`` is seconds, shape ``(B,)``."""
        ref = float(self.lead_ref_sec)
        ratio = (lead / ref).clamp(min=1e-3, max=8.0)
        return torch.stack([ratio.log(), ratio], dim=-1)

    def _lead_query_delta(self, lead: torch.Tensor) -> torch.Tensor:
        """``e(h)=MLP(φ(h))−MLP(φ(1200))``. Zero at +20 min after the MLP learns."""
        feat = self._lead_feat(lead)
        ref = self._lead_feat(lead.new_full(lead.shape, float(self.lead_ref_sec)))
        return self.lead_query(feat) - self.lead_query(ref)

    def _last_plus_hist_context(
        self,
        hist: torch.Tensor,
        query: torch.Tensor,
        frame_valid: torch.Tensor | None,
    ) -> torch.Tensor:
        """Last-frame skip plus gated context from older frames.

        ``skip = last + tanh(g(last)) · context``. ``g`` is zero at init, so
        extra history cannot change the skip until the gate opens.
        """
        last = hist[:, -1] + query.squeeze(1)
        t = int(hist.shape[1])
        if t <= 1 or self.hist_ctx_gate is None:
            return last
        mixed = self.encoder.mix(hist, frame_valid)
        older = mixed[:, :-1]
        older_valid = None if frame_valid is None else frame_valid[:, :-1]
        ctx = self.lead_pool(older, query, key_padding_mask=older_valid)
        gate = torch.tanh(self.hist_ctx_gate(last))
        return last + gate * ctx

    def _upwind_grids(
        self,
        alt: torch.Tensor,
        az: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        lead: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        grids_up: list[torch.Tensor] = []
        bads_up: list[torch.Tensor] = []
        lead_km = lead / 1000.0
        for i, hkm in enumerate(DECKS_KM):
            if u.dim() == 2:
                ui, vi = u[:, i], v[:, i]
            else:
                ui, vi = u, v
            dx = (ui * lead_km).unsqueeze(-1)
            dy = (vi * lead_km).unsqueeze(-1)
            east, north = pierce_en(alt, az, float(hkm))
            bad = ~torch.isfinite(east) | ~torch.isfinite(north)
            grid_up = en_to_grid(
                (east - dx).nan_to_num(0.0), (north - dy).nan_to_num(0.0)
            )
            grid_up = torch.where(
                bad.unsqueeze(-1), torch.full_like(grid_up, 2.0), grid_up
            )
            grids_up.append(grid_up)
            bads_up.append(bad)
        return grids_up, bads_up

    def _decode_lead(self, ctx: dict, lead_sec) -> torch.Tensor:
        """Upwind GOES pierce + Eulerian last T, then gated residual in τ₀."""
        b = int(ctx["b"])
        lead = self._as_lead(lead_sec, batch=b, ref=ctx["hist"])
        alt, az = ctx["alt"], ctx["az"]
        u_deck, v_deck = ctx["u_deck"], ctx["v_deck"]
        grids_up, bads_up = self._upwind_grids(alt, az, u_deck, v_deck, lead)
        deck_w = ctx["deck_w"]
        cnn_up = self._sample_mixed(ctx["z_last"], grids_up, bads_up, deck_w)
        sat_up = self._scale_acm(
            self._sample_mixed(ctx["key"], grids_up, bads_up, deck_w)
        )
        sat_now = ctx["sat_now"]
        drop_sat = ctx.get("drop_sat")
        if drop_sat is not None:
            cnn_up = cnn_up.masked_fill(drop_sat.view(b, 1, 1), 0)
            keep_s = (~drop_sat).to(dtype=sat_now.dtype).view(b, 1, 1)
            sat_up = sat_up * keep_s
        t_last = ctx["t_last"]
        raw_last = torch.cat(
            [
                torch.stack([t_last, ctx["sky_diff_last"]], dim=-1),
                sat_now,
                sat_up,
                cnn_up,
            ],
            dim=-1,
        )
        query = ctx["query"] + self._lead_query_delta(lead).view(b, 1, 1, -1)
        skip = self._last_plus_hist_context(
            ctx["hist"], query, ctx.get("frame_valid")
        )
        d_tau0 = self.decoder(skip, raw_last)
        gate = self.residual_gate(torch.cat([sat_now, sat_up], dim=-1))
        d_tau0 = d_tau0 * gate
        self.last_gate = gate
        sin_alt = torch.sin(torch.deg2rad(alt)).clamp(min=0.0)
        min_sin = math.sin(math.radians(HEAD_MIN_ALT_DEG))
        sin_geom = sin_alt.clamp(min=min_sin)
        tau0_p = (-torch.log(t_last.clamp(*T_CLIP)) * sin_geom).clamp(0.0, TAU_CAP)
        tau0_hat = (tau0_p + self.persist_scale * d_tau0).clamp(0.0, TAU_CAP)
        t_hat = torch.exp(-tau0_hat / sin_geom.clamp(min=1e-3)).clamp(*T_CLIP)
        self.last_skip = skip
        self.last_sin_alt = sin_alt
        self.last_T = t_hat
        return t_hat

    def forward(
        self,
        sky: torch.Tensor,
        sky_mask: torch.Tensor,
        sat: torch.Tensor,
        meta: torch.Tensor,
        meta_q: torch.Tensor,
        wind_set: torch.Tensor,
        wind_mask: torch.Tensor,
        wind_set_q: torch.Tensor,
        wind_mask_q: torch.Tensor,
        unix: torch.Tensor | None = None,
        sat_age_sec: torch.Tensor | None = None,
        lead_sec=None,
    ) -> torch.Tensor:
        ctx = self.encode_history(
            sky,
            sky_mask,
            sat,
            meta,
            meta_q,
            wind_set,
            wind_mask,
            wind_set_q,
            wind_mask_q,
            unix=unix,
            sat_age_sec=sat_age_sec,
        )
        return self._decode_lead(ctx, lead_sec)

    def forward_horizons(
        self,
        sky: torch.Tensor,
        sky_mask: torch.Tensor,
        sat: torch.Tensor,
        meta: torch.Tensor,
        meta_q: torch.Tensor,
        wind_set: torch.Tensor,
        wind_mask: torch.Tensor,
        wind_set_q: torch.Tensor,
        wind_mask_q: torch.Tensor,
        leads,
        unix: torch.Tensor | None = None,
        sat_age_sec: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Encode history once; decode each lead with its own upwind pierce."""
        ctx = self.encode_history(
            sky,
            sky_mask,
            sat,
            meta,
            meta_q,
            wind_set,
            wind_mask,
            wind_set_q,
            wind_mask_q,
            unix=unix,
            sat_age_sec=sat_age_sec,
        )
        return [self._decode_lead(ctx, h) for h in leads]

    def encode_history(
        self,
        sky: torch.Tensor,
        sky_mask: torch.Tensor,
        sat: torch.Tensor,
        meta: torch.Tensor,
        meta_q: torch.Tensor,
        wind_set: torch.Tensor,
        wind_mask: torch.Tensor,
        wind_set_q: torch.Tensor,
        wind_mask_q: torch.Tensor,
        unix: torch.Tensor | None = None,
        sat_age_sec: torch.Tensor | None = None,
    ) -> dict:
        if "sat" in self.ablate:
            sat = self._dummy_sat(sky, int(sky.shape[0]), int(sky.shape[1]))
        if "wind" in self.ablate:
            wind_set = torch.zeros_like(wind_set)
            wind_mask = torch.zeros_like(wind_mask)
            wind_set_q = torch.zeros_like(wind_set_q)
            wind_mask_q = torch.zeros_like(wind_mask_q)
        if "meta" in self.ablate:
            meta = torch.zeros_like(meta)
            meta_q = torch.zeros_like(meta_q)
        if "dream" in self.ablate:
            sky = torch.zeros_like(sky)
            sky_mask = torch.zeros_like(sky_mask)
        meta = meta.clone()
        meta[..., DOY_SLICE] = 0
        meta_q = meta_q.clone()
        meta_q[..., DOY_SLICE] = 0

        if self.training and self.frame_drop > 0:
            keep = sky.new_empty(sky.shape[0], sky.shape[1], 1).bernoulli_(
                1.0 - self.frame_drop
            )
            keep[:, -1] = 1
            sky = sky * keep
            sky_mask = sky_mask * keep

        sky_diff = torch.zeros_like(sky)
        sky_diff2 = torch.zeros_like(sky)
        t_sky = int(sky.shape[1])
        dt_min = frame_dt_sec(unix, t_sky, like=sky) / 60.0
        if t_sky > 1:
            pair = sky_mask[:, 1:] * sky_mask[:, :-1]
            sky_diff[:, 1:] = (
                (sky[:, 1:] - sky[:, :-1])
                / dt_min[:, 1:].unsqueeze(-1).clamp_min(DT_FLOOR_SEC / 60.0)
                * pair
            )
        if t_sky > 2:
            trip = sky_mask[:, 2:] * sky_mask[:, :-2]
            span = (dt_min[:, 2:] + dt_min[:, 1:-1]).clamp_min(
                DT_FLOOR_SEC / 60.0
            )
            sky_diff2[:, 2:] = (sky[:, 2:] - sky[:, :-2]) / span.unsqueeze(
                -1
            ) * trip

        sat = sat.clone()
        if sat.shape[2] > self.dacm_idx:
            sat[:, :, self.dacm_idx] = 0
        sat_sel = select_sat(sat, self.sat_idx)
        b, t, _, _, _ = sat_sel.shape
        n_pix = sky.shape[-1]
        device, dtype = sat_sel.device, sat_sel.dtype

        alt = self.pix_alt.to(device=device, dtype=dtype).unsqueeze(0).expand(b, -1)
        az = self.pix_az.to(device=device, dtype=dtype).unsqueeze(0).expand(b, -1)
        altaz_1 = altaz_feat(alt, az)
        altaz_bt = altaz_1.unsqueeze(1).expand(-1, t, -1, -1)
        sky_fields = torch.cat(
            [torch.stack([sky, sky_diff, sky_diff2], dim=-1), altaz_bt],
            dim=-1,
        )
        spatial = self.sky_enc(sky_fields, sky_mask)

        if sat_sel.shape[1] >= 2:
            now, prev, dt_sec = last_unique_scan_pair(sat_sel, unix, sat_age_sec)
            d_fog = (
                now[:, self.fog_idx : self.fog_idx + 1]
                - prev[:, self.fog_idx : self.fog_idx + 1]
            )
            d_c10 = (
                now[:, self.c10_idx : self.c10_idx + 1]
                - prev[:, self.c10_idx : self.c10_idx + 1]
            )
            scale = (dt_sec / 3600.0).view(b, 1, 1, 1).clamp_min(
                DT_FLOOR_SEC / 3600.0
            )
            d_fog = d_fog / scale
            d_c10 = d_c10 / scale
        else:
            now = sat_sel[:, -1]
            d_fog = now[:, self.fog_idx : self.fog_idx + 1] * 0
            d_c10 = now[:, self.c10_idx : self.c10_idx + 1] * 0
        extras = torch.cat([d_fog, d_c10], dim=1).unsqueeze(1).expand(b, t, -1, -1, -1)
        coords = sat_coord_maps(
            sat_sel.shape[-2],
            sat_sel.shape[-1],
            device=device,
            dtype=dtype,
        )
        sat_in = torch.cat(
            [
                sat_sel,
                extras,
                coords.view(1, 1, 2, *coords.shape[-2:]).expand(b, t, -1, -1, -1),
            ],
            dim=2,
        )

        grids, bads = self._pierce_grids(alt, az)
        acha_map = None
        if int(self.acha_pos.item()) >= 0:
            acha_map = sat_sel[
                :, -1, int(self.acha_pos.item()) : int(self.acha_pos.item()) + 1
            ]
        deck_w = self._acha_deck_weights(acha_map, grids, bads, b, n_pix, sat_sel)

        if wind_set.shape[1] > 1:
            ws = wind_set[:, -2:].reshape(b, -1, WIND_FEAT)
            wm = wind_mask[:, -2:].reshape(b, -1)
        else:
            ws = wind_set[:, -1]
            wm = wind_mask[:, -1]
        u, v, _h = decode_wind_uvh(ws, wm, **self._wind_decode_kwargs())
        u_deck, v_deck = decode_wind_uvh_decks(
            ws, wm, decks_km=DECKS_KM, **self._wind_decode_kwargs()
        )

        pierce_cnn_t, z_last = self._encode_pierce_cnn_t(
            sat_in, grids, bads, deck_w
        )
        key = sat_sel[:, -1].index_select(1, self.sphere_pos)
        sat_now = self._scale_acm(self._sample_mixed(key, grids, bads, deck_w))

        wind_tok = self.wind_enc(wind_set, wind_mask)
        meta_tok = self.meta_enc(meta)
        drop_sat = None
        drop_wind = None
        if self.training and self.modality_drop > 0:
            drop_sat = torch.rand(b, device=sky.device) < SAT_MODALITY_DROP
            drop_wind = torch.rand(b, 1, 1, device=sky.device) < WIND_MODALITY_DROP
            drop_meta = torch.rand(b, 1, 1, device=sky.device) < self.modality_drop
            pierce_cnn_t = pierce_cnn_t.masked_fill(drop_sat.view(b, 1, 1, 1), 0)
            keep_s = (~drop_sat).to(dtype=sat_now.dtype).view(b, 1, 1)
            sat_now = sat_now * keep_s
            wind_tok = wind_tok.masked_fill(drop_wind, 0)
            meta_tok = meta_tok.masked_fill(drop_meta, 0)

        pix = self.pix_fuse(torch.cat([spatial, pierce_cnn_t], dim=-1))
        wind_b = wind_tok.unsqueeze(2).expand(-1, -1, pix.shape[2], -1)
        meta_b = meta_tok.unsqueeze(2).expand(-1, -1, pix.shape[2], -1)
        hist = self.token_drop(
            self.fuse_norm(self.fuse(torch.cat([pix, wind_b, meta_b], dim=-1)))
        )
        hist = hist + self.time_emb(unix, like=sky, t=t).unsqueeze(2)

        if "dream" in self.ablate:
            t_last = sky.new_ones(sky.shape[0], sky.shape[-1])
        else:
            t_last = self._decode_last_t(sky[:, -1], sky_mask[:, -1])

        q_wind = self.wind_enc(wind_set_q.unsqueeze(1), wind_mask_q.unsqueeze(1))
        if self.training and self.modality_drop > 0 and drop_wind is not None:
            q_wind = q_wind.masked_fill(drop_wind, 0)
        q_meta = self.meta_enc(meta_q).unsqueeze(1)
        last_pix = pix[:, -1:]
        q_wind_b = q_wind.unsqueeze(2).expand(-1, -1, last_pix.shape[2], -1)
        q_meta_b = q_meta.unsqueeze(2).expand(-1, -1, last_pix.shape[2], -1)
        query = (
            self.fuse_norm(
                self.fuse(torch.cat([last_pix, q_wind_b, q_meta_b], dim=-1))
            )
            + self.query
        )
        return {
            "b": b,
            "hist": hist,
            "query": query,
            "t_last": t_last,
            "sky_diff_last": sky_diff[:, -1],
            "sat_now": sat_now,
            "z_last": z_last,
            "key": key,
            "deck_w": deck_w,
            "alt": alt,
            "az": az,
            "u": u,
            "v": v,
            "u_deck": u_deck,
            "v_deck": v_deck,
            "drop_sat": drop_sat,
            "frame_valid": sky_mask > 0,
            "unix": unix,
        }

    def compute_loss(self, t_hat, t_true, valid, sin_alt, **kwargs):
        """Airmass yield MSE. Extra kwargs are ignored."""
        return compute_loss(t_hat, t_true, valid, sin_alt, **kwargs)


def build_model(nside: int, cfg) -> SkyMapForecaster:
    """Evaluator entry point. Wire new modules inside SkyMapForecaster, not here."""
    return SkyMapForecaster(
        nside=nside,
        n_frames=int(getattr(cfg, "n_frames", 24)),
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        sky_width=cfg.sky_width,
        sat_width=getattr(cfg, "sat_width", 32),
        dropout=cfg.dropout,
        modality_drop=cfg.modality_drop,
        frame_drop=cfg.frame_drop,
        wind_age_mean=float(getattr(cfg, "wind_age_mean", 0.0)),
        wind_age_std=float(getattr(cfg, "wind_age_std", 1.0)),
        wind_u_med=float(getattr(cfg, "wind_u_med", WIND_U_MED)),
        wind_u_iqr=float(getattr(cfg, "wind_u_iqr", WIND_U_IQR)),
        wind_v_med=float(getattr(cfg, "wind_v_med", WIND_V_MED)),
        wind_v_iqr=float(getattr(cfg, "wind_v_iqr", WIND_V_IQR)),
        wind_h_med=float(getattr(cfg, "wind_h_med", WIND_H_MED)),
        wind_h_iqr=float(getattr(cfg, "wind_h_iqr", WIND_H_IQR)),
    )


def load_backbone(model: SkyMapForecaster, state_dict: dict) -> tuple[list[str], list[str]]:
    """Copy matching encoder/decoder tensors; interpolate ``pos`` over time."""
    own = model.state_dict()
    mapped: dict[str, torch.Tensor] = {}
    skip_prefixes = ("residual_gate.",)
    for key, val in state_dict.items():
        name = key[7:] if key.startswith("module.") else key
        if name not in own or name.startswith(skip_prefixes):
            continue
        tgt = own[name]
        if val.shape == tgt.shape:
            mapped[name] = val
            continue
        if name == "pos" and val.ndim == tgt.ndim == 4:
            new_t = int(tgt.shape[1])
            x = val.squeeze(2).permute(0, 2, 1)
            x = nn.functional.interpolate(x, size=new_t, mode="linear", align_corners=True)
            mapped[name] = x.permute(0, 2, 1).unsqueeze(2)
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    return list(missing), list(unexpected)


def load_matching(model: SkyMapForecaster, state_dict: dict) -> tuple[list[str], list[str]]:
    """Copy every same-shape tensor, including ``residual_gate``.

    Use this for same-architecture fine-tune. ``load_backbone`` still skips
    the residual gate so a T=24 warm-start into multihorizon can relearn it.
    """
    own = model.state_dict()
    mapped: dict[str, torch.Tensor] = {}
    for key, val in state_dict.items():
        name = key[7:] if key.startswith("module.") else key
        if name not in own:
            continue
        if val.shape == own[name].shape:
            mapped[name] = val
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    return list(missing), list(unexpected)


