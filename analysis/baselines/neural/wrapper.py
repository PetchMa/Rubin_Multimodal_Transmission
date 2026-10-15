"""Camera-only ConvLSTM and SimVP nowcasts.

The network predicts a change in zenith optical depth. A zero residual
leaves the last observed transmission in place. ``forward_horizons``
encodes the camera history once and decodes each lead.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from analysis.baselines.neural.convlstm import ConvLSTMNet
from analysis.baselines.neural.lead import LEAD_SKIP_PREFIXES
from analysis.baselines.neural.simvp import SimVPNet
from analysis.healpix_torch import gaussian_smooth_healpix
from analysis.forecaster.geom import LEAD_SEC, local_pix_altaz
from analysis.our_model.mask_train import LABEL_SMOOTH_SIGMA, masked_airmass_yield_mse
from analysis.our_model.model import _dataset_root

CAMERA_ARRAY_KEYS = (
    "sky",
    "sky_mask",
    "y",
    "y_mask",
    "last",
    "last_mask",
    "unix",
)

IN_CH = 2
TAU_CAP = 2.5
T_CLIP = (0.05, 0.999)
HEAD_MIN_ALT_DEG = 20.0


def slim_camera_pairs(data) -> None:
    """Drop sat/wind memmaps. SimVP / ConvLSTM only read camera T and masks."""
    root = _dataset_root(data)
    if root._shapes is None:
        return
    root._arrays = None
    root._shapes = {k: v for k, v in root._shapes.items() if k in CAMERA_ARRAY_KEYS}


def _strip_key(name: str) -> str:
    if name.startswith("module."):
        name = name[7:]
    return name.replace("._orig_mod.", ".")


def load_nowcast_backbone(
    model: nn.Module, state_dict: dict
) -> tuple[list[str], list[str]]:
    """Copy matching tensors; leave the centered lead MLP at identity init.

    Load matching tensors. The residual gate is left at its initialization.
    """
    own = model.state_dict()
    mapped: dict[str, torch.Tensor] = {}
    for key, val in state_dict.items():
        name = _strip_key(key)
        if name not in own or name.startswith(LEAD_SKIP_PREFIXES):
            continue
        if val.shape == own[name].shape:
            mapped[name] = val
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    return list(missing), list(unexpected)


def load_nowcast_matching(
    model: nn.Module, state_dict: dict
) -> tuple[list[str], list[str]]:
    """Copy every same-shape tensor, including the lead MLP."""
    own = model.state_dict()
    mapped: dict[str, torch.Tensor] = {}
    for key, val in state_dict.items():
        name = _strip_key(key)
        if name not in own:
            continue
        if val.shape == own[name].shape:
            mapped[name] = val
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    return list(missing), list(unexpected)


class CameraNowcast(nn.Module):
    """Decode encoded sky → ``(T, valid)`` channels → ``T̂`` (or residual logits)."""

    name: str = "camera_nowcast"

    def __init__(
        self,
        net: nn.Module,
        *,
        nside: int,
        n_frames: int,
        sky_mean: float,
        sky_std: float,
        name: str,
    ) -> None:
        super().__init__()
        self.net = net
        self.nside = int(nside)
        self.n_frames = int(n_frames)
        self.d_model = int(getattr(net, "hid_t", getattr(net, "hidden", 0)) or 0)
        self.name = str(name)
        self.lead_ref_sec = float(LEAD_SEC)
        self.persist_in_residual = True
        self.output_transmission = True
        # ``off`` = plain airmass yield MSE. ``banded`` matches forecaster
        # ``program.compute_loss`` (GOLDEN_GRADE_LOSS default).
        self.grade_loss_mode = "off"
        self.register_buffer("sky_mean", torch.tensor(float(sky_mean)))
        self.register_buffer("sky_std", torch.tensor(float(sky_std)))
        alt, _az = local_pix_altaz(self.nside)
        self.register_buffer("pix_alt", torch.from_numpy(np.asarray(alt, dtype=np.float32)))
        self.last_sin_alt: torch.Tensor | None = None
        self.last_T: torch.Tensor | None = None

    def decode_t(self, encoded: torch.Tensor) -> torch.Tensor:
        return encoded * self.sky_std + self.sky_mean

    def _camera_input(self, t: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = (mask > 0).to(dtype=t.dtype)
        return torch.stack([t * valid, valid], dim=2)

    def _persist_parts(
        self, t: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t_last = t[:, -1]
        if float(LABEL_SMOOTH_SIGMA) > 0.0:
            t_last = gaussian_smooth_healpix(
                t_last, mask[:, -1] > 0, sigma=float(LABEL_SMOOTH_SIGMA)
            )
        alt = self.pix_alt.to(device=t.device, dtype=t.dtype)
        sin_alt = torch.sin(torch.deg2rad(alt)).clamp(min=0.0)
        min_sin = math.sin(math.radians(HEAD_MIN_ALT_DEG))
        sin_geom = sin_alt.clamp(min=min_sin)
        tau0_p = (-torch.log(t_last.clamp(*T_CLIP)) * sin_geom).clamp(0.0, TAU_CAP)
        sin_alt_b = sin_alt.unsqueeze(0).expand(t.shape[0], -1)
        self.last_sin_alt = sin_alt_b
        return tau0_p, sin_geom, sin_alt_b

    def _t_from_residual(
        self, residual: torch.Tensor, tau0_p: torch.Tensor, sin_geom: torch.Tensor
    ) -> torch.Tensor:
        tau0_hat = (tau0_p + residual).clamp(0.0, TAU_CAP)
        t_hat = torch.exp(-tau0_hat / sin_geom.clamp(min=1e-3)).clamp(*T_CLIP)
        self.last_T = t_hat
        return t_hat

    def residual_from_t(
        self, t: torch.Tensor, mask: torch.Tensor, lead_sec=None
    ) -> torch.Tensor:
        x = self._camera_input(t, mask)
        out = self.net(x, lead_sec=lead_sec)
        if not self.output_transmission:
            return out
        tau0_p, sin_geom, _ = self._persist_parts(t, mask)
        return self._t_from_residual(out, tau0_p, sin_geom)

    def forward(
        self,
        sky: torch.Tensor,
        sky_mask: torch.Tensor,
        *args: Any,
        lead_sec=None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del args, kwargs
        t = self.decode_t(sky)
        return self.residual_from_t(t, sky_mask, lead_sec=lead_sec)

    def forward_horizons(
        self,
        sky: torch.Tensor,
        sky_mask: torch.Tensor,
        *args: Any,
        leads,
        unix: torch.Tensor | None = None,
        sat_age_sec: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> list[torch.Tensor]:
        """Encode history once; decode each lead with its own residual."""
        del args, unix, sat_age_sec, kwargs
        t = self.decode_t(sky)
        ctx = self.net.encode(self._camera_input(t, sky_mask))
        if not self.output_transmission:
            return [self.net.decode(ctx, h) for h in leads]
        tau0_p, sin_geom, _ = self._persist_parts(t, sky_mask)
        hats = []
        for h in leads:
            residual = self.net.decode(ctx, h)
            hats.append(self._t_from_residual(residual, tau0_p, sin_geom))
        return hats

    def compute_loss(self, t_hat, t_true, valid, sin_alt, **kwargs):
        """Same train loss as forecaster golden when ``grade_loss_mode`` is banded.

        ``off`` keeps the paper SimVP objective (airmass yield MSE only).
        """
        mode = str(
            kwargs.get("grade_loss_mode", getattr(self, "grade_loss_mode", "off"))
        ).strip().lower()
        if mode in {"", "off", "none", "mse", "yield_mse"}:
            return masked_airmass_yield_mse(t_hat, t_true, valid, sin_alt)
        from analysis.forecaster.program import compute_loss as grade_loss

        kwargs = dict(kwargs)
        kwargs["grade_loss_mode"] = mode
        return grade_loss(t_hat, t_true, valid, sin_alt, **kwargs)


def convlstm_hparams() -> dict[str, int]:
    # Matched to SimVP (~120k): 119,188 params at hidden=57, 2 layers
    # before the centered lead MLP (~3k).
    return {"hidden": 57, "n_layers": 2, "in_ch": IN_CH}


def simvp_hparams() -> dict[str, int]:
    # About 117k parameters before the lead embedding.
    return {"hid_s": 32, "hid_t": 64, "n_s": 2, "n_t": 4, "in_ch": IN_CH}


def build_nowcast(
    arch: str,
    *,
    nside: int,
    n_frames: int,
    sky_mean: float,
    sky_std: float,
) -> CameraNowcast:
    arch = str(arch).strip().lower()
    if arch in {"convlstm", "conv_lstm"}:
        hp = convlstm_hparams()
        net = ConvLSTMNet(
            nside=nside,
            in_ch=hp["in_ch"],
            hidden=hp["hidden"],
            n_layers=hp["n_layers"],
        )
        name = "convlstm"
    elif arch in {"simvp", "temporal_unet", "unet"}:
        hp = simvp_hparams()
        net = SimVPNet(
            nside=nside,
            n_frames=n_frames,
            in_ch=hp["in_ch"],
            hid_s=hp["hid_s"],
            hid_t=hp["hid_t"],
            n_s=hp["n_s"],
            n_t=hp["n_t"],
        )
        name = "simvp"
    else:
        raise ValueError(f"unknown neural baseline {arch!r}")
    return CameraNowcast(
        net,
        nside=nside,
        n_frames=n_frames,
        sky_mean=sky_mean,
        sky_std=sky_std,
        name=name,
    )


def _payload_sky_stats(payload: dict) -> tuple[float, float]:
    if payload.get("sky_mean") is not None and payload.get("sky_std") is not None:
        return float(payload["sky_mean"]), float(payload["sky_std"])
    sky = (payload.get("norm") or {}).get("sky") or {}
    if sky.get("mean") is None or sky.get("std") is None:
        raise KeyError("checkpoint missing sky_mean/sky_std and norm.sky")
    return float(sky["mean"]), float(sky["std"])


def nowcast_from_checkpoint(path, device: torch.device) -> CameraNowcast:
    payload = torch.load(path, map_location=device, weights_only=False)
    info = payload.get("info") or {}
    arch = payload.get("arch") or info.get("arch")
    sky_mean, sky_std = _payload_sky_stats(payload)
    model = build_nowcast(
        str(arch),
        nside=int(payload["nside"]),
        n_frames=int(payload.get("n_frames") or 8),
        sky_mean=sky_mean,
        sky_std=sky_std,
    )
    if "output_transmission" in info:
        model.output_transmission = bool(info["output_transmission"])
    if info.get("grade_loss_mode") is not None:
        model.grade_loss_mode = str(info["grade_loss_mode"])
    model.load_state_dict(payload["state_dict"], strict=False)
    return model.to(device)
