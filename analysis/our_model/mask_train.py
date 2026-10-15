"""Airmass-weighted transmission loss and the forecast readout.

The forecaster emits transmission. Training minimizes yield MSE, with an
optional hinge on false clears. ``model_logits`` reads a checkpoint that
either returns transmission directly or a residual on the last frame.
"""

from __future__ import annotations

import inspect
import math

import torch
import torch.nn as nn

from analysis.our_model.model import point_residual

CLOUD_T_THRESH = 0.9
MASK_PROB_THRESH = 0.5
PERSIST_LOGIT = 4.0
SCHEDULER_MIN_ALT_DEG = 30.0
LABEL_SMOOTH_SIGMA = 3.5
FALSE_CLEAR_T = 0.63


def _smooth_t(t: torch.Tensor, valid: torch.Tensor, sigma: float) -> torch.Tensor:
    if float(sigma) <= 0.0:
        return t
    from analysis.healpix_torch import gaussian_smooth_healpix

    return gaussian_smooth_healpix(t, valid > 0, sigma=float(sigma))


def decode_camera_t(
    field: torch.Tensor,
    mask: torch.Tensor,
    sky_norm,
    *,
    sigma: float = 0.0,
):
    t = sky_norm.decode_t(field)
    valid = mask > 0
    return _smooth_t(t, valid, sigma), valid.to(dtype=torch.float32)


def masked_airmass_yield_mse(
    t_hat: torch.Tensor,
    t_true: torch.Tensor,
    valid: torch.Tensor,
    sin_alt: torch.Tensor,
    *,
    min_alt_deg: float = SCHEDULER_MIN_ALT_DEG,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Airmass-weighted MSE on transmission. Weight ``sin(alt)``; 0 below cutoff."""
    w = sin_alt.to(dtype=t_hat.dtype).clamp(min=0.0)
    min_sin = math.sin(math.radians(float(min_alt_deg)))
    w = torch.where(w >= min_sin, w, torch.zeros_like(w))
    se = (t_hat.clamp(0.0, 1.0) - t_true.clamp(0.0, 1.0)).square()
    e = se.reshape(se.shape[0], -1)
    m = valid.reshape(valid.shape[0], -1).to(dtype=e.dtype)
    ww = w.reshape(w.shape[0], -1) * m
    den = ww.sum(dim=-1)
    ok = den > 0
    if int(ok.sum().item()) == 0:
        return e.new_zeros(())
    per = (e * ww).sum(dim=-1) / den.clamp_min(eps)
    return per[ok].mean()


def masked_false_clear_hinge(
    t_hat: torch.Tensor,
    t_true: torch.Tensor,
    valid: torch.Tensor,
    sin_alt: torch.Tensor,
    *,
    thresh: float = FALSE_CLEAR_T,
    min_alt_deg: float = SCHEDULER_MIN_ALT_DEG,
) -> torch.Tensor:
    """Mean ``relu(T_hat - thresh)`` on cloudy tiles above the altitude cutoff.

    A tile is cloudy when the observed transmission is at or below ``thresh``.
    The hinge is zero once the forecast is already at or below the cut.
    """
    cut = float(thresh)
    min_sin = math.sin(math.radians(float(min_alt_deg)))
    alt = sin_alt.to(dtype=t_hat.dtype)
    while alt.ndim < t_hat.ndim:
        alt = alt.unsqueeze(0)
    cloud = (t_true <= cut) & (valid > 0) & (alt >= min_sin)
    if int(cloud.sum().item()) == 0:
        return t_hat.new_zeros(())
    return (t_hat - cut).clamp(min=0.0)[cloud].mean()


def persist_logits(
    last: torch.Tensor,
    last_mask: torch.Tensor,
    sky_norm,
    thresh: float,
    *,
    band_delta: float = 0.0,
):
    """Persist cloud logits. ``band_delta>0`` leaves |T-thresh|<delta at 0."""
    t = sky_norm.decode_t(last)
    valid = last_mask > 0
    if band_delta <= 0:
        cloud = t < thresh
        sign = 2.0 * cloud.to(dtype=t.dtype) - 1.0
        return sign * PERSIST_LOGIT * valid.to(dtype=t.dtype)
    sign = torch.zeros_like(t)
    sign = torch.where(t < thresh - band_delta, torch.ones_like(t), sign)
    sign = torch.where(t > thresh + band_delta, -torch.ones_like(t), sign)
    return sign * PERSIST_LOGIT * valid.to(dtype=t.dtype)




def model_logits(
    model: nn.Module,
    b: dict[str, torch.Tensor],
    sky_norm,
    thresh: float,
    *,
    band_delta: float = 0.0,
):
    kwargs = {}
    fwd = getattr(model, "forward", model)
    try:
        import inspect

        if "unix" in inspect.signature(fwd).parameters and "unix" in b:
            kwargs["unix"] = b["unix"]
        if "sat_age_sec" in inspect.signature(fwd).parameters and "sat_age_sec" in b:
            kwargs["sat_age_sec"] = b["sat_age_sec"]
        if "lead_sec" in inspect.signature(fwd).parameters:
            if "lead_sec" in b:
                kwargs["lead_sec"] = b["lead_sec"]
            elif "horizon_sec" in b:
                kwargs["lead_sec"] = b["horizon_sec"]
    except (TypeError, ValueError):
        pass
    extra_keys = (
        "sat",
        "meta",
        "meta_q",
        "wind_set",
        "wind_mask",
        "wind_set_q",
        "wind_mask_q",
    )
    extra = [b[k] for k in extra_keys] if all(k in b for k in extra_keys) else []
    out = model(b["sky"], b["sky_mask"], *extra, **kwargs)
    resid = point_residual(out)
    if getattr(model, "persist_in_residual", False) or getattr(
        model, "output_transmission", False
    ):
        return resid
    return persist_logits(
        b["last"], b["last_mask"], sky_norm, thresh, band_delta=band_delta
    ) + resid

