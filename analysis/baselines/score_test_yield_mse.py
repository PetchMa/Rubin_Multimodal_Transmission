#!/usr/bin/env python3
"""Test-holdout scores for the paper table.

The target is σ=3.5-smoothed T on moon-dark windows (illum<0.25 or alt≤0°),
Windows that are more than 99% cloudy are dropped. Maps are horizon-locked.

Paper metrics: MAYE, false-clear η_FC at T≤0.63, Δm_5σ, and D_eff.
Yield MSE is the train surrogate, also logged.

Methods: persist, optical flow, Segev frozen-flow CC, Lagrangian winds
(smoothed with σ=3.5), ConvLSTM, SimVP, and the multimodal forecaster.

    python -u -m analysis.baselines.score_test_yield_mse
"""

from __future__ import annotations

from analysis.paths import MNT

import argparse
import datetime as dt
import json
import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

from analysis.baselines.lagrangian_winds import LagrangianWindAdvection
from analysis.baselines.neural.filter_overcast import (
    CLOUD_T_THRESH,
    window_cloudy_frac,
    window_moon_ok,
)
from analysis.baselines.neural.wrapper import nowcast_from_checkpoint, slim_camera_pairs
from analysis.baselines.optical_flow import DenseOpticalFlow
from analysis.baselines.segev_crosscorr import FrozenFlowCrossCorr
from analysis.eval_harness.data import HORIZON_SEC
from analysis.healpix_torch import gaussian_smooth_healpix
from analysis.forecaster import program as forecaster
from analysis.forecaster.carve_train_val import DEFAULT_OUT as _UNUSED
from analysis.forecaster.geom import (
    LEAD_SEC,
    attach_wind_decode_stats,
    local_pix_altaz,
    nside_from_npix,
)
from analysis.forecaster.port_prep import attach_horizon_lock, lock_numpy_batch
from analysis.forecaster.sat_ingest import wrap_rich_sat
from analysis.forecaster.train import _set_sky_norm
from analysis.our_model.sat_pretrain.dataset import NATIVE_IR_HW
from analysis.our_model.mask_train import (
    LABEL_SMOOTH_SIGMA,
    SCHEDULER_MIN_ALT_DEG,
    decode_camera_t,
    masked_airmass_yield_mse,
    model_logits,
)
from analysis.our_model.model import (
    JD_UNIX_EPOCH,
    PairsDataset,
    TrainConfig,
    apply_batch_preprocess,
    make_loader,
    resolve_pairs,
    _subsample_idx,
    _to_device,
)
from analysis.satellite.sites import MOON_ALT_MIN_DEG, MOON_ILLUM_MAX
from analysis.torch_util import resolve_device

del _UNUSED

DEFAULT_CACHE = Path(
    str(MNT) + "/pairs/dream_20min_encoded_t24"
)
DEFAULT_OUT = Path(
    str(MNT) + "/baselines_neural/"
    "suite_test_yield_mse_sys.json"
)
SIMVP_CKPT = Path(
    str(MNT) + "/baselines_neural/paper_simvp_multih_t24/seed_42/simvp/best.pt"
)
PROPOSED_CKPT = Path(
    str(MNT) + "/baselines_neural/paper_multih_t24/seed_42/model/best.pt"
)
MAX_CLOUDY_FRAC = 0.99
SIGMA = float(LABEL_SMOOTH_SIGMA)
MIN_SIN = math.sin(math.radians(SCHEDULER_MIN_ALT_DEG))
MASK_CLOUD_T = 0.63
D_RUBIN_M = 8.4
T_LOG_FLOOR = 1e-4
PERSIST_HARD_DT = 0.10
REGIME_NAMES = ("incoming", "photometric", "persist_hard")
CLASSICAL_NAMES = (
    "eulerian_persistence",
    "dense_optical_flow",
    "frozen_flow_crosscorr",
    "lagrangian_wind_advection",
)


def per_window_yield_mse(
    t_hat: np.ndarray,
    t_true: np.ndarray,
    valid: np.ndarray,
    sin_alt: np.ndarray,
) -> np.ndarray:
    w = np.where(sin_alt >= MIN_SIN, np.clip(sin_alt, 0.0, None), 0.0) * valid
    se = (np.clip(t_hat, 0.0, 1.0) - np.clip(t_true, 0.0, 1.0)) ** 2
    den = w.sum(axis=-1)
    out = np.full(t_hat.shape[0], np.nan, dtype=np.float64)
    ok = den > 0
    out[ok] = (se[ok] * w[ok]).sum(axis=-1) / den[ok]
    return out


def per_window_plain_mse(
    t_hat: np.ndarray,
    t_true: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    se = (np.clip(t_hat, 0.0, 1.0) - np.clip(t_true, 0.0, 1.0)) ** 2
    n = valid.sum(axis=-1)
    out = np.full(t_hat.shape[0], np.nan, dtype=np.float64)
    ok = n > 0
    out[ok] = (se[ok] * valid[ok]).sum(axis=-1) / n[ok]
    return out


def _airmass_w(valid: np.ndarray, sin_alt: np.ndarray) -> np.ndarray:
    sa = np.asarray(sin_alt)
    if sa.ndim == 1:
        sa = np.broadcast_to(sa, valid.shape)
    return np.where(sa >= MIN_SIN, np.clip(sa, 0.0, None), 0.0) * valid


def per_window_weighted(values: np.ndarray, valid: np.ndarray, sin_alt: np.ndarray) -> np.ndarray:
    w = _airmass_w(valid, sin_alt)
    den = w.sum(axis=-1)
    out = np.full(values.shape[0], np.nan, dtype=np.float64)
    ok = den > 0
    out[ok] = (values[ok] * w[ok]).sum(axis=-1) / den[ok]
    return out


def paper_window_metrics(
    t_hat: np.ndarray, t_true: np.ndarray, valid: np.ndarray, sin_alt: np.ndarray
) -> dict[str, np.ndarray]:
    th = np.clip(t_hat, 0.0, 1.0)
    ty = np.clip(t_true, 0.0, 1.0)
    th_l = np.clip(th, T_LOG_FLOOR, 1.0)
    ty_l = np.clip(ty, T_LOG_FLOOR, 1.0)
    dm = 1.25 * np.log10(th_l / ty_l)
    de = D_RUBIN_M * (np.sqrt(th) - np.sqrt(ty))
    return {
        "yield_mse": per_window_weighted((th - ty) ** 2, valid, sin_alt),
        "maye": per_window_weighted(np.abs(th - ty), valid, sin_alt),
        "dm5": per_window_weighted(dm, valid, sin_alt),
        "dm5_mae": per_window_weighted(np.abs(dm), valid, sin_alt),
        "deff_mae": per_window_weighted(np.abs(de), valid, sin_alt),
    }


def _empty_deff() -> dict[str, float]:
    return {
        "sum_sqrt_t_pred_clear": 0.0,
        "n_pred_clear": 0,
        "sum_sqrt_t_true_clear": 0.0,
        "n_true_clear": 0,
    }


def accumulate_deff(
    acc: dict[str, float],
    t_hat: np.ndarray,
    t_true: np.ndarray,
    valid: np.ndarray,
    last_valid: np.ndarray,
    sin_alt: np.ndarray,
) -> None:
    w = _airmass_w(valid, sin_alt) > 0
    conf = w & np.asarray(last_valid, dtype=bool)
    ty = np.clip(t_true, 0.0, 1.0)
    th = np.clip(t_hat, 0.0, 1.0)
    pred_clear = (th > MASK_CLOUD_T) & conf
    true_clear = (ty > MASK_CLOUD_T) & conf
    if pred_clear.any():
        acc["sum_sqrt_t_pred_clear"] += float(np.sqrt(ty[pred_clear]).sum())
        acc["n_pred_clear"] += int(pred_clear.sum())
    if true_clear.any():
        acc["sum_sqrt_t_true_clear"] += float(np.sqrt(ty[true_clear]).sum())
        acc["n_true_clear"] += int(true_clear.sum())


def finalize_deff(acc: dict[str, float]) -> dict[str, float]:
    n_p = int(acc["n_pred_clear"])
    n_t = int(acc["n_true_clear"])
    deff_pred = D_RUBIN_M * acc["sum_sqrt_t_pred_clear"] / n_p if n_p else float("nan")
    deff_true = D_RUBIN_M * acc["sum_sqrt_t_true_clear"] / n_t if n_t else float("nan")
    return {
        "deff_on_pred_clear_m": float(deff_pred),
        "deff_on_true_clear_m": float(deff_true),
        "deff_clear_gap_m": (
            float(deff_true - deff_pred)
            if math.isfinite(deff_pred) and math.isfinite(deff_true)
            else float("nan")
        ),
        "n_pred_clear": n_p,
        "n_true_clear": n_t,
    }


def _mean_finite(x: np.ndarray) -> float:
    y = x[np.isfinite(x)]
    return float(y.mean()) if y.size else float("nan")


def summarize_windows(yield_w: np.ndarray, plain_w: np.ndarray) -> dict:
    y = yield_w[np.isfinite(yield_w)]
    p = plain_w[np.isfinite(plain_w)]
    return {
        "yield_mse": float(y.mean()) if y.size else float("nan"),
        "plain_mse": float(p.mean()) if p.size else float("nan"),
        "n_windows": int(y.size),
    }


def add_paper_fields(row: dict, paper_w: dict[str, np.ndarray], deff: dict[str, float]) -> dict:
    row["maye"] = _mean_finite(paper_w["maye"])
    row["dm5"] = _mean_finite(paper_w["dm5"])
    row["dm5_mae"] = _mean_finite(paper_w["dm5_mae"])
    row["deff_mae_m"] = _mean_finite(paper_w["deff_mae"])
    row.update(deff)
    return row


def mask_counts(
    t_hat: np.ndarray,
    t_true: np.ndarray,
    valid: np.ndarray,
    sin_alt: np.ndarray,
    last_valid: np.ndarray | None = None,
    *,
    thresh: float = MASK_CLOUD_T,
) -> dict[str, int]:
    """Pooled tiles: cloudy = T ≤ thresh. Support is y ∩ last ∩ alt≥30°."""
    conf = np.asarray(valid, dtype=bool)
    if last_valid is not None:
        conf = conf & np.asarray(last_valid, dtype=bool)
    sa = np.asarray(sin_alt)
    if sa.ndim == 1:
        conf = conf & (sa >= MIN_SIN)
    else:
        conf = conf & (sa >= MIN_SIN)
    gold = (np.clip(t_true, 0.0, 1.0) <= thresh) & conf
    pred = (np.clip(t_hat, 0.0, 1.0) <= thresh) & conf
    tp = int((pred & gold).sum())
    fn = int((~pred & gold).sum())
    fp = int((pred & ~gold & conf).sum())
    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "n_gold_cloud": tp + fn,
        "n_conf": int(conf.sum()),
    }


def false_clear_from_counts(c: dict[str, int]) -> float:
    n = int(c.get("n_gold_cloud", int(c.get("tp", 0)) + int(c.get("fn", 0))))
    return float(c["fn"] / n) if n else float("nan")


def add_mask_fields(row: dict, counts: dict[str, int]) -> dict:
    row["false_clear"] = false_clear_from_counts(counts)
    row["false_clear_thresh"] = MASK_CLOUD_T
    row["false_clear_counts"] = {k: int(v) for k, v in counts.items()}
    return row


def support_mask(
    valid: np.ndarray,
    last_valid: np.ndarray,
    sin_alt: np.ndarray,
) -> np.ndarray:
    """y ∩ last ∩ alt≥30°, same tiles as η_FC."""
    sa = np.asarray(sin_alt)
    if sa.ndim == 1:
        sa = np.broadcast_to(sa, valid.shape)
    return (
        np.asarray(valid, dtype=bool)
        & np.asarray(last_valid, dtype=bool)
        & (sa >= MIN_SIN)
    )


def regime_masks(
    t_last: np.ndarray,
    t_true: np.ndarray,
    support: np.ndarray,
    *,
    persist_hard_dt: float = PERSIST_HARD_DT,
) -> dict[str, np.ndarray]:
    """Complementarity slices. Overlapping; not a partition.

    incoming: last clear, gold cloudy (persist is a false-clear by construction).
    photometric: last and gold both clear (DREAM calibration).
    persist_hard: |T_gold − T_last| ≥ persist_hard_dt on the same support.
    """
    tl = np.clip(t_last, 0.0, 1.0)
    ty = np.clip(t_true, 0.0, 1.0)
    last_clear = (tl > MASK_CLOUD_T) & support
    gold_clear = (ty > MASK_CLOUD_T) & support
    gold_cloud = (ty <= MASK_CLOUD_T) & support
    return {
        "incoming": last_clear & gold_cloud,
        "photometric": last_clear & gold_clear,
        "persist_hard": support & (np.abs(ty - tl) >= float(persist_hard_dt)),
    }


def _empty_regime_acc() -> dict:
    return {
        "paper": {k: [] for k in ("yield_mse", "maye", "dm5", "dm5_mae", "deff_mae")},
        "plain": [],
        "counts": {"tp": 0, "fn": 0, "fp": 0, "n_gold_cloud": 0, "n_conf": 0},
        "deff": _empty_deff(),
        "n_tiles": 0,
    }


def _update_regime(
    acc: dict,
    pred: np.ndarray,
    t_true: np.ndarray,
    regime: np.ndarray,
    sin_alt: np.ndarray,
    last_valid: np.ndarray,
) -> None:
    if not regime.any():
        return
    pm = paper_window_metrics(pred, t_true, regime, sin_alt)
    for k, arr in pm.items():
        acc["paper"][k].append(arr)
    acc["plain"].append(per_window_plain_mse(pred, t_true, regime))
    accumulate_deff(acc["deff"], pred, t_true, regime, last_valid, sin_alt)
    part = mask_counts(pred, t_true, regime, sin_alt, last_valid)
    for k in acc["counts"]:
        acc["counts"][k] += int(part[k])
    acc["n_tiles"] += int(regime.sum())


def _merge_regime(dst: dict, src: dict) -> None:
    """Fold a worker's regime accumulator into the parent's."""
    for k, v in src["paper"].items():
        dst["paper"][k].extend(v)
    dst["plain"].extend(src["plain"])
    for k in dst["counts"]:
        dst["counts"][k] += int(src["counts"][k])
    for k in dst["deff"]:
        dst["deff"][k] += src["deff"][k]
    dst["n_tiles"] += int(src["n_tiles"])


def _finalize_regime(acc: dict) -> dict:
    n_tiles = int(acc["n_tiles"])
    if not acc["paper"]["maye"]:
        row = {
            "yield_mse": float("nan"),
            "plain_mse": float("nan"),
            "n_windows": 0,
            "n_tiles": n_tiles,
            "maye": float("nan"),
            "false_clear": float("nan"),
            "dm5": float("nan"),
            "deff_mae_m": float("nan"),
        }
        return add_mask_fields(row, acc["counts"])
    cat = {k: np.concatenate(v) for k, v in acc["paper"].items()}
    row = add_mask_fields(
        summarize_windows(cat["yield_mse"], np.concatenate(acc["plain"])),
        acc["counts"],
    )
    add_paper_fields(row, cat, finalize_deff(acc["deff"]))
    row["n_tiles"] = n_tiles
    return row


def window_mean_t(
    y: np.ndarray,
    y_mask: np.ndarray,
    *,
    mean: float,
    std: float,
) -> np.ndarray:
    """Per-window mean observed transmission on valid tiles. NaN when no valid tiles."""
    valid = np.asarray(y_mask) > 0
    t = np.asarray(y) * float(std) + float(mean)
    n_valid = valid.reshape(valid.shape[0], -1).sum(axis=1).astype(np.float64)
    acc = np.where(valid, t, 0.0).reshape(t.shape[0], -1).sum(axis=1)
    out = np.full(y.shape[0], np.nan, dtype=np.float64)
    ok = n_valid > 0
    out[ok] = acc[ok] / n_valid[ok]
    return out


def test_keep_indices(
    cache: Path,
    *,
    cloudy_min: float = 0.0,
    cloudy_max: float = MAX_CLOUDY_FRAC,
    cloudy_thresh: float = CLOUD_T_THRESH,
    mean_t_min: float | None = None,
    mean_t_max: float | None = None,
    y: np.ndarray | None = None,
    y_mask: np.ndarray | None = None,
    hit: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    ds = PairsDataset.from_memmap(cache, resolve_device("cpu"))
    n = int(ds.n)
    arr = ds._ensure_arrays()
    sky = ds.norm.sky
    y = np.asarray(arr["y"] if y is None else y)
    ym = np.asarray(arr["y_mask"] if y_mask is None else y_mask)
    moon = window_moon_ok(cache, n)
    frac = window_cloudy_frac(
        y,
        ym,
        mean=float(sky.mean),
        std=float(sky.std),
        thresh=float(cloudy_thresh),
    )
    lo, hi = float(cloudy_min), float(cloudy_max)
    cloudy_ok = np.isfinite(frac) & (frac >= lo) & (frac <= hi)
    ok = moon & cloudy_ok
    if hit is not None:
        hit_ok = np.asarray(hit).reshape(n) > 0
        ok = ok & hit_ok
    else:
        hit_ok = None
    mean_t = None
    t_ok = None
    if mean_t_min is not None or mean_t_max is not None:
        mean_t = window_mean_t(y, ym, mean=float(sky.mean), std=float(sky.std))
        t_lo = float("-inf") if mean_t_min is None else float(mean_t_min)
        t_hi = float("inf") if mean_t_max is None else float(mean_t_max)
        t_ok = np.isfinite(mean_t) & (mean_t >= t_lo) & (mean_t <= t_hi)
        ok = ok & t_ok
    keep = np.where(ok)[0].astype(np.int64)
    kept_frac = frac[keep]
    kept_t = mean_t[keep] if mean_t is not None else None
    info = {
        "n_cache": n,
        "n_moon_dark": int(moon.sum()),
        "n_cloudy_ok": int(cloudy_ok.sum()),
        "n_keep": int(keep.size),
        "min_cloudy_frac": lo,
        "max_cloudy_frac": hi,
        "cloudy_tile_thresh": float(cloudy_thresh),
        "keep_cloudy_frac_mean": (
            float(kept_frac.mean()) if kept_frac.size else float("nan")
        ),
        "keep_cloudy_frac_median": (
            float(np.median(kept_frac)) if kept_frac.size else float("nan")
        ),
        "moon_illum_max": MOON_ILLUM_MAX,
        "moon_alt_min_deg": MOON_ALT_MIN_DEG,
        "camera_product": json.loads((cache / "meta.json").read_text()).get(
            "camera_product"
        ),
        "sky_mean": float(sky.mean),
        "sky_std": float(sky.std),
        "n_hit_ok": n if hit_ok is None else int(hit_ok.sum()),
    }
    extra = ""
    if mean_t is not None:
        info["min_mean_t"] = float("-inf") if mean_t_min is None else float(mean_t_min)
        info["max_mean_t"] = float("inf") if mean_t_max is None else float(mean_t_max)
        info["n_mean_t_ok"] = int(t_ok.sum()) if t_ok is not None else 0
        info["keep_mean_t_mean"] = (
            float(kept_t.mean()) if kept_t is not None and kept_t.size else float("nan")
        )
        extra = (
            f"  meanT {info['min_mean_t']:g}–{info['max_mean_t']:g}"
            f"={info['n_mean_t_ok']}  keep_meanT={info['keep_mean_t_mean']:.3f}"
        )
    print(
        f"test keep  n={keep.size}/{n}  moon_dark={int(moon.sum())}  "
        f"cloudy {lo:g}–{hi:g} (T≤{cloudy_thresh:g})={int(cloudy_ok.sum())}  "
        f"mean_frac={info['keep_cloudy_frac_mean']:.3f}  "
        f"product={info['camera_product']}{extra}",
        flush=True,
    )
    return keep, info


_WORKER: dict = {}


def _init_worker(
    cache: str,
    names: tuple[str, ...],
    persist_hard_dt: float = PERSIST_HARD_DT,
) -> None:
    device = resolve_device("cpu")
    ds = PairsDataset.from_memmap(Path(cache), device)
    _WORKER["persist_hard_dt"] = float(persist_hard_dt)
    _WORKER["arr"] = ds._ensure_arrays()
    times = np.load(Path(cache) / "times.npz")
    _WORKER["unix"] = np.asarray(times["unix"], dtype=np.float64)
    _WORKER["sky_norm"] = ds.norm.sky
    _WORKER["nside"] = nside_from_npix(int(_WORKER["arr"]["y"].shape[1]))
    alt, _ = local_pix_altaz(_WORKER["nside"])
    _WORKER["sin_alt"] = np.sin(np.deg2rad(alt)).clip(min=0.0).astype(np.float32)
    _WORKER["names"] = tuple(names)
    if "dense_optical_flow" in names:
        _WORKER["of"] = DenseOpticalFlow()
    if "frozen_flow_crosscorr" in names:
        _WORKER["cc"] = FrozenFlowCrossCorr()
    if "lagrangian_wind_advection" in names:
        _WORKER["wind"] = LagrangianWindAdvection()


def _smooth_np(t: np.ndarray, valid: np.ndarray) -> np.ndarray:
    tt = torch.from_numpy(np.ascontiguousarray(t))
    vv = torch.from_numpy(np.ascontiguousarray(valid))
    return gaussian_smooth_healpix(tt, vv, sigma=SIGMA).numpy()


def _predict_eq(model, sky, jd, nobs, dt_ahead, last) -> np.ndarray:
    pred = np.asarray(
        model.predict(
            sky,
            jd,
            nobs,
            dt_ahead_days=dt_ahead,
            horizon_sec=float(HORIZON_SEC),
            min_nobs=1,
        ),
        dtype=np.float32,
    )
    miss = ~np.isfinite(pred)
    pred[miss] = last[miss]
    return pred


def _score_chunk(indices: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    arr = _WORKER["arr"]
    unix = _WORKER["unix"]
    sky_norm = _WORKER["sky_norm"]
    nside = int(_WORKER["nside"])
    sin_alt = _WORKER["sin_alt"]
    names = _WORKER["names"]
    dt_ahead = HORIZON_SEC / 86400.0
    idx = np.asarray(indices, dtype=np.int64)

    t_last_eq = np.asarray(sky_norm.decode(np.asarray(arr["last"][idx])), dtype=np.float32)
    t_y_eq = np.asarray(sky_norm.decode(np.asarray(arr["y"][idx])), dtype=np.float32)
    ym_eq = np.asarray(arr["y_mask"][idx]) > 0
    lm_eq = np.asarray(arr["last_mask"][idx]) > 0
    unix_row = unix[idx]
    unix_last = unix_row[:, -1]
    unix_y = unix_last + float(LEAD_SEC)
    sky = np.asarray(sky_norm.decode(np.asarray(arr["sky"][idx])), dtype=np.float64)
    sm = np.asarray(arr["sky_mask"][idx])
    sky = np.where(sm > 0, sky, np.nan)
    sky[:, -1] = np.where(lm_eq, t_last_eq, np.nan)
    nobs = sm.astype(np.float64)
    nobs[:, -1] = lm_eq.astype(np.float64)
    jd = unix_row / 86400.0 + JD_UNIX_EPOCH

    need_motion = [n for n in names if n != "eulerian_persistence"]
    models = {
        "dense_optical_flow": _WORKER.get("of"),
        "frozen_flow_crosscorr": _WORKER.get("cc"),
        "lagrangian_wind_advection": _WORKER.get("wind"),
    }
    eq: dict[str, np.ndarray] = {name: np.empty_like(t_last_eq) for name in need_motion}
    for k in range(int(idx.size)):
        for name in need_motion:
            eq[name][k] = _predict_eq(
                models[name],
                sky[k],
                jd[k],
                nobs[k],
                dt_ahead,
                t_last_eq[k],
            )

    t_last = lock_numpy_batch(t_last_eq, unix_last, nside)
    t_y = lock_numpy_batch(t_y_eq, unix_y, nside)
    ym = lock_numpy_batch(ym_eq, unix_y, nside)
    lm = lock_numpy_batch(lm_eq, unix_last, nside)
    t_last_s = _smooth_np(t_last, lm)
    t_y_s = _smooth_np(t_y, ym)
    preds = {}
    if "eulerian_persistence" in names:
        preds["eulerian_persistence"] = t_last_s
    for name in need_motion:
        locked = lock_numpy_batch(eq[name], unix_y, nside)
        preds[name] = _smooth_np(locked, lm)
    sa = np.broadcast_to(sin_alt, t_y_s.shape)
    slices = regime_masks(
        t_last_s,
        t_y_s,
        support_mask(ym, lm, sa),
        persist_hard_dt=float(_WORKER.get("persist_hard_dt", PERSIST_HARD_DT)),
    )
    out: dict[str, dict] = {}
    for name, pred in preds.items():
        deff = _empty_deff()
        accumulate_deff(deff, pred, t_y_s, ym, lm, sa)
        regimes = {r: _empty_regime_acc() for r in REGIME_NAMES}
        for rname, mask in slices.items():
            _update_regime(regimes[rname], pred, t_y_s, mask, sa, lm)
        out[name] = {
            "paper": paper_window_metrics(pred, t_y_s, ym, sa),
            "plain": per_window_plain_mse(pred, t_y_s, ym),
            "mask": mask_counts(pred, t_y_s, ym, sa, lm),
            "deff": deff,
            "regimes": regimes,
        }
    return out


def score_classical(
    cache: Path,
    indices: np.ndarray,
    *,
    workers: int,
    chunk: int,
    names: tuple[str, ...] = CLASSICAL_NAMES,
    persist_hard_dt: float = PERSIST_HARD_DT,
) -> dict:
    names = tuple(names)
    print(
        f"classical  {','.join(names)}  n={indices.size}  workers={workers}",
        flush=True,
    )
    acc = {
        name: {
            "paper": {k: [] for k in ("yield_mse", "maye", "dm5", "dm5_mae", "deff_mae")},
            "plain": [],
            "counts": {"tp": 0, "fn": 0, "fp": 0, "n_gold_cloud": 0, "n_conf": 0},
            "deff": _empty_deff(),
            "regimes": {r: _empty_regime_acc() for r in REGIME_NAMES},
        }
        for name in names
    }
    chunks = [indices[i : i + chunk] for i in range(0, indices.size, chunk)]
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=int(workers),
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(str(cache), names, float(persist_hard_dt)),
    ) as ex:
        futs = [ex.submit(_score_chunk, c) for c in chunks]
        done = 0
        for fut in as_completed(futs):
            part = fut.result()
            for name, rec in part.items():
                dst = acc[name]
                for k, arr in rec["paper"].items():
                    dst["paper"][k].append(arr)
                dst["plain"].append(rec["plain"])
                for k in dst["counts"]:
                    dst["counts"][k] += int(rec["mask"][k])
                for k in dst["deff"]:
                    dst["deff"][k] += rec["deff"][k]
                for rname in REGIME_NAMES:
                    _merge_regime(dst["regimes"][rname], rec["regimes"][rname])
            done += 1
            if done % 8 == 0 or done == len(futs):
                print(f"  chunks {done}/{len(futs)}", flush=True)
    rows = {}
    for name, rec in acc.items():
        cat = {k: np.concatenate(v) for k, v in rec["paper"].items()}
        row = add_mask_fields(
            summarize_windows(cat["yield_mse"], np.concatenate(rec["plain"])),
            rec["counts"],
        )
        add_paper_fields(row, cat, finalize_deff(rec["deff"]))
        row["regimes"] = {
            r: _finalize_regime(rec["regimes"][r]) for r in REGIME_NAMES
        }
        rows[name] = row
        print(
            f"  {name:28s}  MAYE {row['maye']:.6f}  yield_mse {row['yield_mse']:.6f}  "
            f"false-clear {row['false_clear']:.3f}  "
            f"Δm5 {row['dm5']:+.4f}  D_eff MAE {row['deff_mae_m']:.3f} m  "
            f"(T≤{MASK_CLOUD_T:g})",
            flush=True,
        )
        for rname in REGIME_NAMES:
            rr = row["regimes"][rname]
            print(
                f"    {rname:14s}  n_win={rr.get('n_windows', 0):5d}  "
                f"tiles={rr.get('n_tiles', 0):9d}  "
                f"MAYE {rr.get('maye', float('nan')):.4f}  "
                f"mse {rr.get('yield_mse', float('nan')):.5f}  "
                f"η_FC {rr.get('false_clear', float('nan')):.3f}",
                flush=True,
            )
    return rows


HISTORY_FRAME_KEYS = (
    "sky",
    "sky_mask",
    "sat",
    "meta",
    "wind_set",
    "wind_mask",
    "sat_age_sec",
)


def history_keep_index(n_frames: int, n_keep: int, *, mode: str) -> np.ndarray:
    """Frame indices to keep. ``linspace`` matches 8-frame pack / T=24 resample."""
    t = int(n_frames)
    k = int(n_keep)
    if k <= 0 or t <= 0:
        raise ValueError(f"n_frames={t} n_keep={k}")
    k = min(k, t)
    if mode == "last":
        keep = np.arange(t - k, t, dtype=np.int64)
    elif mode == "linspace":
        keep = np.asarray(_subsample_idx(t, k), dtype=np.int64)
        keep[-1] = t - 1
    else:
        raise ValueError(f"unknown history keep mode {mode!r}")
    return np.unique(keep)


def mask_history_keep(batch: dict, keep_idx: np.ndarray) -> dict:
    """Zero input frames whose index is not in ``keep_idx``. Labels unchanged."""
    sky = batch.get("sky")
    if sky is None or sky.ndim < 2:
        return batch
    t = int(sky.shape[1])
    keep = np.asarray(keep_idx, dtype=np.int64)
    drop = np.ones(t, dtype=bool)
    drop[keep] = False
    if not drop.any():
        return batch
    out = dict(batch)
    drop_t = torch.as_tensor(drop, device=sky.device)
    for key in HISTORY_FRAME_KEYS:
        x = out.get(key)
        if x is None or not torch.is_tensor(x) or x.ndim < 2 or int(x.shape[1]) != t:
            continue
        y = x.clone()
        y[:, drop_t] = 0
        out[key] = y
    return out


def mask_older_history(batch: dict, n_keep: int) -> dict:
    """Zero all but the last ``n_keep`` input frames. Last frame / labels unchanged."""
    sky = batch.get("sky")
    if sky is None or sky.ndim < 2:
        return batch
    keep = history_keep_index(int(sky.shape[1]), int(n_keep), mode="last")
    return mask_history_keep(batch, keep)


@torch.no_grad()
def score_model_loader(
    model,
    loader,
    sky_norm,
    device,
    *,
    name: str,
    persist_hard_dt: float = PERSIST_HARD_DT,
    keep_last_frames: int | None = None,
    keep_linspace_frames: int | None = None,
) -> dict:
    model.eval()
    paper: dict[str, list[np.ndarray]] = {
        k: [] for k in ("yield_mse", "maye", "dm5", "dm5_mae", "deff_mae")
    }
    plains = []
    persist_paper: dict[str, list[np.ndarray]] = {k: [] for k in paper}
    persist_plains = []
    counts = {"tp": 0, "fn": 0, "fp": 0, "n_gold_cloud": 0, "n_conf": 0}
    persist_c = {"tp": 0, "fn": 0, "fp": 0, "n_gold_cloud": 0, "n_conf": 0}
    deff = _empty_deff()
    persist_deff = _empty_deff()
    regimes = {r: _empty_regime_acc() for r in REGIME_NAMES}
    persist_regimes = {r: _empty_regime_acc() for r in REGIME_NAMES}
    if keep_last_frames is not None and keep_linspace_frames is not None:
        raise ValueError("use only one of keep_last_frames, keep_linspace_frames")
    keep_idx = None
    if keep_last_frames is not None:
        print(f"  {name} keep_last_frames={int(keep_last_frames)}", flush=True)
    if keep_linspace_frames is not None:
        print(f"  {name} keep_linspace_frames={int(keep_linspace_frames)}", flush=True)
    for raw in loader:
        b = apply_batch_preprocess(model, _to_device(raw, device))
        if keep_last_frames is not None or keep_linspace_frames is not None:
            t = int(b["sky"].shape[1])
            if keep_idx is None:
                if keep_linspace_frames is not None:
                    keep_idx = history_keep_index(t, int(keep_linspace_frames), mode="linspace")
                else:
                    keep_idx = history_keep_index(t, int(keep_last_frames), mode="last")
                print(f"  {name} keep_idx={keep_idx.tolist()}", flush=True)
            b = mask_history_keep(b, keep_idx)
        t_y, valid = decode_camera_t(
            b["y"], b["y_mask"], sky_norm, sigma=SIGMA
        )
        t_last, last_valid = decode_camera_t(
            b["last"], b["last_mask"], sky_norm, sigma=SIGMA
        )
        out = model_logits(model, b, sky_norm, 0.9)
        sin_alt = getattr(model, "last_sin_alt", None)
        if sin_alt is None:
            alt, _ = local_pix_altaz(nside_from_npix(int(t_y.shape[-1])))
            sin_alt = torch.sin(
                torch.deg2rad(torch.as_tensor(alt, device=t_y.device, dtype=t_y.dtype))
            ).clamp(min=0.0)
            if sin_alt.ndim == 1:
                sin_alt = sin_alt.unsqueeze(0).expand_as(t_y)
        if sin_alt.shape != t_y.shape:
            sin_alt = sin_alt.reshape(t_y.shape[0], -1)
            if sin_alt.shape[-1] == 1:
                sin_alt = sin_alt.expand_as(t_y)
        ym = (valid > 0).cpu().numpy()
        lm = (last_valid > 0).cpu().numpy()
        ty = t_y.detach().cpu().numpy()
        th = out.detach().cpu().numpy()
        tl = t_last.detach().cpu().numpy()
        sa = sin_alt.detach().cpu().numpy()
        if sa.ndim == 1:
            sa = np.broadcast_to(sa, ty.shape)
        pm = paper_window_metrics(th, ty, ym, sa)
        pp = paper_window_metrics(tl, ty, ym, sa)
        for k, arr in pm.items():
            paper[k].append(arr)
        for k, arr in pp.items():
            persist_paper[k].append(arr)
        plains.append(per_window_plain_mse(th, ty, ym))
        persist_plains.append(per_window_plain_mse(tl, ty, ym))
        accumulate_deff(deff, th, ty, ym, lm, sa)
        accumulate_deff(persist_deff, tl, ty, ym, lm, sa)
        for dst, pred in ((counts, th), (persist_c, tl)):
            part = mask_counts(pred, ty, ym, sa, lm)
            for k in dst:
                dst[k] += int(part[k])
        slices = regime_masks(tl, ty, support_mask(ym, lm, sa), persist_hard_dt=persist_hard_dt)
        for rname, mask in slices.items():
            _update_regime(regimes[rname], th, ty, mask, sa, lm)
            _update_regime(persist_regimes[rname], tl, ty, mask, sa, lm)
    cat = {k: np.concatenate(v) for k, v in paper.items()}
    pcat = {k: np.concatenate(v) for k, v in persist_paper.items()}
    row = add_mask_fields(
        summarize_windows(cat["yield_mse"], np.concatenate(plains)), counts
    )
    add_paper_fields(row, cat, finalize_deff(deff))
    prow = add_mask_fields(
        summarize_windows(pcat["yield_mse"], np.concatenate(persist_plains)),
        persist_c,
    )
    add_paper_fields(prow, pcat, finalize_deff(persist_deff))
    row["regimes"] = {r: _finalize_regime(regimes[r]) for r in REGIME_NAMES}
    prow["regimes"] = {r: _finalize_regime(persist_regimes[r]) for r in REGIME_NAMES}
    print(
        f"  {name:28s}  MAYE {row['maye']:.6f}  yield_mse {row['yield_mse']:.6f}  "
        f"false-clear {row['false_clear']:.3f}  "
        f"Δm5 {row['dm5']:+.4f}  D_eff MAE {row['deff_mae_m']:.3f} m  "
        f"(persist MAYE {prow['maye']:.6f}  fc {prow['false_clear']:.3f})",
        flush=True,
    )
    for rname in REGIME_NAMES:
        rr = row["regimes"][rname]
        pr = prow["regimes"][rname]
        print(
            f"    {rname:14s}  n_win={rr.get('n_windows', 0):5d}  "
            f"tiles={rr.get('n_tiles', 0):9d}  "
            f"MAYE {rr.get('maye', float('nan')):.4f}  "
            f"mse {rr.get('yield_mse', float('nan')):.5f}  "
            f"η_FC {rr.get('false_clear', float('nan')):.3f}  "
            f"(persist MAYE {pr.get('maye', float('nan')):.4f}  "
            f"fc {pr.get('false_clear', float('nan')):.3f})",
            flush=True,
        )
    row["persist_check"] = prow
    return row


def score_camera_nowcast(
    cache_root: Path,
    indices: np.ndarray,
    device,
    *,
    ckpt: Path,
    name: str,
    persist_hard_dt: float = PERSIST_HARD_DT,
    batch_size: int = 64,
) -> dict:
    print(f"{name}  ckpt {ckpt}  n={indices.size}", flush=True)
    test = PairsDataset.from_memmap(cache_root / "test", device)
    slim_camera_pairs(test)
    test = attach_horizon_lock(test, cache_root / "test")
    test = Subset(test, indices.tolist())
    cfg = TrainConfig(batch_size=int(batch_size), num_workers=4, seed=42, verbose=False)
    loader = make_loader(test, cfg, shuffle=False, device=device)
    model = nowcast_from_checkpoint(ckpt, device)
    test_meta = json.loads((cache_root / "test" / "meta.json").read_text())
    cache_n_frames = int(test_meta.get("n_frames", 8))
    if cache_n_frames != int(model.n_frames):
        raise SystemExit(
            f"{name}: test cache n_frames={cache_n_frames} != model "
            f"n_frames={int(model.n_frames)}; pass matching --cache-dir"
        )
    print(
        f"  {name} n_frames={int(model.n_frames)}  "
        f"output_transmission={bool(getattr(model, 'output_transmission', False))}",
        flush=True,
    )
    sky_norm = PairsDataset.from_memmap(cache_root / "test", resolve_device("cpu")).norm.sky
    return score_model_loader(
        model, loader, sky_norm, device, name=name, persist_hard_dt=persist_hard_dt
    )


def score_simvp(
    cache_root: Path,
    indices: np.ndarray,
    device,
    *,
    ckpt: Path = SIMVP_CKPT,
    persist_hard_dt: float = PERSIST_HARD_DT,
) -> dict:
    return score_camera_nowcast(
        cache_root,
        indices,
        device,
        ckpt=ckpt,
        name="simvp",
        persist_hard_dt=persist_hard_dt,
    )


def score_proposed(
    cache_root: Path,
    pairs_dir: Path,
    indices: np.ndarray,
    device,
    *,
    ckpt: Path = PROPOSED_CKPT,
    name: str = "proposed",
    ablate: str = "",
    persist_hard_dt: float = PERSIST_HARD_DT,
    n_frames: int | None = None,
    lead_sec: float | None = None,
) -> dict:
    print(f"{name}  ckpt {ckpt}  ablate={ablate or 'off'}  n={indices.size}", flush=True)
    cfg = TrainConfig(
        batch_size=16,
        num_workers=4,
        seed=42,
        verbose=True,
        d_model=64,
        n_heads=4,
        n_layers=2,
        sky_width=16,
    )
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    ckpt_cfg: dict = {}
    if isinstance(payload, dict):
        top_cfg = payload.get("config") or {}
        info_cfg = (payload.get("info") or {}).get("config") or {}
        if isinstance(top_cfg, dict):
            ckpt_cfg.update(top_cfg)
        if isinstance(info_cfg, dict):
            ckpt_cfg.update(info_cfg)
        if payload.get("n_frames") is not None:
            ckpt_cfg.setdefault("n_frames", payload["n_frames"])
        info = payload.get("info") or {}
        if isinstance(info, dict) and "n_frames" in info:
            ckpt_cfg.setdefault("n_frames", info["n_frames"])
    if n_frames is None:
        n_frames = int(ckpt_cfg.get("n_frames", 24))
    cfg.n_frames = int(n_frames)
    print(f"  {name} n_frames={cfg.n_frames}  last+gate unique-scan", flush=True)
    test_cache = cache_root / "test"
    test_data = PairsDataset.from_memmap(test_cache, device)
    cfg.wind_age_mean = float(test_data.norm.wind[-1].mean)
    cfg.wind_age_std = float(test_data.norm.wind[-1].std)
    attach_wind_decode_stats(cfg, test_data.norm)
    test_meta = json.loads((test_cache / "meta.json").read_text())
    nside = int(test_meta["nside"])
    cache_n_frames = int(test_meta.get("n_frames", cfg.n_frames))
    if cache_n_frames != cfg.n_frames:
        raise SystemExit(
            f"{name}: test cache n_frames={cache_n_frames} != model "
            f"n_frames={cfg.n_frames}; resample test or pass matching --cache-dir"
        )
    test_data = wrap_rich_sat(
        test_data,
        pairs_dir,
        "test",
        test_cache,
        n_frames=cfg.n_frames,
        sat_size=NATIVE_IR_HW,
    )
    lock_lead = float(LEAD_SEC if lead_sec is None else lead_sec)
    test_data = attach_horizon_lock(test_data, test_cache, lead_sec=lock_lead)
    test_data = Subset(test_data, indices.tolist())
    loader = make_loader(test_data, cfg, shuffle=False, device=device)
    model = forecaster.build_model(int(nside), cfg).to(device)
    model.set_ablate(ablate)
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
    if missing or unexpected:
        print(
            f"  {name} load_state_dict strict=False  "
            f"missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )
    sky_norm = PairsDataset.from_memmap(test_cache, resolve_device("cpu")).norm.sky
    _set_sky_norm(model, sky_norm)
    model.eval()
    return score_model_loader(
        model,
        loader,
        sky_norm,
        device,
        name=name,
        persist_hard_dt=persist_hard_dt,
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--pairs", type=Path, default=None)
    p.add_argument(
        "--out",
        type=Path,
        default=Path(
            str(MNT) + "/baselines_neural/"
            "suite_test_yield_mse_sys_holdout/suite_test_yield_mse_sys.json"
        ),
    )
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--chunk", type=int, default=128)
    p.add_argument("--skip-classical", action="store_true")
    p.add_argument(
        "--classical-names",
        nargs="+",
        default=None,
        help="Subset of classical models (default: all four).",
    )
    p.add_argument("--skip-simvp", action="store_true")
    p.add_argument("--skip-convlstm", action="store_true")
    p.add_argument("--skip-proposed", action="store_true")
    p.add_argument("--simvp-ckpt", type=Path, default=SIMVP_CKPT)
    p.add_argument(
        "--convlstm-ckpt",
        type=Path,
        default=None,
        help="Score ConvLSTM if set (camera-only nowcast, same loader as SimVP).",
    )
    p.add_argument("--proposed-ckpt", type=Path, default=PROPOSED_CKPT)
    p.add_argument(
        "--proposed-ablate",
        default="",
        help=(
            "Modalities to zero when scoring --proposed-ckpt. "
            "Same spec as train_multihorizon.py --ablate."
        ),
    )
    p.add_argument("--ablate-goes-ckpt", type=Path, default=None)
    p.add_argument("--ablate-dream-ckpt", type=Path, default=None)
    p.add_argument(
        "--n-frames",
        type=int,
        default=None,
        help=(
            "DREAM history length for model scoring (default: read from ckpt, "
            "else 24)."
        ),
    )
    p.add_argument("--paper-tex", type=Path, default=None)
    p.add_argument(
        "--cloudy-min",
        type=float,
        default=0.0,
        help="Keep windows whose gold cloudy fraction is ≥ this.",
    )
    p.add_argument(
        "--cloudy-max",
        type=float,
        default=MAX_CLOUDY_FRAC,
        help="Keep windows whose gold cloudy fraction is ≤ this.",
    )
    p.add_argument(
        "--cloudy-thresh",
        type=float,
        default=CLOUD_T_THRESH,
        help=(
            "Window-level overcast cut: drop windows whose fraction of "
            f"tiles with T≤this exceeds --cloudy-max (default {CLOUD_T_THRESH:g}, "
            "same as training). Incoming/photometric tiles still use T≤0.63."
        ),
    )
    p.add_argument(
        "--persist-hard-dt",
        type=float,
        default=PERSIST_HARD_DT,
        help="Persist-hard tiles have |T_gold−T_last| ≥ this (default 0.10).",
    )
    p.add_argument(
        "--mean-t-min",
        type=float,
        default=None,
        help="Keep windows whose mean observed transmission is ≥ this.",
    )
    p.add_argument(
        "--mean-t-max",
        type=float,
        default=None,
        help="Keep windows whose mean observed transmission is ≤ this.",
    )
    args = p.parse_args(argv)
    hard_dt = float(args.persist_hard_dt)

    cache_root = Path(args.cache_dir)
    test_cache = cache_root / "test"
    keep, keep_info = test_keep_indices(
        test_cache,
        cloudy_min=args.cloudy_min,
        cloudy_max=args.cloudy_max,
        cloudy_thresh=args.cloudy_thresh,
        mean_t_min=args.mean_t_min,
        mean_t_max=args.mean_t_max,
    )
    pairs_dir = resolve_pairs(args.pairs)
    device = resolve_device(None)
    print(f"  pairs {pairs_dir}  device {device}  sigma={SIGMA}", flush=True)

    models: dict[str, dict] = {}
    merge_prev = args.skip_classical or args.skip_simvp or args.skip_proposed or (
        args.skip_convlstm or args.classical_names is not None
    )
    if args.out.is_file() and merge_prev:
        prev = json.loads(args.out.read_text()).get("models", {})
        if isinstance(prev, dict):
            models.update(prev)
    if not args.skip_classical:
        names = tuple(args.classical_names) if args.classical_names else CLASSICAL_NAMES
        models.update(
            score_classical(
                test_cache,
                keep,
                workers=args.workers,
                chunk=args.chunk,
                names=names,
                persist_hard_dt=hard_dt,
            )
        )
    if not args.skip_convlstm and args.convlstm_ckpt is not None:
        models["convlstm"] = score_camera_nowcast(
            cache_root,
            keep,
            device,
            ckpt=Path(args.convlstm_ckpt),
            name="convlstm",
            persist_hard_dt=hard_dt,
        )
        if "eulerian_persistence" not in models:
            models["eulerian_persistence"] = models["convlstm"].get(
                "persist_check", {}
            )
    if not args.skip_simvp:
        models["simvp"] = score_simvp(
            cache_root,
            keep,
            device,
            ckpt=Path(args.simvp_ckpt),
            persist_hard_dt=hard_dt,
        )
        if "eulerian_persistence" not in models:
            models["eulerian_persistence"] = models["simvp"].get("persist_check", {})
    if not args.skip_proposed:
        models["proposed"] = score_proposed(
            cache_root,
            pairs_dir,
            keep,
            device,
            ckpt=Path(args.proposed_ckpt),
            name="proposed",
            ablate=str(args.proposed_ablate or ""),
            persist_hard_dt=hard_dt,
            n_frames=args.n_frames,
        )
        if "eulerian_persistence" not in models:
            models["eulerian_persistence"] = models["proposed"].get(
                "persist_check", {}
            )
    if args.ablate_goes_ckpt is not None:
        models["ablate_goes"] = score_proposed(
            cache_root,
            pairs_dir,
            keep,
            device,
            ckpt=Path(args.ablate_goes_ckpt),
            name="ablate_goes",
            ablate="sat,wind",
            persist_hard_dt=hard_dt,
            n_frames=args.n_frames,
        )
        if "eulerian_persistence" not in models:
            models["eulerian_persistence"] = models["ablate_goes"].get(
                "persist_check", {}
            )
    if args.ablate_dream_ckpt is not None:
        models["ablate_dream"] = score_proposed(
            cache_root,
            pairs_dir,
            keep,
            device,
            ckpt=Path(args.ablate_dream_ckpt),
            name="ablate_dream",
            ablate="dream",
            persist_hard_dt=hard_dt,
            n_frames=args.n_frames,
        )
        if "eulerian_persistence" not in models:
            models["eulerian_persistence"] = models["ablate_dream"].get(
                "persist_check", {}
            )

    report = {
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset": "dream_pairs_full_odd_holdout",
        "cache": str(test_cache),
        "gold": (
            f"horizon-locked T, Gaussian σ={SIGMA} on y and on every "
            "prediction (persist last, OF, frozen-flow CC, winds, model T̂); "
            "moon-dark; gold cloudy fraction "
            f"{keep_info['min_cloudy_frac']:g}–{keep_info['max_cloudy_frac']:g} "
            f"(tile T≤{keep_info['cloudy_tile_thresh']:g})"
        ),
        "horizon_sec": HORIZON_SEC,
        "label_sigma": SIGMA,
        "maye": (
            f"airmass-weighted MAE on T, sin(alt) weight, "
            f"alt≥{SCHEDULER_MIN_ALT_DEG:g}°"
        ),
        "yield_mse": (
            f"airmass-weighted MSE on T, sin(alt) weight, "
            f"alt≥{SCHEDULER_MIN_ALT_DEG:g}°"
        ),
        "false_clear": (
            f"FN/(TP+FN) on horizon-locked σ={SIGMA} T; "
            f"cloudy = T≤{MASK_CLOUD_T:g}; "
            f"tiles y_mask ∩ last_mask ∩ alt≥{SCHEDULER_MIN_ALT_DEG:g}°"
        ),
        "dm5": "1.25 log10(T̂/T), airmass-weighted; positive = overestimate of depth",
        "deff": f"8.4 m × √T; deff_mae_m is airmass-weighted |D_eff(T̂)−D_eff(T)|",
        "false_clear_thresh": MASK_CLOUD_T,
        "regimes": {
            "incoming": (
                f"last T>{MASK_CLOUD_T:g} and observed transmission≤{MASK_CLOUD_T:g} "
                f"on y∩last∩alt≥{SCHEDULER_MIN_ALT_DEG:g}°"
            ),
            "photometric": (
                f"last T>{MASK_CLOUD_T:g} and observed transmission>{MASK_CLOUD_T:g} "
                f"on y∩last∩alt≥{SCHEDULER_MIN_ALT_DEG:g}°"
            ),
            "persist_hard": (
                f"|T_gold−T_last|≥{hard_dt:g} "
                f"on y∩last∩alt≥{SCHEDULER_MIN_ALT_DEG:g}°"
            ),
            "persist_hard_dt": hard_dt,
        },
        "keep": keep_info,
        "models": models,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    persist_row = models.get("eulerian_persistence", {})
    persist = persist_row.get("yield_mse")
    persist_maye = persist_row.get("maye")
    order = (
        "eulerian_persistence",
        "convlstm",
        "simvp",
        "proposed",
        "ablate_goes",
        "ablate_dream",
    )
    names = [n for n in order if n in models] + [
        n for n in models if n not in order
    ]
    print(
        f"{'model':22s} {'MAYE':>9s} {'yield_mse':>10s} {'η_FC':>8s} "
        f"{'Δm5':>8s} {'D_eff MAE':>10s} {'D_eff pred':>10s}",
        flush=True,
    )
    for name in names:
        row = models[name]
        maye = row.get("maye")
        ym = row.get("yield_mse")
        fc = row.get("false_clear")
        dm = row.get("dm5")
        de = row.get("deff_mae_m")
        dp = row.get("deff_on_pred_clear_m")
        print(
            f"{name:22s} "
            f"{maye if maye is not None else float('nan'):9.6f} "
            f"{ym if ym is not None else float('nan'):10.6f} "
            f"{fc if fc is not None else float('nan'):8.3f} "
            f"{dm if dm is not None else float('nan'):+8.4f} "
            f"{de if de is not None else float('nan'):10.3f} "
            f"{dp if dp is not None else float('nan'):10.3f}",
            flush=True,
        )
    print(
        f"MAYE / yield MSE: airmass-weighted T, alt≥{SCHEDULER_MIN_ALT_DEG:g}°, "
        f"σ={SIGMA}.  η_FC = FN/(TP+FN) at T≤{MASK_CLOUD_T:g}.  "
        f"Δm5 = 1.25 log10(T̂/T).  D_eff = 8.4√T metres.",
        flush=True,
    )
    if persist_maye:
        print(f"persist MAYE = {persist_maye:.6f}  persist yield_mse = {persist}", flush=True)
    _print_regime_tables(models, names, persist_hard_dt=hard_dt)
    tex_path = args.paper_tex
    if tex_path is None:
        tex_path = args.out.with_suffix(".tex")
    _write_paper_tex(tex_path, models, names, persist_hard_dt=hard_dt)
    print(f"wrote {args.out}", flush=True)
    print(f"wrote {tex_path}", flush=True)


def _fmt(x, spec: str) -> str:
    if x is None:
        return "---"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "---"
    if not math.isfinite(v):
        return "---"
    return format(v, spec)


def _print_regime_tables(
    models: dict, names: list[str], *, persist_hard_dt: float = PERSIST_HARD_DT
) -> None:
    titles = {
        "incoming": "incoming  (last clear, gold cloudy)",
        "photometric": "photometric  (last clear, gold clear)",
        "persist_hard": f"persist-hard  (|ΔT|≥{persist_hard_dt:g})",
    }
    for rname in REGIME_NAMES:
        print(f"\n{titles.get(rname, rname)}", flush=True)
        print(
            f"{'model':22s} {'n_win':>6s} {'tiles':>10s} {'MAYE':>8s} "
            f"{'yield_mse':>10s} {'η_FC':>8s} {'Δm5':>8s} {'D_eff':>8s}",
            flush=True,
        )
        for name in names:
            rr = (models[name].get("regimes") or {}).get(rname) or {}
            print(
                f"{name:22s} "
                f"{int(rr.get('n_windows') or 0):6d} "
                f"{int(rr.get('n_tiles') or 0):10d} "
                f"{rr.get('maye', float('nan')):8.4f} "
                f"{rr.get('yield_mse', float('nan')):10.5f} "
                f"{rr.get('false_clear', float('nan')):8.3f} "
                f"{rr.get('dm5', float('nan')):+8.4f} "
                f"{rr.get('deff_mae_m', float('nan')):8.3f}",
                flush=True,
            )


def _write_paper_tex(
    path: Path,
    models: dict,
    names: list[str],
    *,
    persist_hard_dt: float = PERSIST_HARD_DT,
) -> None:
    lines = [
        r"% Auto-generated test-holdout table (MAYE, false-clear, $\Delta m_{5\sigma}$, $D_{\mathrm{eff}}$).",
        r"\begin{tabular}{lcccc}",
        r"\hline",
        r"Model & MAYE & $\eta_{\mathrm{FC}}$ & $\Delta m_{5\sigma}$ & $D_{\mathrm{eff}}$ MAE (m) \\",
        r"\hline",
    ]
    labels = {
        "eulerian_persistence": r"Eulerian persistence",
        "convlstm": r"ConvLSTM",
        "simvp": r"SimVP",
        "proposed": r"Proposed",
        "ablate_goes": r"Proposed, no GOES",
        "ablate_dream": r"Proposed, no DREAM",
    }
    for name in names:
        row = models[name]
        lab = labels.get(name, name.replace("_", r"\_"))
        lines.append(
            f"{lab} & {_fmt(row.get('maye'), '.4f')} & "
            f"{_fmt(row.get('false_clear'), '.3f')} & "
            f"{_fmt(row.get('dm5'), '+.4f')} & "
            f"{_fmt(row.get('deff_mae_m'), '.3f')} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", ""])
    captions = {
        "incoming": r"Incoming: last clear, gold cloudy ($T\le0.63$).",
        "photometric": r"Photometric: last and gold both clear.",
        "persist_hard": (
            r"Persist-hard: $|T_{\mathrm{gold}}-T_{\mathrm{last}}|\ge"
            + f"{persist_hard_dt:g}$."
        ),
    }
    for rname in REGIME_NAMES:
        lines.append(f"% {captions[rname]}")
        lines.extend(
            [
                r"\begin{tabular}{lcccc}",
                r"\hline",
                r"Model & MAYE & $\eta_{\mathrm{FC}}$ & $\Delta m_{5\sigma}$ & "
                r"$D_{\mathrm{eff}}$ MAE (m) \\",
                r"\hline",
            ]
        )
        for name in names:
            rr = (models[name].get("regimes") or {}).get(rname) or {}
            lab = labels.get(name, name.replace("_", r"\_"))
            lines.append(
                f"{lab} & {_fmt(rr.get('maye'), '.4f')} & "
                f"{_fmt(rr.get('false_clear'), '.3f')} & "
                f"{_fmt(rr.get('dm5'), '+.4f')} & "
                f"{_fmt(rr.get('deff_mae_m'), '.3f')} \\\\"
            )
        lines.extend([r"\hline", r"\end{tabular}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
