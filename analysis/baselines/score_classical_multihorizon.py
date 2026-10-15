#!/usr/bin/env python3
"""Holdout MAYE and false-clear rate for the classical baselines at every cached lead.

Same holdout as ``score_multihorizon``: moon-dark, cloudy fraction ≤ 1,
mean observed transmission ∈ [0.1, 0.9] from the +20 min ``y.npy``. The observed map at 90 s / 5 min /
20 min / 30 min is ``y_h`` on ``dream_multih_t24_encoded``. Windows with
``horizon_hit==0`` are dropped at that lead.

    python -u -m analysis.baselines.score_classical_multihorizon
    python -u -m analysis.baselines.score_classical_multihorizon --persist-only
"""

from __future__ import annotations

from analysis.paths import MNT

import argparse
import datetime as dt
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from analysis.baselines.lagrangian_winds import LagrangianWindAdvection
from analysis.baselines.optical_flow import DenseOpticalFlow
from analysis.baselines.score_test_yield_mse import (
    CLASSICAL_NAMES,
    MASK_CLOUD_T,
    PERSIST_HARD_DT,
    SIGMA,
    _empty_deff,
    accumulate_deff,
    add_mask_fields,
    add_paper_fields,
    finalize_deff,
    mask_counts,
    paper_window_metrics,
    per_window_plain_mse,
    summarize_windows,
    test_keep_indices,
)
from analysis.baselines.segev_crosscorr import FrozenFlowCrossCorr
from analysis.healpix_torch import gaussian_smooth_healpix
from analysis.forecaster.geom import local_pix_altaz, nside_from_npix
from analysis.forecaster.multihorizon import (
    HIT_NAME,
    HORIZON_LABELS,
    HORIZONS_SEC,
    Y_H_MASK_NAME,
    Y_H_NAME,
    load_spec,
)
from analysis.forecaster.port_prep import lock_numpy_batch
from analysis.our_model.model import JD_UNIX_EPOCH, PairsDataset
from analysis.torch_util import resolve_device

DEFAULT_CACHE = Path(str(MNT) + "/pairs/dream_multih_t24_encoded")
DEFAULT_OUT = Path(
    str(MNT) + "/baselines_neural/"
    "classical_multih_t24/suite_test_mean_t_0.1_0.9_paperkeep_classical_multih.json"
)
MOTION_NAMES = (
    "dense_optical_flow",
    "frozen_flow_crosscorr",
    "lagrangian_wind_advection",
)
_WORKER: dict = {}


def _lab(horizon_sec: int) -> str:
    return HORIZON_LABELS.get(int(horizon_sec), f"{int(horizon_sec)}s")


def _smooth_np(t: np.ndarray, valid: np.ndarray) -> np.ndarray:
    import torch

    tt = torch.from_numpy(np.ascontiguousarray(t))
    vv = torch.from_numpy(np.ascontiguousarray(valid))
    return gaussian_smooth_healpix(tt, vv, sigma=SIGMA).numpy()


def _predict_eq(model, sky, jd, nobs, horizon_sec, last) -> np.ndarray:
    pred = np.asarray(
        model.predict(
            sky,
            jd,
            nobs,
            dt_ahead_days=float(horizon_sec) / 86400.0,
            horizon_sec=float(horizon_sec),
            min_nobs=1,
        ),
        dtype=np.float32,
    )
    miss = ~np.isfinite(pred)
    pred[miss] = last[miss]
    return pred


def _init_worker(
    cache: str,
    names: tuple[str, ...],
    leads: tuple[str, ...] | None = None,
) -> None:
    import torch

    torch.set_num_threads(1)
    device = resolve_device("cpu")
    dest = Path(cache)
    ds = PairsDataset.from_memmap(dest, device)
    spec = load_spec(dest)
    shapes = {k: tuple(v) for k, v in spec["shapes"].items()}
    _WORKER["arr"] = ds._ensure_arrays()
    times = np.load(dest / "times.npz")
    _WORKER["unix"] = np.asarray(times["unix"], dtype=np.float64)
    _WORKER["sky_norm"] = ds.norm.sky
    _WORKER["nside"] = nside_from_npix(int(_WORKER["arr"]["y"].shape[1]))
    alt, _ = local_pix_altaz(_WORKER["nside"])
    _WORKER["sin_alt"] = np.sin(np.deg2rad(alt)).clip(min=0.0).astype(np.float32)
    _WORKER["names"] = tuple(names)
    _WORKER["keep_leads"] = None if leads is None else tuple(leads)
    _WORKER["horizons"] = tuple(int(h) for h in spec["horizons_sec"])
    _WORKER["y_h"] = np.memmap(dest / Y_H_NAME, mode="r", dtype=np.float32, shape=shapes["y_h"])
    _WORKER["y_h_mask"] = np.memmap(
        dest / Y_H_MASK_NAME, mode="r", dtype=np.float32, shape=shapes["y_h_mask"]
    )
    _WORKER["hit"] = np.memmap(dest / HIT_NAME, mode="r", dtype=np.float32, shape=shapes["horizon_hit"])
    if "dense_optical_flow" in names:
        _WORKER["of"] = DenseOpticalFlow()
    if "frozen_flow_crosscorr" in names:
        _WORKER["cc"] = FrozenFlowCrossCorr()
    if "lagrangian_wind_advection" in names:
        _WORKER["wind"] = LagrangianWindAdvection()


def _score_chunk(indices: np.ndarray) -> dict[str, dict[str, dict]]:
    arr = _WORKER["arr"]
    unix = _WORKER["unix"]
    sky_norm = _WORKER["sky_norm"]
    nside = int(_WORKER["nside"])
    sin_alt = _WORKER["sin_alt"]
    names = _WORKER["names"]
    horizons = _WORKER["horizons"]
    idx = np.asarray(indices, dtype=np.int64)

    t_last_eq = np.asarray(sky_norm.decode(np.asarray(arr["last"][idx])), dtype=np.float32)
    lm_eq = np.asarray(arr["last_mask"][idx]) > 0
    unix_row = unix[idx]
    unix_last = unix_row[:, -1]
    sky = np.asarray(sky_norm.decode(np.asarray(arr["sky"][idx])), dtype=np.float64)
    sm = np.asarray(arr["sky_mask"][idx])
    sky = np.where(sm > 0, sky, np.nan)
    sky[:, -1] = np.where(lm_eq, t_last_eq, np.nan)
    nobs = sm.astype(np.float64)
    nobs[:, -1] = lm_eq.astype(np.float64)
    jd = unix_row / 86400.0 + JD_UNIX_EPOCH

    t_last = lock_numpy_batch(t_last_eq, unix_last, nside)
    lm = lock_numpy_batch(lm_eq, unix_last, nside)
    t_last_s = _smooth_np(t_last, lm)
    sa = np.broadcast_to(sin_alt, t_last_s.shape)

    models = {
        "dense_optical_flow": _WORKER.get("of"),
        "frozen_flow_crosscorr": _WORKER.get("cc"),
        "lagrangian_wind_advection": _WORKER.get("wind"),
    }
    need_motion = [n for n in names if n in models and models[n] is not None]
    keep_leads = _WORKER.get("keep_leads")
    out: dict[str, dict[str, dict]] = {}
    for k, horizon_sec in enumerate(horizons):
        if keep_leads is not None and _lab(horizon_sec) not in keep_leads:
            continue
        t_y_eq = np.asarray(sky_norm.decode(np.asarray(_WORKER["y_h"][idx, k])), dtype=np.float32)
        ym_eq = np.asarray(_WORKER["y_h_mask"][idx, k]) > 0
        hit = np.asarray(_WORKER["hit"][idx, k]) > 0
        ym_eq = ym_eq & hit[:, None]
        unix_y = unix_last + float(horizon_sec)
        t_y = lock_numpy_batch(t_y_eq, unix_y, nside)
        ym = lock_numpy_batch(ym_eq, unix_y, nside)
        t_y_s = _smooth_np(t_y, ym)
        preds = {}
        if "eulerian_persistence" in names:
            preds["eulerian_persistence"] = t_last_s
        for name in need_motion:
            eq = np.empty_like(t_last_eq)
            for i in range(int(idx.size)):
                eq[i] = _predict_eq(
                    models[name],
                    sky[i],
                    jd[i],
                    nobs[i],
                    horizon_sec,
                    t_last_eq[i],
                )
            locked = lock_numpy_batch(eq, unix_y, nside)
            preds[name] = _smooth_np(locked, lm)
        recs = {}
        for name, pred in preds.items():
            deff = _empty_deff()
            accumulate_deff(deff, pred, t_y_s, ym, lm, sa)
            recs[name] = {
                "paper": paper_window_metrics(pred, t_y_s, ym, sa),
                "plain": per_window_plain_mse(pred, t_y_s, ym),
                "mask": mask_counts(pred, t_y_s, ym, sa, lm),
                "deff": deff,
            }
        out[_lab(horizon_sec)] = recs
    return out


def _window_fc(
    t_hat: np.ndarray,
    t_true: np.ndarray,
    valid: np.ndarray,
    sin_alt: np.ndarray,
    last_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-window FN, cloudy tiles, support, and tiles called clear."""
    from analysis.baselines.score_test_yield_mse import MIN_SIN

    conf = np.asarray(valid, dtype=bool) & np.asarray(last_valid, dtype=bool)
    sa = np.asarray(sin_alt)
    if sa.ndim == 1:
        conf = conf & (sa >= MIN_SIN)
    else:
        conf = conf & (sa >= MIN_SIN)
    gold = (np.clip(t_true, 0.0, 1.0) <= MASK_CLOUD_T) & conf
    called_clear = conf & ~(np.clip(t_hat, 0.0, 1.0) <= MASK_CLOUD_T)
    fn = np.count_nonzero(called_clear & gold, axis=-1).astype(np.int32)
    n_gold = np.count_nonzero(gold, axis=-1).astype(np.int32)
    n_conf = np.count_nonzero(conf, axis=-1).astype(np.int32)
    n_clear = np.count_nonzero(called_clear, axis=-1).astype(np.int32)
    return fn, n_gold, n_conf, n_clear


def _frozen_fc_chunk(payload: tuple[int, np.ndarray]) -> tuple[int, dict[str, dict[str, np.ndarray]]]:
    """Per-window cloud-failure counts for frozen-flow cross-correlation."""
    start, indices = payload
    arr = _WORKER["arr"]
    unix = _WORKER["unix"]
    sky_norm = _WORKER["sky_norm"]
    nside = int(_WORKER["nside"])
    sin_alt = _WORKER["sin_alt"]
    horizons = _WORKER["horizons"]
    idx = np.asarray(indices, dtype=np.int64)
    model = _WORKER["cc"]

    t_last_eq = np.asarray(sky_norm.decode(np.asarray(arr["last"][idx])), dtype=np.float32)
    lm_eq = np.asarray(arr["last_mask"][idx]) > 0
    unix_row = unix[idx]
    unix_last = unix_row[:, -1]
    sky = np.asarray(sky_norm.decode(np.asarray(arr["sky"][idx])), dtype=np.float64)
    sm = np.asarray(arr["sky_mask"][idx])
    sky = np.where(sm > 0, sky, np.nan)
    sky[:, -1] = np.where(lm_eq, t_last_eq, np.nan)
    nobs = sm.astype(np.float64)
    nobs[:, -1] = lm_eq.astype(np.float64)
    jd = unix_row / 86400.0 + JD_UNIX_EPOCH

    t_last = lock_numpy_batch(t_last_eq, unix_last, nside)
    lm = lock_numpy_batch(lm_eq, unix_last, nside)
    sa = np.broadcast_to(sin_alt, t_last.shape)
    out: dict[str, dict[str, np.ndarray]] = {}
    for k, horizon_sec in enumerate(horizons):
        lab = _lab(horizon_sec)
        if _WORKER.get("keep_leads") is not None and lab not in _WORKER["keep_leads"]:
            continue
        t_y_eq = np.asarray(sky_norm.decode(np.asarray(_WORKER["y_h"][idx, k])), dtype=np.float32)
        ym_eq = np.asarray(_WORKER["y_h_mask"][idx, k]) > 0
        hit = np.asarray(_WORKER["hit"][idx, k]) > 0
        ym_eq = ym_eq & hit[:, None]
        unix_y = unix_last + float(horizon_sec)
        t_y = lock_numpy_batch(t_y_eq, unix_y, nside)
        ym = lock_numpy_batch(ym_eq, unix_y, nside)
        t_y_s = _smooth_np(t_y, ym)
        eq = np.empty_like(t_last_eq)
        for i in range(int(idx.size)):
            eq[i] = _predict_eq(model, sky[i], jd[i], nobs[i], horizon_sec, t_last_eq[i])
        pred = _smooth_np(lock_numpy_batch(eq, unix_y, nside), lm)
        fn, n_gold, n_conf, n_clear = _window_fc(pred, t_y_s, ym, sa, lm)
        out[lab] = {"fn": fn, "n_gold": n_gold, "n_conf": n_conf, "n_clear": n_clear}
    return int(start), out


def score_frozen_windows(
    cache: Path,
    indices: np.ndarray,
    *,
    workers: int,
    chunk: int,
    leads: tuple[str, ...],
) -> dict[str, dict[str, np.ndarray]]:
    """Per-window frozen-flow counts in ``indices`` order."""
    indices = np.asarray(indices, dtype=np.int64)
    n = int(indices.size)
    acc = {
        lab: {key: np.zeros(n, dtype=np.int32) for key in ("fn", "n_gold", "n_conf", "n_clear")}
        for lab in leads
    }
    chunks = [(i, indices[i : i + chunk]) for i in range(0, n, chunk)]
    print(
        f"frozen-flow cross-corr  n={n}  workers={workers}  chunks={len(chunks)}",
        flush=True,
    )
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=int(workers),
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(str(cache), ("frozen_flow_crosscorr",), tuple(leads)),
    ) as ex:
        futs = [ex.submit(_frozen_fc_chunk, c) for c in chunks]
        done = 0
        for fut in as_completed(futs):
            start, part = fut.result()
            for lab, rec in part.items():
                stop = start + int(rec["fn"].shape[0])
                for key, arr in rec.items():
                    acc[lab][key][start:stop] = arr
            done += 1
            if done % 8 == 0 or done == len(futs):
                print(f"  chunks {done}/{len(futs)}", flush=True)
    return acc


def _empty_acc(names: tuple[str, ...]) -> dict:
    return {
        name: {
            "paper": {k: [] for k in ("yield_mse", "maye", "dm5", "dm5_mae", "deff_mae")},
            "plain": [],
            "counts": {"tp": 0, "fn": 0, "fp": 0, "n_gold_cloud": 0, "n_conf": 0},
            "deff": _empty_deff(),
        }
        for name in names
    }


def _finalize(acc: dict) -> dict:
    cat = {k: np.concatenate(v) for k, v in acc["paper"].items()}
    row = add_mask_fields(
        summarize_windows(cat["yield_mse"], np.concatenate(acc["plain"])),
        acc["counts"],
    )
    add_paper_fields(row, cat, finalize_deff(acc["deff"]))
    return row


def score_classical_horizons(
    cache: Path,
    indices: np.ndarray,
    *,
    workers: int,
    chunk: int,
    names: tuple[str, ...],
) -> dict:
    names = tuple(names)
    spec = load_spec(cache)
    horizons = tuple(int(h) for h in spec["horizons_sec"])
    print(
        f"classical multihorizon  {','.join(names)}  n={indices.size}  "
        f"leads={list(horizons)}  workers={workers}",
        flush=True,
    )
    acc = {_lab(h): _empty_acc(names) for h in horizons}
    chunks = [indices[i : i + chunk] for i in range(0, indices.size, chunk)]
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=int(workers),
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(str(cache), names),
    ) as ex:
        futs = [ex.submit(_score_chunk, c) for c in chunks]
        done = 0
        for fut in as_completed(futs):
            part = fut.result()
            for lab, models in part.items():
                for name, rec in models.items():
                    dst = acc[lab][name]
                    for k, arr in rec["paper"].items():
                        dst["paper"][k].append(arr)
                    dst["plain"].append(rec["plain"])
                    for k in dst["counts"]:
                        dst["counts"][k] += int(rec["mask"][k])
                    for k in dst["deff"]:
                        dst["deff"][k] += rec["deff"][k]
            done += 1
            if done % 4 == 0 or done == len(futs):
                print(f"  chunks {done}/{len(futs)}", flush=True)
    rows = {}
    for lab in (_lab(h) for h in horizons):
        rows[lab] = {name: _finalize(acc[lab][name]) for name in names}
        for name, row in rows[lab].items():
            print(
                f"  {lab:6s}  {name:28s}  MAYE {row['maye']:.5f}  "
                f"yield {row['yield_mse']:.5f}  η_FC {row['false_clear']:.3f}  "
                f"n={row['n_windows']}",
                flush=True,
            )
    return {"horizons": rows, "horizons_sec": list(horizons)}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--chunk", type=int, default=64)
    p.add_argument("--persist-only", action="store_true")
    p.add_argument("--cloudy-min", type=float, default=0.0)
    p.add_argument("--cloudy-max", type=float, default=1.0)
    p.add_argument("--mean-t-min", type=float, default=0.1)
    p.add_argument("--mean-t-max", type=float, default=0.9)
    p.add_argument("--max-windows", type=int, default=0, help="Score only the first N keep windows.")
    args = p.parse_args(argv)

    cache_root = Path(args.cache_dir)
    test_cache = cache_root / "test"
    keep, keep_info = test_keep_indices(
        test_cache,
        cloudy_min=args.cloudy_min,
        cloudy_max=args.cloudy_max,
        mean_t_min=args.mean_t_min,
        mean_t_max=args.mean_t_max,
    )
    if args.max_windows and args.max_windows > 0:
        keep = keep[: int(args.max_windows)]
        print(f"max-windows  n={keep.size}", flush=True)
    names = ("eulerian_persistence",) if args.persist_only else CLASSICAL_NAMES
    scored = score_classical_horizons(
        test_cache,
        keep,
        workers=args.workers,
        chunk=args.chunk,
        names=names,
    )
    report = {
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(),
        "cache": str(test_cache),
        "gold": (
            f"horizon-locked T, Gaussian σ={SIGMA}; moon-dark; "
            f"mean observed transmission {args.mean_t_min:g}–{args.mean_t_max:g} from +20 min y.npy; "
            "per-lead gold from y_h"
        ),
        "false_clear_thresh": MASK_CLOUD_T,
        "persist_hard_dt": PERSIST_HARD_DT,
        "keep": keep_info,
        "models": list(names),
        **scored,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
