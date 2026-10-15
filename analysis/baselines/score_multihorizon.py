#!/usr/bin/env python3
"""Holdout scores for the 24-frame multihorizon models.

Same holdout as ``score_test_yield_mse``: moon-dark, cloudy fraction ≤0.99,
mean observed transmission ∈ [0.1, 0.9] on the +20 min map.
Reports MAYE, yield MSE, η_FC@0.63 vs persist at 90 s / 5 min / 20 min / 30 min.

Loads the multimodal forecaster, or camera-only SimVP / ConvLSTM when the
checkpoint ``arch`` is ``simvp`` / ``convlstm``.

    python -u -m analysis.baselines.score_multihorizon \\
        --ckpt $TRANSMISSIONS_MNT/baselines_neural/paper_multih_t24/seed_42/model/best.pt
    python -u -m analysis.baselines.score_multihorizon \\
        --ckpt .../paper_simvp_multih_t24/seed_42/simvp/best.pt
"""

from __future__ import annotations

from analysis.paths import MNT

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

from analysis.baselines.neural.wrapper import nowcast_from_checkpoint, slim_camera_pairs
from analysis.baselines.score_test_yield_mse import (
    MASK_CLOUD_T,
    MIN_SIN,
    PERSIST_HARD_DT,
    SIGMA,
    T_LOG_FLOOR,
    accumulate_deff,
    add_mask_fields,
    add_paper_fields,
    finalize_deff,
    mask_counts,
    paper_window_metrics,
    per_window_plain_mse,
    summarize_windows,
    test_keep_indices,
    _empty_deff,
)
from analysis.forecaster import program as forecaster
from analysis.forecaster.program import parse_ablate
from analysis.forecaster.geom import attach_wind_decode_stats, local_pix_altaz, nside_from_npix
from analysis.forecaster.multihorizon import HORIZON_LABELS, HORIZONS_SEC, wrap_multihorizon
from analysis.forecaster.port_prep import attach_horizon_lock
from analysis.forecaster.sat_ingest import wrap_rich_sat
from analysis.forecaster.train import _set_sky_norm
from analysis.forecaster.train_multihorizon import _forward_horizons
from analysis.our_model.mask_train import decode_camera_t
from analysis.our_model.sat_pretrain.dataset import NATIVE_IR_HW
from analysis.our_model.model import (
    PairsDataset,
    TrainConfig,
    apply_batch_preprocess,
    make_loader,
    resolve_pairs,
    _to_device,
)
from analysis.torch_util import resolve_device

DEFAULT_CACHE = Path(str(MNT) + "/pairs/dream_multih_t24_encoded")
DEFAULT_CKPT = Path(
    str(MNT) + "/baselines_neural/paper_multih_t24/seed_42/model/best.pt"
)
_PAPER_KEYS = ("yield_mse", "maye", "dm5", "dm5_mae", "deff_mae")
_CANDIDATE_KS = (2, 5, 10, 20)
_CANDIDATE_DRAWS = 32
_RISK_EDGES = np.linspace(0.0, 1.0, 201)


def _lab(horizon_sec: int) -> str:
    return HORIZON_LABELS.get(int(horizon_sec), f"{int(horizon_sec)}s")


def _new_acc() -> dict:
    return {
        "paper": {name: [] for name in _PAPER_KEYS},
        "persist_paper": {name: [] for name in _PAPER_KEYS},
        "plains": [],
        "persist_plains": [],
        "counts": {"tp": 0, "fn": 0, "fp": 0, "n_gold_cloud": 0, "n_conf": 0},
        "persist_c": {"tp": 0, "fn": 0, "fp": 0, "n_gold_cloud": 0, "n_conf": 0},
        "deff": _empty_deff(),
        "persist_deff": _empty_deff(),
        "selection": {
            "model_true_t": [],
            "persist_true_t": [],
            "oracle_true_t": [],
        },
        "candidate_sets": {
            k: {"model_true_t": [], "persist_true_t": [], "oracle_true_t": []}
            for k in _CANDIDATE_KS
        },
        "risk_coverage": {
            name: {
                "all": np.zeros(_RISK_EDGES.size - 1, dtype=np.int64),
                "cloud": np.zeros(_RISK_EDGES.size - 1, dtype=np.int64),
            }
            for name in ("model", "persist")
        },
    }


def _selected_true_t(
    score: np.ndarray, t_true: np.ndarray, eligible: np.ndarray
) -> np.ndarray:
    """True transmission at each row's highest-scored eligible tile."""
    ranked = np.where(eligible, np.nan_to_num(score, nan=-np.inf), -np.inf)
    has_choice = eligible.any(axis=-1) & np.isfinite(ranked).any(axis=-1)
    choice = np.argmax(ranked, axis=-1)
    out = np.full(score.shape[0], np.nan, dtype=np.float64)
    rows = np.flatnonzero(has_choice)
    out[rows] = t_true[rows, choice[rows]]
    return out


def _selection_summary(selection: dict[str, list[np.ndarray]]) -> dict:
    model_t = np.concatenate(selection["model_true_t"])
    persist_t = np.concatenate(selection["persist_true_t"])
    oracle_t = np.concatenate(selection["oracle_true_t"])
    valid = np.isfinite(model_t) & np.isfinite(persist_t) & np.isfinite(oracle_t)
    model_t = np.clip(model_t[valid], T_LOG_FLOOR, 1.0)
    persist_t = np.clip(persist_t[valid], T_LOG_FLOOR, 1.0)
    oracle_t = np.clip(oracle_t[valid], T_LOG_FLOOR, 1.0)
    gain = 1.25 * np.log10(model_t / persist_t)
    n = int(gain.size)
    return {
        "n_paired": n,
        "model_mean_true_t_at_argmax": float(np.mean(model_t)) if n else float("nan"),
        "persist_mean_true_t_at_argmax": float(np.mean(persist_t)) if n else float("nan"),
        "oracle_mean_true_t": float(np.mean(oracle_t)) if n else float("nan"),
        "mean_depth_gain_vs_persist_mag": float(np.mean(gain)) if n else float("nan"),
        "median_depth_gain_vs_persist_mag": float(np.median(gain)) if n else float("nan"),
        "depth_gain_standard_error_mag": (
            float(np.std(gain, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        ),
        "fraction_model_deeper": float(np.mean(gain > 0.0)) if n else float("nan"),
        "fraction_same_choice_depth": (
            float(np.mean(np.isclose(gain, 0.0, atol=1e-12))) if n else float("nan")
        ),
    }


def _candidate_set_true_t(
    model_score: np.ndarray,
    persist_score: np.ndarray,
    t_true: np.ndarray,
    eligible: np.ndarray,
    rng: np.random.Generator,
) -> dict[int, dict[str, np.ndarray]]:
    """Average policy value per window over random candidate-set draws."""
    bsz = model_score.shape[0]
    max_k = max(_CANDIDATE_KS)
    out = {
        k: {
            name: np.full(bsz, np.nan, dtype=np.float64)
            for name in ("model_true_t", "persist_true_t", "oracle_true_t")
        }
        for k in _CANDIDATE_KS
    }
    for row in range(bsz):
        pool = np.flatnonzero(
            eligible[row]
            & np.isfinite(model_score[row])
            & np.isfinite(persist_score[row])
            & np.isfinite(t_true[row])
        )
        if pool.size < max_k:
            continue
        candidates = np.empty((_CANDIDATE_DRAWS, max_k), dtype=np.int64)
        for draw in range(_CANDIDATE_DRAWS):
            candidates[draw] = rng.choice(pool, size=max_k, replace=False)
        for k in _CANDIDATE_KS:
            cand = candidates[:, :k]
            truth = t_true[row, cand]
            model_pick = np.argmax(model_score[row, cand], axis=1)
            persist_pick = np.argmax(persist_score[row, cand], axis=1)
            oracle_pick = np.argmax(truth, axis=1)
            draws = np.arange(_CANDIDATE_DRAWS)
            out[k]["model_true_t"][row] = np.mean(truth[draws, model_pick])
            out[k]["persist_true_t"][row] = np.mean(truth[draws, persist_pick])
            out[k]["oracle_true_t"][row] = np.mean(truth[draws, oracle_pick])
    return out


def _candidate_summary(
    selection: dict[str, list[np.ndarray]], rng: np.random.Generator
) -> dict:
    model_t = np.concatenate(selection["model_true_t"])
    persist_t = np.concatenate(selection["persist_true_t"])
    oracle_t = np.concatenate(selection["oracle_true_t"])
    valid = np.isfinite(model_t) & np.isfinite(persist_t) & np.isfinite(oracle_t)
    model_t = np.clip(model_t[valid], T_LOG_FLOOR, 1.0)
    persist_t = np.clip(persist_t[valid], T_LOG_FLOOR, 1.0)
    oracle_t = np.clip(oracle_t[valid], T_LOG_FLOOR, 1.0)
    n = int(model_t.size)
    photon_ratio = float(model_t.sum() / persist_t.sum()) if n else float("nan")
    depth = 1.25 * np.log10(photon_ratio) if n else float("nan")
    boot = np.full(2000, np.nan, dtype=np.float64)
    if n:
        for j in range(boot.size):
            idx = rng.integers(0, n, size=n)
            ratio = model_t[idx].sum() / persist_t[idx].sum()
            boot[j] = 1.25 * np.log10(ratio)
    per_window_gain = 1.25 * np.log10(model_t / persist_t)
    return {
        "n_windows": n,
        "draws_per_window": _CANDIDATE_DRAWS,
        "model_mean_true_t": float(np.mean(model_t)) if n else float("nan"),
        "persist_mean_true_t": float(np.mean(persist_t)) if n else float("nan"),
        "oracle_mean_true_t": float(np.mean(oracle_t)) if n else float("nan"),
        "photon_yield_ratio_vs_persist": photon_ratio,
        "photon_yield_gain_percent": (
            100.0 * (photon_ratio - 1.0) if n else float("nan")
        ),
        "equivalent_depth_gain_mag": float(depth),
        "depth_gain_bootstrap_95_mag": (
            [float(x) for x in np.percentile(boot, [2.5, 97.5])]
            if n
            else [float("nan"), float("nan")]
        ),
        "fraction_windows_model_deeper": (
            float(np.mean(per_window_gain > 0.0)) if n else float("nan")
        ),
        "fraction_model_selected_clear": (
            float(np.mean(model_t > MASK_CLOUD_T)) if n else float("nan")
        ),
        "fraction_persist_selected_clear": (
            float(np.mean(persist_t > MASK_CLOUD_T)) if n else float("nan")
        ),
    }


def _accumulate_risk_coverage(
    acc: dict[str, dict[str, np.ndarray]],
    model_score: np.ndarray,
    persist_score: np.ndarray,
    t_true: np.ndarray,
    eligible: np.ndarray,
) -> None:
    cloudy = t_true <= MASK_CLOUD_T
    for name, score in (("model", model_score), ("persist", persist_score)):
        valid = eligible & np.isfinite(score)
        values = np.clip(score[valid], 0.0, 1.0)
        acc[name]["all"] += np.histogram(values, bins=_RISK_EDGES)[0]
        acc[name]["cloud"] += np.histogram(
            values[cloudy[valid]], bins=_RISK_EDGES
        )[0]


def _risk_coverage_summary(
    acc: dict[str, dict[str, np.ndarray]],
) -> dict:
    out = {
        "definition": (
            "coverage is fraction of eligible tiles accepted at predicted-T threshold; "
            "risk is fraction of accepted tiles with true T <= 0.63"
        ),
        "curves": {},
    }
    for name, hist in acc.items():
        accepted = np.cumsum(hist["all"][::-1])[::-1].astype(np.float64)
        failures = np.cumsum(hist["cloud"][::-1])[::-1].astype(np.float64)
        total = float(hist["all"].sum())
        coverage = accepted / total if total else np.full_like(accepted, np.nan)
        risk = np.divide(
            failures,
            accepted,
            out=np.full_like(failures, np.nan),
            where=accepted > 0,
        )
        out["curves"][name] = {
            "threshold": [float(x) for x in _RISK_EDGES[:-1]],
            "coverage": [float(x) for x in coverage],
            "cloud_failure_risk": [float(x) for x in risk],
            "n_eligible_tiles": int(total),
        }
    return out


def _sin_alt_np(model, t_last: torch.Tensor) -> np.ndarray:
    sin_alt = getattr(model, "last_sin_alt", None)
    if sin_alt is None:
        alt, _ = local_pix_altaz(nside_from_npix(int(t_last.shape[-1])))
        sin_alt = torch.sin(
            torch.deg2rad(torch.as_tensor(alt, device=t_last.device, dtype=t_last.dtype))
        ).clamp(min=0.0)
    if sin_alt.ndim == 1:
        sin_alt = sin_alt.unsqueeze(0).expand(t_last.shape[0], -1)
    sa = sin_alt.detach().cpu().numpy()
    tl = t_last.detach().cpu().numpy()
    if sa.ndim == 1:
        sa = np.broadcast_to(sa, tl.shape)
    return np.asarray(sa)


@torch.no_grad()
def score_ckpt(
    cache_root: Path,
    pairs_dir: Path,
    indices: np.ndarray,
    device,
    *,
    ckpt: Path,
    n_frames: int = 24,
    ablate: str = "",
) -> dict:
    cfg = TrainConfig(
        batch_size=16,
        num_workers=4,
        seed=42,
        verbose=True,
        d_model=64,
        n_heads=4,
        n_layers=2,
        sky_width=16,
        n_frames=int(n_frames),
    )
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    info = payload.get("info") or {} if isinstance(payload, dict) else {}
    arch = str(payload.get("arch") or info.get("arch") or "").strip().lower()
    neural = arch in {"simvp", "convlstm", "conv_lstm", "camera_nowcast"}
    ablate_set = parse_ablate(ablate)
    if isinstance(payload, dict):
        if payload.get("n_frames") is not None:
            cfg.n_frames = int(payload["n_frames"])
        elif isinstance(info, dict) and info.get("n_frames") is not None:
            cfg.n_frames = int(info["n_frames"])
    test_cache = cache_root / "test"
    test_data = PairsDataset.from_memmap(test_cache, device)
    test_meta = json.loads((test_cache / "meta.json").read_text())
    nside = int(test_meta["nside"])
    if neural:
        slim_camera_pairs(test_data)
        print(f"  neural {arch or 'camera_nowcast'}  camera-only", flush=True)
    else:
        cfg.wind_age_mean = float(test_data.norm.wind[-1].mean)
        cfg.wind_age_std = float(test_data.norm.wind[-1].std)
        attach_wind_decode_stats(cfg, test_data.norm)
        if "sat" not in ablate_set:
            test_data = wrap_rich_sat(
                test_data,
                pairs_dir,
                "test",
                test_cache,
                n_frames=cfg.n_frames,
                sat_size=NATIVE_IR_HW,
            )
        elif ablate_set:
            print(
                f"  ablate sat  skip rich-sat overlay  "
                f"({','.join(sorted(ablate_set))})",
                flush=True,
            )
    test_data = wrap_multihorizon(test_data, test_cache)
    test_data = attach_horizon_lock(test_data, test_cache)
    test_data = Subset(test_data, indices.tolist())
    loader = make_loader(test_data, cfg, shuffle=False, device=device)
    if neural:
        model = nowcast_from_checkpoint(ckpt, device)
        print(f"  load {ckpt}  arch={model.name}", flush=True)
    else:
        model = forecaster.build_model(int(nside), cfg).to(device)
        model.set_ablate(ablate_set)
        missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
        print(
            f"  load {ckpt}  missing={len(missing)} unexpected={len(unexpected)}"
            + (f"  ablate={','.join(sorted(ablate_set))}" if ablate_set else ""),
            flush=True,
        )
    sky_norm = PairsDataset.from_memmap(test_cache, resolve_device("cpu")).norm.sky
    _set_sky_norm(model, sky_norm)
    model.eval()

    acc = {k: _new_acc() for k in range(len(HORIZONS_SEC))}
    candidate_rng = [np.random.default_rng(4200 + k) for k in range(len(HORIZONS_SEC))]

    for raw in loader:
        b = apply_batch_preprocess(model, _to_device(raw, device))
        hat = _forward_horizons(model, b)
        y, ym = b["y"], b["y_mask"]
        if y.ndim != 3:
            raise SystemExit(f"expected stacked y (B, K, npix), got {tuple(y.shape)}")
        bsz, n_h, npix = int(y.shape[0]), int(y.shape[1]), int(y.shape[2])
        t_y, y_valid = decode_camera_t(
            y.reshape(bsz * n_h, npix),
            ym.reshape(bsz * n_h, npix),
            sky_norm,
            sigma=SIGMA,
        )
        t_y = t_y.reshape(bsz, n_h, npix)
        y_valid = y_valid.reshape(bsz, n_h, npix)
        t_last, last_valid = decode_camera_t(
            b["last"], b["last_mask"], sky_norm, sigma=SIGMA
        )
        hit = b.get("horizon_hit")
        sa = _sin_alt_np(model, t_last)
        tl = t_last.detach().cpu().numpy()
        lm = (last_valid > 0).cpu().numpy()
        th_all = hat.detach().cpu().numpy()
        ty_all = t_y.detach().cpu().numpy()
        yv = (y_valid > 0).cpu().numpy()
        hit_np = None if hit is None else hit.detach().cpu().numpy()
        for k in range(n_h):
            ym_k = yv[:, k]
            if hit_np is not None:
                ym_k = ym_k & (hit_np[:, k] > 0)[:, None]
            th = th_all[:, k]
            ty = ty_all[:, k]
            eligible = ym_k & lm & (sa >= MIN_SIN)
            acc[k]["selection"]["model_true_t"].append(
                _selected_true_t(th, ty, eligible)
            )
            acc[k]["selection"]["persist_true_t"].append(
                _selected_true_t(tl, ty, eligible)
            )
            acc[k]["selection"]["oracle_true_t"].append(
                _selected_true_t(ty, ty, eligible)
            )
            candidate = _candidate_set_true_t(
                th, tl, ty, eligible, candidate_rng[k]
            )
            for candidate_k, values in candidate.items():
                for name, value in values.items():
                    acc[k]["candidate_sets"][candidate_k][name].append(value)
            _accumulate_risk_coverage(
                acc[k]["risk_coverage"], th, tl, ty, eligible
            )
            pm = paper_window_metrics(th, ty, ym_k, sa)
            pp = paper_window_metrics(tl, ty, ym_k, sa)
            for name in _PAPER_KEYS:
                acc[k]["paper"][name].append(pm[name])
                acc[k]["persist_paper"][name].append(pp[name])
            acc[k]["plains"].append(per_window_plain_mse(th, ty, ym_k))
            acc[k]["persist_plains"].append(per_window_plain_mse(tl, ty, ym_k))
            part = mask_counts(th, ty, ym_k, sa, lm)
            persist_part = mask_counts(tl, ty, ym_k, sa, lm)
            for name in acc[k]["counts"]:
                acc[k]["counts"][name] += int(part[name])
                acc[k]["persist_c"][name] += int(persist_part[name])
            accumulate_deff(acc[k]["deff"], th, ty, ym_k, lm, sa)
            accumulate_deff(acc[k]["persist_deff"], tl, ty, ym_k, lm, sa)

    horizons = {}
    for k, h in enumerate(HORIZONS_SEC):
        cat = {name: np.concatenate(v) for name, v in acc[k]["paper"].items()}
        pcat = {name: np.concatenate(v) for name, v in acc[k]["persist_paper"].items()}
        row = add_mask_fields(
            summarize_windows(cat["yield_mse"], np.concatenate(acc[k]["plains"])),
            acc[k]["counts"],
        )
        add_paper_fields(row, cat, finalize_deff(acc[k]["deff"]))
        prow = add_mask_fields(
            summarize_windows(pcat["yield_mse"], np.concatenate(acc[k]["persist_plains"])),
            acc[k]["persist_c"],
        )
        add_paper_fields(prow, pcat, finalize_deff(acc[k]["persist_deff"]))
        pers = float(prow["yield_mse"])
        mdl = float(row["yield_mse"])
        row["horizon_sec"] = int(h)
        row["label"] = _lab(h)
        row["persist_yield_mse"] = pers
        row["persist_maye"] = float(prow["maye"])
        row["persist_false_clear"] = float(prow["false_clear"])
        row["yield_vs_persist"] = (
            float(mdl / pers) if pers and np.isfinite(pers) and pers > 0 else float("nan")
        )
        row["persist_check"] = prow
        row["argmax_survey_depth"] = _selection_summary(acc[k]["selection"])
        row["candidate_set_survey_depth"] = {
            str(candidate_k): _candidate_summary(
                acc[k]["candidate_sets"][candidate_k],
                np.random.default_rng(8400 + 100 * k + candidate_k),
            )
            for candidate_k in _CANDIDATE_KS
        }
        row["risk_coverage"] = _risk_coverage_summary(acc[k]["risk_coverage"])
        horizons[_lab(h)] = row
        selection = row["argmax_survey_depth"]
        print(
            f"  {_lab(h):6s}  MAYE {row['maye']:.5f}  yield {mdl:.5f}  "
            f"η_FC {row['false_clear']:.3f}  vs persist yield {pers:.5f}  "
            f"({row['yield_vs_persist']:.3f})  persist η_FC {prow['false_clear']:.3f}  "
            f"argmax Δm5 {selection['mean_depth_gain_vs_persist_mag']:+.4f} mag",
            flush=True,
        )
        for candidate_k, candidate_row in row["candidate_set_survey_depth"].items():
            print(
                f"    K={candidate_k:>2s}  photons "
                f"{candidate_row['photon_yield_gain_percent']:+.2f}%  "
                f"depth {candidate_row['equivalent_depth_gain_mag']:+.4f} mag  "
                f"95% CI {candidate_row['depth_gain_bootstrap_95_mag']}",
                flush=True,
            )
    info = payload.get("info") if isinstance(payload, dict) else None
    return {
        "horizons": horizons,
        "horizons_sec": list(HORIZONS_SEC),
        "ckpt_info": info if isinstance(info, dict) else {},
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--pairs", type=Path, default=None)
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--n-frames", type=int, default=24)
    p.add_argument("--cloudy-min", type=float, default=0.0)
    p.add_argument("--cloudy-max", type=float, default=0.99)
    p.add_argument("--mean-t-min", type=float, default=0.1)
    p.add_argument("--mean-t-max", type=float, default=0.9)
    p.add_argument(
        "--ablate",
        default="",
        help=(
            "Zero the same inputs as train_multihorizon.py --ablate "
            "(sat,wind or dream). Required for curriculum ablation ckpts."
        ),
    )
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
    pairs_dir = resolve_pairs(args.pairs)
    device = resolve_device(None)
    print(
        f"multihorizon paper test  ckpt={args.ckpt}  n_keep={keep.size}  "
        f"ablate={args.ablate or 'off'}  device={device}",
        flush=True,
    )
    scored = score_ckpt(
        cache_root,
        pairs_dir,
        keep,
        device,
        ckpt=Path(args.ckpt),
        n_frames=args.n_frames,
        ablate=str(args.ablate or ""),
    )
    out = (
        Path(args.out)
        if args.out is not None
        else Path(args.ckpt).parent / "suite_test_mean_t_0.1_0.9_multih.json"
    )
    report = {
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(),
        "ckpt": str(args.ckpt),
        "cache": str(test_cache),
        "gold": (
            f"horizon-locked T, Gaussian σ={SIGMA}; moon-dark; "
            f"mean observed transmission {args.mean_t_min:g}–{args.mean_t_max:g} from +20 min y.npy"
        ),
        "false_clear_thresh": MASK_CLOUD_T,
        "persist_hard_dt": PERSIST_HARD_DT,
        "keep": keep_info,
        "ablate": str(args.ablate or ""),
        "horizons": scored["horizons"],
        "horizons_sec": scored["horizons_sec"],
        "ckpt_info": scored["ckpt_info"],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
