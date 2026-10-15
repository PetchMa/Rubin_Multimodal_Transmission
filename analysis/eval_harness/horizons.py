"""Horizon matching and metric helpers."""

from __future__ import annotations

import numpy as np

from .data import HORIZON_LABELS, TRUTH_FLOOR


def find_future(
    jd: np.ndarray,
    i: int,
    horizon_days: float,
    *,
    tol_frac: float = 0.25,
) -> int | None:
    """Return index of the map nearest to ``jd[i] + horizon``, if close enough.

    Parameters
    ----------
    tol_frac:
        Accept a match only if ``|t_j - (t_i + H)| <= tol_frac * H``.
    """
    n_t = len(jd)
    target = jd[i] + horizon_days
    j = int(np.searchsorted(jd, target))
    if j >= n_t:
        return None
    cands = [j]
    if j - 1 > i:
        cands.append(j - 1)
    j_best = min(cands, key=lambda jj: abs(jd[jj] - target))
    if j_best <= i:
        return None
    if abs(jd[j_best] - target) > tol_frac * horizon_days:
        return None
    return j_best


def build_forecast_pairs(
    jd: np.ndarray,
    horizons_sec: tuple[int, ...],
    *,
    tol_frac: float = 0.25,
) -> dict[int, list[tuple[int, int, float]]]:
    """Precompute (origin_idx, target_idx, dt_days) pairs per horizon.

    Origins start at index 0; pairs with no valid future map are omitted.
    """
    pairs: dict[int, list[tuple[int, int, float]]] = {H: [] for H in horizons_sec}
    for i in range(len(jd)):
        for H in horizons_sec:
            j = find_future(jd, i, H / 86400.0, tol_frac=tol_frac)
            if j is None:
                continue
            pairs[H].append((i, j, float(jd[j] - jd[i])))
    return pairs


def rmse(err: np.ndarray) -> float:
    return float(np.sqrt(np.mean(err**2)))


def debias(err: np.ndarray) -> np.ndarray:
    """Remove the mean error, leaving only spatial-pattern error.

    A model that predicts the correct spatial structure but the wrong overall
    sky level scores badly on plain RMSE; a model that predicts only a uniform
    level shift scores well. Subtracting the per-window mean error separates
    the two, so debiased RMSE measures all-sky *pattern* skill alone.
    """
    return err - float(np.mean(err))


def mae(err: np.ndarray) -> float:
    return float(np.mean(np.abs(err)))


def fractional_metrics(
    err: np.ndarray,
    truth: np.ndarray,
    *,
    floor: float = TRUTH_FLOOR,
) -> dict:
    """Relative-error metrics, ``err / truth``, restricted to ``truth >= floor``.

    Transmission approaches zero under thick cloud, where a fractional error is
    unbounded and dominated by the denominator. The floor drops those tiles so
    the statistic reflects forecast quality rather than division by a small
    number; ``n`` and ``coverage`` report how much was kept.
    """
    ok = np.isfinite(truth) & np.isfinite(err) & (np.abs(truth) >= floor)
    if not np.any(ok):
        return {
            "n": 0,
            "coverage": 0.0,
            "mean_abs_frac_pct": float("nan"),
            "median_abs_frac_pct": float("nan"),
            "rms_frac_pct": float("nan"),
            "frac_bias_pct": float("nan"),
        }
    rel = err[ok] / truth[ok]
    return {
        "n": int(rel.size),
        "coverage": float(ok.mean()),
        "mean_abs_frac_pct": float(100.0 * np.mean(np.abs(rel))),
        "median_abs_frac_pct": float(100.0 * np.median(np.abs(rel))),
        "rms_frac_pct": float(100.0 * np.sqrt(np.mean(rel**2))),
        "frac_bias_pct": float(100.0 * np.mean(rel)),
    }


def summarize_errors(
    err_model: np.ndarray,
    err_baseline: np.ndarray,
    *,
    horizon_sec: int,
    truth: np.ndarray | None = None,
    truth_floor: float = TRUTH_FLOOR,
    err_model_debiased: np.ndarray | None = None,
    err_baseline_debiased: np.ndarray | None = None,
) -> dict:
    """Aggregate look-ahead metrics for one model at one horizon."""
    if err_model.shape != err_baseline.shape:
        raise ValueError("model and baseline error arrays must match")
    extra: dict = {}
    if err_model_debiased is not None and err_baseline_debiased is not None:
        d_m, d_b = rmse(err_model_debiased), rmse(err_baseline_debiased)
        extra.update(
            {
                "debiased_rmse": d_m,
                "baseline_debiased_rmse": d_b,
                "debiased_rmse_improvement_pct": float(
                    100.0 * (1.0 - d_m / d_b)
                )
                if d_b > 0
                else float("nan"),
            }
        )
    if truth is not None:
        if truth.shape != err_model.shape:
            raise ValueError("truth array must match error arrays")
        frac = fractional_metrics(err_model, truth, floor=truth_floor)
        frac_base = fractional_metrics(err_baseline, truth, floor=truth_floor)
        extra.update({
            "truth_floor": float(truth_floor),
            "n_frac": frac["n"],
            "frac_coverage": frac["coverage"],
            "mean_abs_frac_pct": frac["mean_abs_frac_pct"],
            "median_abs_frac_pct": frac["median_abs_frac_pct"],
            "rms_frac_pct": frac["rms_frac_pct"],
            "frac_bias_pct": frac["frac_bias_pct"],
            "baseline_median_abs_frac_pct": frac_base["median_abs_frac_pct"],
            "baseline_rms_frac_pct": frac_base["rms_frac_pct"],
            "median_abs_frac_improvement_pct": float(
                100.0
                * (
                    1.0
                    - frac["median_abs_frac_pct"] / frac_base["median_abs_frac_pct"]
                )
            )
            if frac_base["median_abs_frac_pct"]
            else float("nan"),
            "rms_frac_improvement_pct": float(
                100.0 * (1.0 - frac["rms_frac_pct"] / frac_base["rms_frac_pct"])
            )
            if frac_base["rms_frac_pct"]
            else float("nan"),
        })
    return {
        **extra,
        "horizon_sec": int(horizon_sec),
        "horizon_label": HORIZON_LABELS.get(horizon_sec, f"{horizon_sec}s"),
        "n": int(err_model.size),
        "rmse": rmse(err_model),
        "mae": mae(err_model),
        "bias": float(np.mean(err_model)),
        "median_abs": float(np.median(np.abs(err_model))),
        "baseline_rmse": rmse(err_baseline),
        "baseline_mae": mae(err_baseline),
        "rmse_improvement_pct": float(100.0 * (1.0 - rmse(err_model) / rmse(err_baseline))),
        "mae_improvement_pct": float(100.0 * (1.0 - mae(err_model) / mae(err_baseline))),
        "fraction_closer_than_baseline": float(
            np.mean(np.abs(err_model) < np.abs(err_baseline))
        ),
    }
