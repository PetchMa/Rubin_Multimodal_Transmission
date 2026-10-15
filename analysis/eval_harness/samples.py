"""Build all-sky eval samples: historical all-sky window → future sky."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import HISTORY_SEC, HORIZON_SEC
from .horizons import find_future


@dataclass(frozen=True)
class WindowSample:
    """One all-sky example.

    History is the full-sky cube on ``window_indices`` (ending at ``end_idx``).
    Target is the measured full-sky map at ``target_idx`` (~ end + horizon).
    """

    end_idx: int
    target_idx: int
    window_indices: tuple[int, ...]
    t_end_jd: float
    t_target_jd: float
    dt_ahead_days: float
    history_sec: float
    horizon_sec: float


def build_window_samples(
    jd: np.ndarray,
    *,
    history_sec: float = HISTORY_SEC,
    horizon_sec: float = HORIZON_SEC,
    tol_frac: float = 0.25,
    min_window_frames: int = 30,
) -> list[WindowSample]:
    """Enumerate valid (history window → +horizon target) samples."""
    hist = history_sec / 86400.0
    samples: list[WindowSample] = []
    for end_idx in range(len(jd)):
        t_end = float(jd[end_idx])
        if t_end - hist < jd[0]:
            continue
        target_idx = find_future(
            jd, end_idx, horizon_sec / 86400.0, tol_frac=tol_frac
        )
        if target_idx is None:
            continue
        win = np.where((jd >= t_end - hist) & (jd <= t_end))[0]
        if win.size < min_window_frames:
            continue
        samples.append(
            WindowSample(
                end_idx=int(end_idx),
                target_idx=int(target_idx),
                window_indices=tuple(int(i) for i in win.tolist()),
                t_end_jd=t_end,
                t_target_jd=float(jd[target_idx]),
                dt_ahead_days=float(jd[target_idx] - t_end),
                history_sec=float(history_sec),
                horizon_sec=float(horizon_sec),
            )
        )
    return samples


def sample_windows(
    samples: list[WindowSample],
    *,
    n: int | None = None,
    seed: int = 0,
) -> list[WindowSample]:
    """Random subset of windows (without replacement). ``n=None`` keeps all."""
    if n is None or n >= len(samples):
        return list(samples)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(samples), size=n, replace=False)
    return [samples[i] for i in sorted(idx.tolist())]
