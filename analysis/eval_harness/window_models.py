"""all-sky model API: historical all-sky window → future all-sky map.

This module defines only the interface. Concrete baselines live in
``analysis.baselines``; ``eval_harness`` owns the protocol, not the models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class WindowForecastModel(ABC):
    """Predict the full-sky field at ``t_end + horizon`` from a history window.

    This is the all-sky I/O contract:

    - **Input:** transmission (or extinction) cube ``(T_w, npix)`` over
      ``[t_end - history, t_end]`` in **HEALPix nested** order (spherical
      equal-area tiles — not a flat Euclidean image), plus times and ``nobs``.
    - **Output:** predicted full-sky map ``(npix,)`` on the same HEALPix grid.
    """

    name: str = "window_forecast_model"

    @abstractmethod
    def predict(
        self,
        field_window: np.ndarray,
        jd_window: np.ndarray,
        nobs_window: np.ndarray,
        *,
        dt_ahead_days: float,
        horizon_sec: float,
        min_nobs: int,
    ) -> np.ndarray:
        """Return predicted all-sky map, shape ``(npix,)``."""
