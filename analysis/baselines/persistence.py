"""Eulerian persistence: future sky = latest sky."""

from __future__ import annotations

import numpy as np

from analysis.eval_harness.window_models import WindowForecastModel


class EulerianPersistence(WindowForecastModel):
    """Â(t+H) = A(t). The nowcasting floor: freeze the current sky and copy it."""

    name = "eulerian_persistence"

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
        return field_window[-1].astype(np.float64, copy=True)
