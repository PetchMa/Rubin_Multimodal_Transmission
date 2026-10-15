"""Segev frozen-flow 2D cross-correlation, remapped to HEALPix.

Ports the causal motion estimator from Segev's DREAM scheduler
(https://github.com/segevl/dynamic_scheduler): project HEALPix extinction
onto the native 30×30 gnomonic (r = R tan z) km grid, Gaussian-smooth,
brute-force Pearson correlation over ±10 cells, keep corr > 0.5,
average recent vectors, and advect.

That estimator is a single global translation, not a dense flow. Here the
same velocity advects the last transmission map, then returns to nested
HEALPix so it can be scored against the other kinematic baselines.

Source: ``compute_motion_with_correlation`` / ``_healpix_to_grid`` in
``scripts/run_dream_scheduler_grid.py`` (search_range=10, sigma=5, conf>0.5,
BIN_SIZE_KM=1000, R_PROJECTION=10000, MIN_ALT=15).
"""

from __future__ import annotations

from math import hypot

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, shift as nd_shift

from analysis.eval_harness.window_models import WindowForecastModel
from analysis.geometry import equatorial_to_altaz, pix_radec
from analysis.satellite.goes import SITE_LON

# Native Segev cartesian grid: 30 × 30, 1000 km bins, r = R tan(zenith).
CHART_SIZE = 30
BIN_SIZE_KM = 1000.0
R_PROJECTION = 10000.0
GRID_RANGE = 15
MIN_ALT_DEG = 15.0
R_CLIP_KM = 15000.0
SMOOTH_SIGMA = 5.0
SEARCH_RANGE = 10
MIN_CORR = 0.5
MOTION_HISTORY = 5
MIN_OVERLAP = 100


def _nside(npix: int) -> int:
    import healpy as hp

    return int(hp.npix2nside(npix))


def _lst_deg_jd(jd: float) -> float:
    d = float(jd) - 2451545.0
    return (280.46061837 + 360.98564736629 * d + SITE_LON) % 360.0


def _altaz_to_xy(alt_deg: np.ndarray, az_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Segev gnomonic km chart: r = R / sin(alt) * cos(alt) = R cot(alt)."""
    alt_r = np.deg2rad(alt_deg)
    az_r = np.deg2rad(az_deg)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = R_PROJECTION / np.sin(alt_r)
        x_km = -np.cos(alt_r) * np.sin(az_r) * scale
        y_km = np.cos(alt_r) * np.cos(az_r) * scale
    bad = (~np.isfinite(alt_deg)) | (np.asarray(alt_deg) < MIN_ALT_DEG)
    x_km = np.where(bad, np.nan, x_km)
    y_km = np.where(bad, np.nan, y_km)
    return x_km, y_km


def xy_to_altaz(x_km: float, y_km: float) -> tuple[float, float]:
    r = float(np.hypot(x_km, y_km))
    alt_deg = float(np.degrees(np.arctan2(R_PROJECTION, r)))
    az_deg = float(np.degrees(np.arctan2(-x_km, y_km)) % 360.0)
    return alt_deg, az_deg


def xy_to_grid_idx(x_km: float, y_km: float) -> tuple[int, int]:
    xi = int(round((x_km / BIN_SIZE_KM) + GRID_RANGE))
    yi = int(round((y_km / BIN_SIZE_KM) + GRID_RANGE))
    return xi, yi


def grid_idx_to_xy(xi: int, yi: int) -> tuple[float, float]:
    return (xi - GRID_RANGE) * BIN_SIZE_KM, (yi - GRID_RANGE) * BIN_SIZE_KM


def _grid_edges() -> tuple[np.ndarray, np.ndarray]:
    edges = np.arange(-GRID_RANGE, GRID_RANGE + 1, dtype=np.float64) * BIN_SIZE_KM
    return edges, edges.copy()


def healpix_to_segev_grid(
    field: np.ndarray,
    jd: float,
    *,
    nside: int,
    nest: bool = True,
) -> np.ndarray:
    """Bin a HEALPix map onto Segev's 30×30 gnomonic alt-az chart."""
    ra, dec = pix_radec(nside, nest)
    alt, az = equatorial_to_altaz(ra, dec, _lst_deg_jd(jd))
    x_km, y_km = _altaz_to_xy(alt, az)
    vals = np.asarray(field, dtype=np.float64)
    ok = np.isfinite(vals) & np.isfinite(x_km) & np.isfinite(y_km)
    r = np.hypot(x_km, y_km)
    ok &= r <= R_CLIP_KM
    x_edges, y_edges = _grid_edges()
    if not np.any(ok):
        return np.full((CHART_SIZE, CHART_SIZE), np.nan, dtype=np.float64)
    hs, _, _ = np.histogram2d(
        x_km[ok], y_km[ok], bins=[x_edges, y_edges], weights=vals[ok]
    )
    hc, _, _ = np.histogram2d(x_km[ok], y_km[ok], bins=[x_edges, y_edges])
    with np.errstate(divide="ignore", invalid="ignore"):
        grid = hs / hc
    grid[hc == 0] = np.nan
    grid = grid.T
    xc = 0.5 * (x_edges[:-1] + x_edges[1:])
    yc = 0.5 * (y_edges[:-1] + y_edges[1:])
    xg, yg = np.meshgrid(xc, yc)
    grid[np.hypot(xg, yg) > R_CLIP_KM] = np.nan
    return grid


def segev_grid_to_healpix(
    grid: np.ndarray,
    jd: float,
    *,
    nside: int,
    nest: bool = True,
) -> np.ndarray:
    """Bilinear-sample a Segev chart back onto nested HEALPix."""
    import healpy as hp

    ra, dec = pix_radec(nside, nest)
    alt, az = equatorial_to_altaz(ra, dec, _lst_deg_jd(jd))
    x_km, y_km = _altaz_to_xy(alt, az)
    col = (x_km / BIN_SIZE_KM) + GRID_RANGE
    row = (y_km / BIN_SIZE_KM) + GRID_RANGE
    n = int(grid.shape[0])
    on = (
        np.isfinite(x_km)
        & np.isfinite(y_km)
        & (col >= 0)
        & (col <= n - 1)
        & (row >= 0)
        & (row <= n - 1)
        & (alt >= MIN_ALT_DEG)
    )
    filled = np.nan_to_num(grid, nan=0.0)
    sampled = map_coordinates(
        filled,
        np.vstack([np.where(on, row, 0.0), np.where(on, col, 0.0)]),
        order=1,
        mode="nearest",
    )
    valid = np.isfinite(grid).astype(np.float64)
    weight = map_coordinates(
        valid,
        np.vstack([np.where(on, row, 0.0), np.where(on, col, 0.0)]),
        order=1,
        mode="nearest",
    )
    out = np.full(hp.nside2npix(nside), np.nan, dtype=np.float64)
    good = on & (weight > 0.5)
    out[good] = sampled[good]
    return out


def compute_motion_with_correlation(
    grid1: np.ndarray,
    grid2: np.ndarray,
    *,
    sigma: float = SMOOTH_SIGMA,
    search_range: int = SEARCH_RANGE,
    min_overlap: int = MIN_OVERLAP,
) -> tuple[int, int, float]:
    """Integer (dx, dy) that shifts ``grid2`` onto ``grid1`` (Segev sign).

    ``scipy.ndimage.shift(grid2, (dy, dx))`` is the candidate alignment.
    Returned ``(dx, dy)`` is therefore **minus** the cloud displacement from
    frame 1 to frame 2, matching ``run_dream_scheduler_grid.py``.
    """
    g1 = np.nan_to_num(grid1, nan=0.0)
    g2 = np.nan_to_num(grid2, nan=0.0)
    g1s = gaussian_filter(g1, sigma=sigma)
    g2s = gaussian_filter(g2, sigma=sigma)
    m1 = np.isfinite(grid1) & (grid1 != 0)
    m2 = np.isfinite(grid2) & (grid2 != 0)
    best_corr, best_dx, best_dy = -np.inf, 0, 0
    for dy in range(-search_range, search_range + 1):
        for dx in range(-search_range, search_range + 1):
            sh = nd_shift(g2s, (dy, dx), order=1, mode="constant", cval=0.0)
            sm = nd_shift(m2.astype(np.float64), (dy, dx), order=0, mode="constant", cval=0.0) > 0.5
            val = m1 & sm
            n = int(val.sum())
            if n < min_overlap:
                continue
            v1, v2 = g1s[val], sh[val]
            if v1.std() > 0.0 and v2.std() > 0.0:
                corr = float(np.corrcoef(v1, v2)[0, 1])
                if corr > best_corr:
                    best_corr, best_dx, best_dy = corr, dx, dy
    if not np.isfinite(best_corr):
        return 0, 0, -np.inf
    return int(best_dx), int(best_dy), float(best_corr)


def estimate_chart_velocity(
    charts: list[np.ndarray],
    dt_sec: np.ndarray,
    *,
    sigma: float = SMOOTH_SIGMA,
    search: int = SEARCH_RANGE,
    min_corr: float = MIN_CORR,
    min_overlap: int = MIN_OVERLAP,
) -> tuple[float, float] | None:
    """Mean cloud velocity on the Segev chart, in pixels / second.

    Converts Segev's (shift of frame 2 onto frame 1) into feature motion
    from frame 1 → 2: ``(vx, vy) = (-dx, -dy) / dt``.
    """
    if len(charts) < 2 or dt_sec.size != len(charts) - 1:
        return None
    vxs: list[float] = []
    vys: list[float] = []
    for i in range(len(charts) - 1):
        dt = float(dt_sec[i])
        if dt <= 1e-6:
            continue
        dx, dy, corr = compute_motion_with_correlation(
            charts[i],
            charts[i + 1],
            sigma=sigma,
            search_range=search,
            min_overlap=min_overlap,
        )
        if corr < min_corr:
            continue
        vxs.append(-dx / dt)
        vys.append(-dy / dt)
    if not vxs:
        return None
    return float(np.mean(vxs)), float(np.mean(vys))


def _advect_chart(chart: np.ndarray, vx: float, vy: float, horizon_sec: float) -> np.ndarray:
    """Semi-Lagrangian: look back ``(vx, vy) * horizon`` chart pixels."""
    valid = np.isfinite(chart)
    filled = chart.copy()
    if np.any(valid):
        filled[~valid] = float(np.median(chart[valid]))
    else:
        return chart.copy()
    h, w = chart.shape
    yy, xx = np.mgrid[0:h, 0:w]
    src_y = yy - float(horizon_sec) * vy
    src_x = xx - float(horizon_sec) * vx
    advected = map_coordinates(filled, [src_y, src_x], order=1, mode="nearest")
    wt = map_coordinates(valid.astype(np.float64), [src_y, src_x], order=1, mode="nearest")
    return np.where(wt > 0.5, advected, np.nan)


def transmission_to_extinction(field: np.ndarray) -> np.ndarray:
    """``A = -2.5 log10(T)``. Non-positive T becomes NaN."""
    x = np.asarray(field, dtype=np.float64)
    out = np.full_like(x, np.nan)
    pos = np.isfinite(x) & (x > 0)
    out[pos] = -2.5 * np.log10(np.clip(x[pos], 1e-8, None))
    return out


def _from_extinction(ext: np.ndarray, *, as_transmission: bool) -> np.ndarray:
    if not as_transmission:
        return ext
    out = np.full_like(ext, np.nan, dtype=np.float64)
    ok = np.isfinite(ext)
    out[ok] = np.power(10.0, -0.4 * ext[ok])
    return out


class FrozenFlowCrossCorr(WindowForecastModel):
    """Global frozen-flow translation from Segev-style 2D cross-correlation."""

    name = "frozen_flow_crosscorr"

    def __init__(
        self,
        *,
        sigma: float = SMOOTH_SIGMA,
        search_range: int = SEARCH_RANGE,
        min_corr: float = MIN_CORR,
        history: int = MOTION_HISTORY,
        min_overlap: int = MIN_OVERLAP,
        nest: bool = True,
        field: str = "T",
        # Kept so older kwargs still construct; projection is Segev-native.
        chart_size: int = CHART_SIZE,
        max_zenith_deg: float = 90.0 - MIN_ALT_DEG,
    ) -> None:
        self.sigma = float(sigma)
        self.search_range = int(search_range)
        self.min_corr = float(min_corr)
        self.history = int(history)
        self.min_overlap = int(min_overlap)
        self.nest = bool(nest)
        self.field = str(field)
        self.chart_size = int(chart_size)
        self.max_zenith_deg = float(max_zenith_deg)

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
        last = field_window[-1].astype(np.float64, copy=True)
        n_hist = min(self.history, int(field_window.shape[0]))
        if n_hist < 2 or dt_ahead_days <= 0:
            return last

        as_t = self.field.upper() != "A"
        nside = _nside(last.size)
        sl = slice(-n_hist, None)
        fields = field_window[sl].astype(np.float64, copy=True)
        nobs = nobs_window[sl]
        jds = jd_window[sl]
        fields[(nobs < min_nobs) | ~np.isfinite(fields)] = np.nan
        if as_t:
            ext = np.stack([transmission_to_extinction(fields[k]) for k in range(n_hist)])
        else:
            ext = fields

        charts: list[np.ndarray] = []
        for k in range(n_hist):
            charts.append(
                healpix_to_segev_grid(
                    ext[k],
                    float(jds[k]),
                    nside=nside,
                    nest=self.nest,
                )
            )
        dt_sec = np.diff(jds.astype(np.float64)) * 86400.0
        vel = estimate_chart_velocity(
            charts,
            dt_sec,
            sigma=self.sigma,
            search=self.search_range,
            min_corr=self.min_corr,
            min_overlap=self.min_overlap,
        )
        if vel is None:
            return last

        vx, vy = vel
        # A 30×30 all-sky chart has ~5 deg cells. Native DREAM cadence is
        # sub-pixel; resampling through the chart with a 0-pixel (or noisy
        # 1-pixel) shift is worse than persistence. Only advect when the
        # implied displacement is at least half a cell.
        if hypot(vx, vy) * abs(float(horizon_sec)) < 1.0:
            return last
        advected = _advect_chart(charts[-1], vx, vy, float(horizon_sec))
        t_target = float(jds[-1]) + dt_ahead_days
        pred_ext = segev_grid_to_healpix(
            advected,
            t_target,
            nside=nside,
            nest=self.nest,
        )
        pred = _from_extinction(pred_ext, as_transmission=as_t)
        missing = ~np.isfinite(pred)
        pred[missing] = last[missing]
        return pred
