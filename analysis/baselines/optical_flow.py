"""Dense optical flow on a zenithal chart, remapped to HEALPix.

Traditional nowcasting benchmark: estimate a pixel-level velocity field
between recent observer-frame images (dense Lucas–Kanade) and extrapolate
the latest cloud map along that flow.

The HEALPix product is equatorial, so each frame is first projected to an
azimuthal-equidistant alt-az image at that frame's LST. Flow is therefore
atmospheric motion in the camera frame, not a mix of clouds and Earth
rotation. After semi-Lagrangian advection the chart is sampled back onto
HEALPix at the target LST.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.ndimage import map_coordinates, sobel, uniform_filter, zoom

from analysis.eval_harness.window_models import WindowForecastModel
from analysis.geometry import altaz_to_equatorial, equatorial_to_altaz, pix_radec
from analysis.satellite.goes import SITE_LON


CHART_SIZE = 256
MAX_ZENITH_DEG = 80.0
LK_WIN = 15
LK_LEVELS = 2
LK_EPS = 1e-8


@lru_cache(maxsize=4)
def _chart_altaz(size: int, max_zenith_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Altitude / azimuth (deg) at every chart pixel; NaN outside the disk."""
    cy = cx = (size - 1) / 2.0
    rmax = size / 2.0 - 1.0
    yy, xx = np.mgrid[0:size, 0:size]
    east = xx - cx
    north = cy - yy
    radius = np.hypot(east, north)
    zenith = radius * (max_zenith_deg / rmax)
    alt = 90.0 - zenith
    az = np.rad2deg(np.arctan2(east, north)) % 360.0
    outside = radius > rmax
    alt = np.where(outside, np.nan, alt)
    az = np.where(outside, np.nan, az)
    return alt.astype(np.float64), az.astype(np.float64)


@lru_cache(maxsize=8)
def _pix_radec(nside: int, nest: bool) -> tuple[np.ndarray, np.ndarray]:
    return pix_radec(nside, nest)


def _nside(npix: int) -> int:
    import healpy as hp

    return int(hp.npix2nside(npix))


def _lst_deg_jd(jd: float) -> float:
    """Local mean sidereal time [deg]. Same formula as ``geom.lst_deg_from_unix``."""
    d = float(jd) - 2451545.0
    return (280.46061837 + 360.98564736629 * d + SITE_LON) % 360.0


def _lst_key(jd: float) -> int:
    """Quantize LST to 0.01 deg (≪ nside=32 tile)."""
    return int(round(_lst_deg_jd(jd) * 100.0)) % 36000


@lru_cache(maxsize=8192)
def _chart_pix(
    nside: int, nest: bool, size: int, max_zenith_deg: float, lst_key: int
) -> tuple[np.ndarray, np.ndarray]:
    """HEALPix index and usable mask for each chart pixel at this LST."""
    import healpy as hp

    alt, az = _chart_altaz(size, max_zenith_deg)
    lst = float(lst_key) * 0.01
    ra, dec = altaz_to_equatorial(alt, az, lst)
    usable = np.isfinite(alt) & np.isfinite(ra) & np.isfinite(dec)
    theta = np.deg2rad(90.0 - np.where(usable, dec, 0.0))
    phi = np.deg2rad(np.where(usable, ra, 0.0))
    pix = hp.ang2pix(nside, theta, phi, nest=nest).astype(np.int64, copy=False)
    return pix, usable


@lru_cache(maxsize=8192)
def _chart_sample_coords(
    nside: int, nest: bool, size: int, max_zenith_deg: float, lst_key: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row, col, and on-disk mask to sample a chart back onto HEALPix."""
    cy = cx = (size - 1) / 2.0
    rmax = size / 2.0 - 1.0
    ra, dec = _pix_radec(nside, nest)
    lst = float(lst_key) * 0.01
    alt, az = equatorial_to_altaz(ra, dec, lst)
    zenith = 90.0 - alt
    az_rad = np.deg2rad(az)
    radius = zenith * (rmax / max_zenith_deg)
    col = cx + radius * np.sin(az_rad)
    row = cy - radius * np.cos(az_rad)
    on = (
        np.isfinite(alt)
        & (zenith <= max_zenith_deg)
        & (alt > 0)
        & (col >= 0)
        & (col <= size - 1)
        & (row >= 0)
        & (row <= size - 1)
    )
    return row, col, on


def project_to_chart(
    field: np.ndarray,
    jd: float,
    *,
    nside: int,
    nest: bool = True,
    size: int = CHART_SIZE,
    max_zenith_deg: float = MAX_ZENITH_DEG,
) -> np.ndarray:
    """Sample a HEALPix map onto a zenith-centered alt-az chart."""
    pix, usable = _chart_pix(nside, nest, size, max_zenith_deg, _lst_key(jd))
    out = np.full((size, size), np.nan, dtype=np.float64)
    vals = field.astype(np.float64, copy=False)
    hit = usable & np.isfinite(vals[pix])
    out[hit] = vals[pix[hit]]
    return out


def sample_chart_to_healpix(
    chart: np.ndarray,
    jd: float,
    *,
    nside: int,
    nest: bool = True,
    max_zenith_deg: float = MAX_ZENITH_DEG,
) -> np.ndarray:
    """Bilinear-sample a zenithal chart back onto nested HEALPix."""
    import healpy as hp

    size = int(chart.shape[0])
    row, col, on = _chart_sample_coords(
        nside, nest, size, max_zenith_deg, _lst_key(jd)
    )
    sampled = map_coordinates(
        np.nan_to_num(chart, nan=0.0),
        np.vstack([np.where(on, row, 0.0), np.where(on, col, 0.0)]),
        order=1,
        mode="nearest",
    )
    valid_chart = np.isfinite(chart).astype(np.float64)
    weight = map_coordinates(
        valid_chart,
        np.vstack([np.where(on, row, 0.0), np.where(on, col, 0.0)]),
        order=1,
        mode="nearest",
    )
    npix = hp.nside2npix(nside)
    out = np.full(npix, np.nan, dtype=np.float64)
    good = on & (weight > 0.5)
    out[good] = sampled[good]
    return out


def _fill_nan(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Replace NaNs with the valid median; return (filled, valid_mask)."""
    valid = np.isfinite(img)
    filled = img.copy()
    if np.any(valid):
        filled[~valid] = float(np.median(img[valid]))
    else:
        filled[:] = 0.0
    return filled, valid


def _lucas_kanade(im0: np.ndarray, im1: np.ndarray, win: int) -> tuple[np.ndarray, np.ndarray]:
    """Dense Lucas–Kanade (one scale). ``u`` is +column (east), ``v`` is +row."""
    ix = sobel(im1, axis=1)
    iy = sobel(im1, axis=0)
    it = im1 - im0
    ixx = uniform_filter(ix * ix, win)
    iyy = uniform_filter(iy * iy, win)
    ixy = uniform_filter(ix * iy, win)
    ixt = uniform_filter(ix * it, win)
    iyt = uniform_filter(iy * it, win)
    det = ixx * iyy - ixy * ixy
    ok = det > LK_EPS
    u = np.zeros_like(im1)
    v = np.zeros_like(im1)
    u[ok] = (-iyy[ok] * ixt[ok] + ixy[ok] * iyt[ok]) / det[ok]
    v[ok] = (ixy[ok] * ixt[ok] - ixx[ok] * iyt[ok]) / det[ok]
    return u, v


def _warp(im: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    h, w = im.shape
    yy, xx = np.mgrid[0:h, 0:w]
    return map_coordinates(im, [yy - v, xx - u], order=1, mode="nearest")


def dense_lucas_kanade(
    im0: np.ndarray, im1: np.ndarray, *, win: int = LK_WIN, levels: int = LK_LEVELS
) -> tuple[np.ndarray, np.ndarray]:
    """Coarse-to-fine dense Lucas–Kanade. Flow is in chart pixels per frame pair."""
    pyr0 = [im0]
    pyr1 = [im1]
    for _ in range(max(levels, 1) - 1):
        pyr0.append(zoom(pyr0[-1], 0.5, order=1))
        pyr1.append(zoom(pyr1[-1], 0.5, order=1))

    u = np.zeros_like(pyr0[-1])
    v = np.zeros_like(pyr0[-1])
    for lvl in range(len(pyr0) - 1, -1, -1):
        if u.shape != pyr0[lvl].shape:
            u = zoom(u, 2.0, order=1) * 2.0
            v = zoom(v, 2.0, order=1) * 2.0
            # zoom can be 1 pixel off on odd sizes
            if u.shape != pyr0[lvl].shape:
                u = u[: pyr0[lvl].shape[0], : pyr0[lvl].shape[1]]
                v = v[: pyr0[lvl].shape[0], : pyr0[lvl].shape[1]]
        warped = _warp(pyr0[lvl], u, v)
        du, dv = _lucas_kanade(warped, pyr1[lvl], win)
        u = u + du
        v = v + dv
    return u, v


def _advect_chart(
    chart: np.ndarray, u: np.ndarray, v: np.ndarray, scale: float
) -> np.ndarray:
    """Semi-Lagrangian: look back ``scale`` frame-pairs along the flow."""
    filled, valid = _fill_nan(chart)
    h, w = chart.shape
    yy, xx = np.mgrid[0:h, 0:w]
    src_y = yy - scale * v
    src_x = xx - scale * u
    advected = map_coordinates(filled, [src_y, src_x], order=1, mode="nearest")
    wt = map_coordinates(valid.astype(np.float64), [src_y, src_x], order=1, mode="nearest")
    out = np.where(wt > 0.5, advected, np.nan)
    return out


class DenseOpticalFlow(WindowForecastModel):
    """Dense Lucas–Kanade optical flow on a zenithal chart, remapped to HEALPix."""

    name = "dense_optical_flow"

    def __init__(
        self,
        *,
        pair_offset: int = 1,
        chart_size: int = CHART_SIZE,
        max_zenith_deg: float = MAX_ZENITH_DEG,
        lk_win: int = LK_WIN,
        lk_levels: int = LK_LEVELS,
        nest: bool = True,
    ) -> None:
        self.pair_offset = int(pair_offset)
        self.chart_size = int(chart_size)
        self.max_zenith_deg = float(max_zenith_deg)
        self.lk_win = int(lk_win)
        self.lk_levels = int(lk_levels)
        self.nest = bool(nest)

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
        if field_window.shape[0] < self.pair_offset + 1 or dt_ahead_days <= 0:
            return last

        nside = _nside(last.size)
        i1, i0 = -1, -1 - self.pair_offset
        dt_obs = float(jd_window[i1] - jd_window[i0]) * 86400.0
        if dt_obs <= 1e-6:
            return last

        f0 = field_window[i0].astype(np.float64)
        f1 = last
        f0[(nobs_window[i0] < min_nobs) | ~np.isfinite(f0)] = np.nan
        f1 = f1.copy()
        f1[(nobs_window[i1] < min_nobs) | ~np.isfinite(f1)] = np.nan

        im0 = project_to_chart(
            f0,
            float(jd_window[i0]),
            nside=nside,
            nest=self.nest,
            size=self.chart_size,
            max_zenith_deg=self.max_zenith_deg,
        )
        im1 = project_to_chart(
            f1,
            float(jd_window[i1]),
            nside=nside,
            nest=self.nest,
            size=self.chart_size,
            max_zenith_deg=self.max_zenith_deg,
        )
        filled0, valid0 = _fill_nan(im0)
        filled1, valid1 = _fill_nan(im1)
        if valid0.sum() < 32 or valid1.sum() < 32:
            return last

        u, v = dense_lucas_kanade(
            filled0, filled1, win=self.lk_win, levels=self.lk_levels
        )
        scale = float(horizon_sec) / dt_obs
        advected = _advect_chart(im1, u, v, scale)
        t_target = float(jd_window[i1]) + dt_ahead_days
        pred = sample_chart_to_healpix(
            advected,
            t_target,
            nside=nside,
            nest=self.nest,
            max_zenith_deg=self.max_zenith_deg,
        )
        missing = ~np.isfinite(pred)
        pred[missing] = last[missing]
        return pred
