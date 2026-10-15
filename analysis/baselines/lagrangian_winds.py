"""Lagrangian advection of the cloud field using GOES Derived Motion Winds.

Isolates the predictive value of meteorological winds versus inferred optical
motion: the latest transmission map is translated with the independent DMW
vector (u, v, height) rather than with a velocity fitted from the camera.

Advection is a semi-Lagrangian backtrace in the local alt-az / ground frame
(winds are Earth-relative). Inflow from outside the camera footprint falls
back to Eulerian persistence.
"""

from __future__ import annotations

import datetime as dt

import numpy as np

from analysis.eval_harness.window_models import WindowForecastModel
from analysis.geometry import SkyToGround, altaz_to_ground
from analysis.satellite.archive import WindArchive, dmw_path, jd_to_unix

MIN_ALT_DEG = 10.0


def _nside(npix: int) -> int:
    import healpy as hp

    return int(hp.npix2nside(npix))


def _date_keys(jd: float) -> list[str]:
    """Candidate archive keys (ISO and compact) for a Julian date."""
    unix = float(jd_to_unix(jd))
    utc = dt.datetime.fromtimestamp(unix, tz=dt.timezone.utc)
    iso = utc.strftime("%Y-%m-%d")
    compact = utc.strftime("%Y%m%d")
    # Neighbouring UTC date: a Chilean night straddles 00:00 UTC.
    prev = (utc - dt.timedelta(days=1)).strftime("%Y%m%d")
    prev_iso = (utc - dt.timedelta(days=1)).strftime("%Y-%m-%d")
    out: list[str] = []
    for key in (iso, compact, prev_iso, prev):
        if key not in out:
            out.append(key)
    return out


def load_winds_for_jd(
    jd: float,
    *,
    radius_km: float = 200.0,
    min_height_km: float = 4.0,
    cache: dict[str, WindArchive | None] | None = None,
) -> WindArchive | None:
    """Load the DMW night covering ``jd``, or ``None`` if no file exists."""
    store = cache if cache is not None else {}
    for key in _date_keys(jd):
        if key in store:
            if store[key] is not None:
                return store[key]
            continue
        path = dmw_path(key)
        if not path.is_file():
            store[key] = None
            continue
        try:
            archive = WindArchive.load(
                key, radius_km=radius_km, min_height_km=min_height_km
            )
        except Exception:  # noqa: BLE001
            store[key] = None
            continue
        store[key] = archive
        return archive
    return None


def backtrace_source_pixels(
    geo: SkyToGround,
    *,
    jd_end: float,
    jd_target: float,
    dx_km: float,
    dy_km: float,
    height_km: float,
    min_alt_deg: float = MIN_ALT_DEG,
) -> np.ndarray:
    """HEALPix source tile at ``jd_end`` for each target tile at ``jd_target``.

    A cloud currently on a target sightline arrived from
    ``(east - u Δt, north - v Δt)`` in the ground frame.
    """
    alt, az = geo.altaz(jd_target)
    east, north = altaz_to_ground(alt, az, height_km, min_alt_deg=min_alt_deg)
    return geo.ground_to_pix(east - dx_km, north - dy_km, jd_end, height_km)


class LagrangianWindAdvection(WindowForecastModel):
    """Extrapolate the last map with GOES-ABI Derived Motion Winds."""

    name = "lagrangian_wind_advection"

    def __init__(
        self,
        *,
        radius_km: float = 200.0,
        min_height_km: float = 4.0,
        min_alt_deg: float = MIN_ALT_DEG,
        nest: bool = True,
    ) -> None:
        self.radius_km = float(radius_km)
        self.min_height_km = float(min_height_km)
        self.min_alt_deg = float(min_alt_deg)
        self.nest = bool(nest)
        self._winds: dict[str, WindArchive | None] = {}
        self._geo: dict[int, SkyToGround] = {}

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
        if dt_ahead_days <= 0:
            return last

        nside = _nside(last.size)
        t_end = float(jd_window[-1])
        t_tgt = t_end + float(dt_ahead_days)
        winds = load_winds_for_jd(
            t_end,
            radius_km=self.radius_km,
            min_height_km=self.min_height_km,
            cache=self._winds,
        )
        if winds is None or winds.unix.size == 0:
            return last

        u, v, height_km = winds.at(t_end)
        if not np.isfinite(u) or not np.isfinite(v) or (abs(u) + abs(v) < 1e-6):
            return last
        if not np.isfinite(height_km) or height_km <= 0.2:
            return last

        dt_sec = float(dt_ahead_days) * 86400.0
        dx_km = u * dt_sec * 1e-3
        dy_km = v * dt_sec * 1e-3
        geo = self._geo.get(nside)
        if geo is None:
            geo = SkyToGround(nside=nside, nest=self.nest)
            self._geo[nside] = geo

        src = backtrace_source_pixels(
            geo,
            jd_end=t_end,
            jd_target=t_tgt,
            dx_km=dx_km,
            dy_km=dy_km,
            height_km=float(height_km),
            min_alt_deg=self.min_alt_deg,
        )
        pred = last.copy()
        good = (src >= 0) & np.isfinite(last[np.clip(src, 0, last.size - 1)])
        pred[good] = last[src[good]]
        return pred
