"""Time-indexed access to the GOES archive, sampled onto the all-sky grid.

Wraps the ``.npz`` files written by ``analysis.satellite.fetch_cerro_pachon`` and
exposes the one operation every model here needs: *given a Julian date and a
cloud height, return the satellite field as seen on the HEALPix sky grid.*

Causality is enforced throughout. Lookups return the most recent scan at or
before the requested time, never the nearest in absolute value, so a model
forecasting from ``t_end`` cannot see a satellite image taken after it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from analysis.geometry import (
    KM_PER_DEG_LAT,
    SkyToGround,
    ground_to_latlon,
)
from analysis.eval_harness.data import ROOT
from analysis.satellite.goes import SITE_LAT, SITE_LON, acha_npz_is_2km

DATA_DIR = ROOT / "data/satellite"
DREAM_SAT_DIR = DATA_DIR / "dream"
LEGACY_DATE = "20240220"
JD_UNIX_EPOCH = 2440587.5

#: GOES-16 sub-satellite longitude (GOES-East, 2024) and orbit radius.
SUBSAT_LON = -75.2
SUBSAT_LAT = 0.0
EARTH_R_KM = 6378.0
GEO_R_KM = 42164.0

#: ABI full-disk scan timing. Filenames carry the *start* of a 9.5 min sweep
#: that runs north to south across roughly +81 to -81 deg latitude, so the
#: instrument reaches Cerro Pachon at -30.2 deg about 69% of the way through,
#: 6.5 min after the nominal start. Treating the start time as the observation
#: time therefore invents ~6.5 min of latency, and advecting to correct for it
#: displaces the field by ~7 km in the wrong direction.
SCAN_DURATION_SEC = 570.0
SITE_SCAN_OFFSET_SEC = 390.0
#: A scan cannot be used before it has finished and been processed. Without
#: this, a forecast issued at 02:05 would consume the scan starting 02:00,
#: which does not exist yet.
SCAN_LATENCY_SEC = 630.0


def jd_to_unix(jd: float | np.ndarray) -> np.ndarray:
    return (np.asarray(jd, dtype=np.float64) - JD_UNIX_EPOCH) * 86400.0


@dataclass
class Grid:
    """One gridded GOES product over the night."""

    values: np.ndarray  # (n_time, ny, nx)
    unix: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    var: str

    @property
    def lat0(self) -> float:
        return float(self.lat[0, 0])

    def jd_of(self, idx: int) -> float:
        """Julian date at which ABI actually observed the site in scan ``idx``."""
        return (
            float(self.unix[idx]) + SITE_SCAN_OFFSET_SEC
        ) / 86400.0 + JD_UNIX_EPOCH

    def index_at(self, jd: float) -> int:
        """Most recent scan *available* at ``jd`` (clamped to the first scan).

        Availability, not observation time: a scan is usable only once the
        sweep has finished and been processed.
        """
        t = float(jd_to_unix(jd)) - SCAN_LATENCY_SEC
        i = int(np.searchsorted(self.unix, t, side="right") - 1)
        return max(i, 0)

    def sample_latlon(self, lat: np.ndarray, lon: np.ndarray, idx: int) -> np.ndarray:
        """Nearest-neighbour sample of scan ``idx`` at given lat/lon."""
        # The cutout is small enough that the grid is very nearly regular in
        # lat/lon, so nearest-neighbour reduces to index arithmetic.
        ny, nx = self.values.shape[1:]
        dlat = (self.lat[-1, 0] - self.lat[0, 0]) / max(ny - 1, 1)
        dlon = (self.lon[0, -1] - self.lon[0, 0]) / max(nx - 1, 1)
        finite = np.isfinite(lat) & np.isfinite(lon)
        j = np.rint(np.where(finite, lat - self.lat[0, 0], 0.0) / dlat).astype(int)
        i = np.rint(np.where(finite, lon - self.lon[0, 0], 0.0) / dlon).astype(int)
        inside = (j >= 0) & (j < ny) & (i >= 0) & (i < nx) & finite
        out = np.full(np.shape(lat), np.nan, dtype=np.float64)
        jj, ii = np.clip(j, 0, ny - 1), np.clip(i, 0, nx - 1)
        vals = self.values[idx][jj, ii]
        return np.where(inside, vals, out)


def grid_path(product: str, date: str = LEGACY_DATE) -> Path:
    """Path to one gridded product: FITS-night ``goes16_*`` or a DREAM folder."""
    if "-" in date:
        return DREAM_SAT_DIR / date / f"{product}.npz"
    return DATA_DIR / f"goes16_{product}_{date}.npz"


def load_grid(product: str, date: str = LEGACY_DATE) -> Grid:
    path = grid_path(product, date)
    d = np.load(path, allow_pickle=True)
    return Grid(
        values=d["values"].astype(np.float64),
        unix=d["unix"].astype(np.float64),
        lat=d["lat"].astype(np.float64),
        lon=d["lon"].astype(np.float64),
        var=str(d["var"]),
    )


def dmw_path(date: str = LEGACY_DATE) -> Path:
    if "-" in date:
        return DREAM_SAT_DIR / date / "dmw.npz"
    return DATA_DIR / f"goes16_dmw_{date}.npz"


def available_dream_nights(*, products: tuple[str, ...] = ("c13", "acha", "dmw")) -> list[str]:
    """DREAM dates that have every requested satellite product on disk."""
    if not DREAM_SAT_DIR.is_dir():
        return []
    nights = []
    for p in sorted(DREAM_SAT_DIR.iterdir()):
        if not p.is_dir() or p.name.count("-") != 2:
            continue
        ok = True
        for prod in products:
            f = p / f"{prod}.npz"
            if not f.is_file():
                ok = False
                break
            if prod == "acha" and not acha_npz_is_2km(f):
                ok = False
                break
        if ok:
            nights.append(p.name)
    return nights


def dream_archive_ready(dates: list[str] | None = None) -> bool:
    """True if the DREAM GOES archive covers ``dates`` (or any night)."""
    have = set(available_dream_nights())
    if not have:
        return False
    if dates is None:
        return True
    return set(dates) <= have


def parallax_offset(
    lat: np.ndarray, lon: np.ndarray, height_km: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Offset from a cloud's true position to where GOES reports it, in km.

    A geostationary imager georeferences each pixel by intersecting its line of
    sight with the ellipsoid, so a cloud at altitude is reported *beyond* its
    true position, displaced directly away from the sub-satellite point by
    ``h tan(za)``. Over Cerro Pachon the satellite zenith angle is 35.6 deg, so
    the error runs from 1.4 km at 2 km altitude to 8.6 km at 12 km — one to
    three pixels, and comparable to the structures being matched.

    Add the returned offset to a true position to get the position at which to
    read the GOES grid.
    """
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    phi, lam = np.deg2rad(lat), np.deg2rad(lon)
    phi_s, lam_s = np.deg2rad(SUBSAT_LAT), np.deg2rad(SUBSAT_LON)

    # Great-circle angle to the sub-satellite point, and the satellite zenith
    # angle it implies at the surface.
    cos_d = np.sin(phi) * np.sin(phi_s) + np.cos(phi) * np.cos(phi_s) * np.cos(
        lam_s - lam
    )
    d = np.arccos(np.clip(cos_d, -1.0, 1.0))
    za = np.arctan2(GEO_R_KM * np.sin(d), GEO_R_KM * np.cos(d) - EARTH_R_KM)

    # Bearing from the point toward the sub-satellite point; displacement is
    # 180 deg from it.
    y = np.sin(lam_s - lam) * np.cos(phi_s)
    x = np.cos(phi) * np.sin(phi_s) - np.sin(phi) * np.cos(phi_s) * np.cos(
        lam_s - lam
    )
    bearing = np.arctan2(y, x) + np.pi

    dist = np.asarray(height_km, dtype=np.float64) * np.tan(za)
    return dist * np.sin(bearing), dist * np.cos(bearing)


@dataclass
class WindArchive:
    """Derived motion winds, reduced to a single layer-mean vector per scan."""

    unix: np.ndarray
    u: np.ndarray  # eastward, m/s
    v: np.ndarray  # northward, m/s
    height_km: np.ndarray

    @classmethod
    def load(
        cls,
        date: str = LEGACY_DATE,
        radius_km: float = 200.0,
        min_height_km: float = 4.0,
    ) -> WindArchive:
        """Pool vectors per scan time, keeping those above ``min_height_km``.

        Low-level vectors are excluded because the transmission-relevant cloud
        on this night is the ice layer at 8-10 km; including a stray boundary
        layer vector would bias the mean badly given how few vectors exist.
        """
        from analysis.satellite.winds import pressure_to_height_km

        d = np.load(dmw_path(date))
        h = pressure_to_height_km(d["pressure"])
        keep = (d["dist_km"] <= radius_km) & (h >= min_height_km) & (d["dqf"] == 0)

        times = np.unique(d["time"][keep])
        u, v, hh = [], [], []
        for t in times:
            m = keep & (d["time"] == t)
            u.append(float(np.mean(d["u"][m])))
            v.append(float(np.mean(d["v"][m])))
            hh.append(float(np.median(h[m])))
        return cls(
            unix=times,
            u=np.array(u),
            v=np.array(v),
            height_km=np.array(hh),
        )

    def at(self, jd: float) -> tuple[float, float, float]:
        """Most recent wind at or before ``jd`` as ``(u, v, height_km)`` in m/s."""
        t = float(jd_to_unix(jd))
        i = max(int(np.searchsorted(self.unix, t, side="right") - 1), 0)
        return float(self.u[i]), float(self.v[i]), float(self.height_km[i])


#: Do not use a scan from another night: ABI cadence is 10 min, so 30 min is
#: already three missed cycles and almost certainly a daytime/gap crossing.
MAX_SCAN_AGE_SEC = 1800.0
MAX_WIND_AGE_SEC = 4 * 3600.0
_GRID_CACHE_SIZE = 6


class GoesCatalog:
    """Night-keyed GOES archive covering DREAM dates (and the FITS night).

    ``SkySampler`` talks to a ``Grid``; :class:`CatalogGrid` is a facade that
    routes each Julian date to the matching night so concatenated DREAM cubes
    cannot leak a previous night's last scan across a multi-day gap.
    """

    def __init__(self, dates: list[str] | None = None) -> None:
        if dates is None:
            dates = available_dream_nights()
        if not dates:
            raise FileNotFoundError(
                f"No DREAM GOES nights under {DREAM_SAT_DIR}. "
                "Run python -m analysis.satellite.fetch_dream"
            )
        self.dates = list(dates)
        self._ranges: dict[str, tuple[float, float]] = {}
        self._grids: dict[tuple[str, str], Grid] = {}
        self._winds: dict[str, WindArchive] = {}
        self._order: list[tuple[str, str]] = []
        for date in self.dates:
            path = grid_path("c13", date)
            if not path.is_file():
                continue
            unix = np.load(path)["unix"].astype(np.float64)
            if unix.size:
                self._ranges[date] = (float(unix.min()), float(unix.max()))

    def night_at(self, jd: float) -> str | None:
        t = float(jd_to_unix(jd))
        hits = [
            date
            for date, (a, b) in self._ranges.items()
            if a - 600.0 <= t <= b + 600.0
        ]
        if not hits:
            return None
        if len(hits) == 1:
            return hits[0]

        def _dist(date: str) -> float:
            a, b = self._ranges[date]
            if a <= t <= b:
                return 0.0
            return min(abs(t - a), abs(t - b))

        return min(hits, key=_dist)

    def grid(self, product: str, date: str) -> Grid:
        key = (product, date)
        if key in self._grids:
            return self._grids[key]
        grid = load_grid(product, date)
        self._grids[key] = grid
        self._order.append(key)
        if len(self._order) > _GRID_CACHE_SIZE:
            old = self._order.pop(0)
            self._grids.pop(old, None)
        return grid

    def wind(self, date: str, min_height_km: float = 4.0) -> WindArchive:
        if date not in self._winds:
            self._winds[date] = WindArchive.load(date, min_height_km=min_height_km)
        return self._winds[date]

    def catalog_grid(self, product: str) -> "CatalogGrid":
        return CatalogGrid(self, product)

    def catalog_wind(self, min_height_km: float = 4.0) -> "CatalogWind":
        return CatalogWind(self, min_height_km=min_height_km)


class CatalogGrid:
    """``Grid`` facade that binds ``index_at`` / ``sample_latlon`` to one night."""

    def __init__(self, catalog: GoesCatalog, product: str) -> None:
        self.catalog = catalog
        self.product = product
        self._active: Grid | None = None
        self._stale = False
        self.var = product

    def _bind(self, jd: float) -> Grid | None:
        date = self.catalog.night_at(jd)
        if date is None:
            self._stale = True
            if self.catalog.dates:
                self._active = self.catalog.grid(self.product, self.catalog.dates[0])
            return self._active
        self._active = self.catalog.grid(self.product, date)
        return self._active

    def index_at(self, jd: float) -> int:
        grid = self._bind(jd)
        if grid is None:
            self._stale = True
            return 0
        i = grid.index_at(jd)
        t = float(jd_to_unix(jd))
        age = t - float(grid.unix[i]) - SCAN_LATENCY_SEC
        self._stale = bool(age > MAX_SCAN_AGE_SEC)
        return i

    def jd_of(self, idx: int) -> float:
        if self._active is None:
            raise RuntimeError("index_at must be called before jd_of")
        return self._active.jd_of(idx)

    def sample_latlon(self, lat: np.ndarray, lon: np.ndarray, idx: int) -> np.ndarray:
        if self._active is None or self._stale:
            return np.full(np.shape(lat), np.nan, dtype=np.float64)
        return self._active.sample_latlon(lat, lon, idx)


class CatalogWind:
    """Wind facade that will not return a vector from a different night."""

    def __init__(self, catalog: GoesCatalog, min_height_km: float = 4.0) -> None:
        self.catalog = catalog
        self.min_height_km = min_height_km

    def at(self, jd: float) -> tuple[float, float, float]:
        date = self.catalog.night_at(jd)
        if date is None:
            return 0.0, 0.0, 8.0
        wind = self.catalog.wind(date, min_height_km=self.min_height_km)
        if wind.unix.size == 0:
            return 0.0, 0.0, 8.0
        u, v, h = wind.at(jd)
        t = float(jd_to_unix(jd))
        i = max(int(np.searchsorted(wind.unix, t, side="right") - 1), 0)
        if t - float(wind.unix[i]) > MAX_WIND_AGE_SEC:
            return 0.0, 0.0, 8.0
        return u, v, h


class SkySampler:
    """Samples gridded GOES products onto the HEALPix sky grid."""

    def __init__(self, nside: int = 16, nest: bool = True) -> None:
        self.geo = SkyToGround(nside, nest)
        self.npix = self.geo.ra.size

    def solve_heights(
        self,
        acha: Grid,
        jd: float,
        fallback_km: float,
        *,
        min_alt_deg: float = 15.0,
        parallax: bool = True,
        n_iter: int = 3,
        idx: int | None = None,
    ) -> np.ndarray:
        """Per-tile cloud-top height along each sightline, in km.

        A single plane-parallel layer is a poor description of a deck spanning
        3.3-9.9 km, but the correct height for a sightline depends on where that
        sightline lands, which depends on the height. The fixed point is found
        by iterating; it converges in two or three passes because the mapping is
        a mild contraction at these geometries.

        Tiles where the retrieval reports no cloud fall back to ``fallback_km``;
        the height is immaterial there because clear air does not attenuate.
        """
        i = acha.index_at(jd) if idx is None else idx
        h = np.full(self.npix, float(fallback_km))
        for _ in range(n_iter):
            east, north = self.geo.footprint(jd, h, min_alt_deg=min_alt_deg)
            lat, lon = ground_to_latlon(east, north)
            if parallax:
                de, dn = parallax_offset(lat, lon, h)
                lat, lon = ground_to_latlon(east + de, north + dn)
            metres = acha.sample_latlon(lat, lon, i)
            h = np.where(np.isfinite(metres), metres / 1000.0, fallback_km)
            h = np.clip(h, 0.5, 18.0)
        return h

    def project(
        self,
        grid: Grid,
        jd: float,
        height_km,
        *,
        min_alt_deg: float = 15.0,
        idx: int | None = None,
        parallax: bool = False,
    ) -> np.ndarray:
        """Satellite field as seen on the sky at ``jd``, shape ``(npix,)``.

        ``height_km`` may be a scalar layer or a per-tile array from
        :meth:`solve_heights`.
        """
        east, north = self.geo.footprint(jd, height_km, min_alt_deg=min_alt_deg)
        lat, lon = ground_to_latlon(east, north)
        if parallax:
            de, dn = parallax_offset(lat, lon, height_km)
            lat, lon = ground_to_latlon(east + de, north + dn)
        i = grid.index_at(jd) if idx is None else idx
        return grid.sample_latlon(lat, lon, i)

    def project_shifted(
        self,
        grid: Grid,
        jd_view: float,
        height_km,
        *,
        shift_east_km: float,
        shift_north_km: float,
        min_alt_deg: float = 15.0,
        idx: int | None = None,
        parallax: bool = False,
    ) -> np.ndarray:
        """As :meth:`project`, but reading the grid at a translated position.

        This is how advection is applied: rather than warping a field on the
        sphere, the *sampling point* is displaced by the wind, which keeps the
        operation exact and avoids resampling twice.
        """
        east, north = self.geo.footprint(jd_view, height_km, min_alt_deg=min_alt_deg)
        east, north = east + shift_east_km, north + shift_north_km
        lat, lon = ground_to_latlon(east, north)
        if parallax:
            de, dn = parallax_offset(lat, lon, height_km)
            lat, lon = ground_to_latlon(east + de, north + dn)
        i = grid.index_at(jd_view) if idx is None else idx
        return grid.sample_latlon(lat, lon, i)
