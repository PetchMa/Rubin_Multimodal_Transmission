"""Nearby night-time GOES boxes for satellite nowcast pretraining.

Cerro Pachón is held out of the pretrain *train* split so we can measure
transfer onto the camera site. Other boxes are Andes observatories and
inflow regions a few hundred km away (marine west, Andes east, Atacama
north, Elqui south). Half-width is the same ±1.5° (~165 km) used for the
camera GOES cutout.
"""

from __future__ import annotations

import datetime as dt
import threading
from dataclasses import dataclass

import numpy as np

from analysis.satellite.goes import SITE_ALT_M, SITE_LAT, SITE_LON

_SUN_LOCK = threading.Lock()
#: Drop windows with the moon up and at least this illuminated fraction.
MOON_ILLUM_MAX = 0.25
#: Geometric altitude (deg) above which a bright moon counts as up.
MOON_ALT_MIN_DEG = 0.0
_MOON_UNIQUE_SEC = 60.0
_MOON_CHUNK = 4096


@dataclass(frozen=True)
class Site:
    name: str
    lat: float
    lon: float
    kind: str
    note: str = ""
    pretrain_split: str = "train"  # train | val | test


SITES: tuple[Site, ...] = (
    Site(
        "pachon",
        SITE_LAT,
        SITE_LON,
        "observatory",
        "Rubin / Cerro Pachón — camera site, transfer holdout",
        pretrain_split="val",
    ),
    Site(
        "las_campanas",
        -29.0146,
        -70.6926,
        "observatory",
        "~140 km N, similar Atacama climate — unseen-site test",
        pretrain_split="test",
    ),
    Site(
        "paranal",
        -24.6272,
        -70.4042,
        "observatory",
        "~620 km N, VLT, drier / higher",
    ),
    Site(
        "copiapo",
        -27.3665,
        -70.3323,
        "atacama",
        "~320 km N, Atacama corridor",
    ),
    Site(
        "elqui_south",
        -32.80,
        -70.80,
        "andes",
        "~280 km S toward Santiago / Elqui",
    ),
    Site(
        "pacific_west",
        -30.25,
        -73.80,
        "marine",
        "~300 km W, marine stratocu inflow toward Pachón",
    ),
    Site(
        "andes_east",
        -30.25,
        -67.80,
        "andes",
        "~280 km E, Argentine Andes / lee",
    ),
)

SITE_BY_NAME = {s.name: s for s in SITES}


def train_sites() -> tuple[Site, ...]:
    return tuple(s for s in SITES if s.pretrain_split == "train")


def val_sites() -> tuple[Site, ...]:
    return tuple(s for s in SITES if s.pretrain_split == "val")


def test_sites() -> tuple[Site, ...]:
    return tuple(s for s in SITES if s.pretrain_split == "test")


def sun_altitude_deg(
    lat: float, lon: float, times: list[dt.datetime] | np.ndarray
) -> np.ndarray:
    """Apparent solar altitude (deg) at ``lat``/``lon`` for UTC times."""
    import astropy.units as u
    from astropy.coordinates import AltAz, EarthLocation, get_sun
    from astropy.time import Time

    loc = EarthLocation(lat=float(lat) * u.deg, lon=float(lon) * u.deg)
    t = Time(times)
    with _SUN_LOCK:
        alt = get_sun(t).transform_to(AltAz(obstime=t, location=loc)).alt.deg
    return np.asarray(alt, dtype=np.float64)


def moon_altitude_illumination(
    unix: np.ndarray,
    *,
    lat: float = SITE_LAT,
    lon: float = SITE_LON,
    alt_m: float = SITE_ALT_M,
) -> tuple[np.ndarray, np.ndarray]:
    """Moon altitude (deg) and illuminated fraction at ``lat``/``lon``.

    ``unix`` is UTC seconds. Illumination is the elongation approximation
    ``(1 − cos ψ) / 2`` (0 at new moon, 1 at full). Times are unique-rounded
    to 60 s — the moon moves ~0.5°/h, well inside the 0° altitude cut.
    """
    import astropy.units as u
    from astropy.coordinates import AltAz, EarthLocation, get_body, get_sun
    from astropy.time import Time

    unix = np.asarray(unix, dtype=np.float64).reshape(-1)
    alt = np.full(unix.shape, np.nan, dtype=np.float64)
    illum = np.full(unix.shape, np.nan, dtype=np.float64)
    ok = np.isfinite(unix) & (unix > 0)
    if not bool(ok.any()):
        return alt, illum

    rounded = np.round(unix[ok] / _MOON_UNIQUE_SEC) * _MOON_UNIQUE_SEC
    uniq, inv = np.unique(rounded, return_inverse=True)
    u_alt = np.empty(uniq.size, dtype=np.float64)
    u_illum = np.empty(uniq.size, dtype=np.float64)
    loc = EarthLocation(
        lat=float(lat) * u.deg,
        lon=float(lon) * u.deg,
        height=float(alt_m) * u.m,
    )
    for i0 in range(0, uniq.size, _MOON_CHUNK):
        sl = slice(i0, min(i0 + _MOON_CHUNK, uniq.size))
        t = Time(uniq[sl], format="unix")
        with _SUN_LOCK:
            moon = get_body("moon", t)
            sun = get_sun(t)
            moon_alt = moon.transform_to(AltAz(obstime=t, location=loc)).alt.deg
            elong = sun.separation(moon)
        u_alt[sl] = np.asarray(moon_alt, dtype=np.float64)
        u_illum[sl] = 0.5 * (1.0 - np.cos(np.asarray(elong.radian, dtype=np.float64)))

    alt[ok] = u_alt[inv]
    illum[ok] = u_illum[inv]
    return alt, illum


def moon_is_bright(
    alt_deg: np.ndarray,
    illumination: np.ndarray,
    *,
    max_illum: float = MOON_ILLUM_MAX,
    min_alt_deg: float = MOON_ALT_MIN_DEG,
) -> np.ndarray:
    """True where lunar scatter would bias camera extinction.

    Bright = illumination ≥ ``max_illum`` and altitude > ``min_alt_deg``.
    Invalid ephemeris is treated as bright so those windows are dropped.
    """
    alt = np.asarray(alt_deg, dtype=np.float64)
    illum = np.asarray(illumination, dtype=np.float64)
    bright = (illum >= float(max_illum)) & (alt > float(min_alt_deg))
    return bright | ~np.isfinite(alt) | ~np.isfinite(illum)


def night_window(
    site: Site,
    evening: dt.date,
    *,
    max_alt_deg: float = -12.0,
    step_min: int = 10,
) -> tuple[dt.datetime, dt.datetime] | None:
    """Nautical-night span starting on ``evening`` (UTC search 18:00 → +24 h)."""
    start = dt.datetime(
        evening.year, evening.month, evening.day, 18, 0, tzinfo=dt.timezone.utc
    )
    n = int(24 * 60 / step_min)
    times = [start + dt.timedelta(minutes=step_min * i) for i in range(n + 1)]
    alt = sun_altitude_deg(site.lat, site.lon, times)
    dark = alt < float(max_alt_deg)
    if not bool(dark.any()):
        return None
    # Longest contiguous dark run (handles polar-adjacent glitches; at 30S
    # there is one night per search window).
    best_i = best_j = best_n = 0
    i = 0
    while i < len(dark):
        if not dark[i]:
            i += 1
            continue
        j = i
        while j < len(dark) and dark[j]:
            j += 1
        if j - i > best_n:
            best_i, best_j, best_n = i, j, j - i
        i = j
    if best_n < 6:  # < 1 h
        return None
    return times[best_i], times[best_j - 1]
