"""GOES-16 Derived Motion Winds over Cerro Pachon.

The Lagrangian wind baseline advects the camera field with these vectors
instead of a velocity inferred from the all-sky maps. GOES DMW is an
independent measurement of the same motion, produced by tracking features
between successive ABI scans, and it carries a *pressure* for every vector so
the wind can be selected at the altitude the clouds actually sit at.

Night-capable channels only. C02 is visible and useless in darkness; C07
(3.9 um), C14 (11.2 um) track cloud-top features, and C08/C09/C10 are water
vapour channels that report upper-tropospheric flow.

Vectors are sparse — order 10 within 100 km per channel per scan — so the whole
night across all channels is pooled to characterize the wind profile.
"""

from __future__ import annotations

import datetime as dt

import numpy as np

from analysis.satellite.goes import (
    SITE_LAT,
    SITE_LON,
    _HDF5_LOCK,
    _list_keys,
    _start_time,
    bucket_name,
)

#: Channels that produce winds at night, with a short description.
NIGHT_CHANNELS = {
    7: "3.9 um shortwave IR (cloud top)",
    8: "6.2 um upper-level water vapour",
    9: "6.9 um mid-level water vapour",
    10: "7.3 um low-level water vapour",
    14: "11.2 um IR window (cloud top)",
}


def find_wind_scans(
    start: dt.datetime,
    end: dt.datetime,
    channel: int,
    *,
    satellite: str | None = None,
    bucket: str | None = None,
) -> list[tuple[dt.datetime, str]]:
    host = bucket or bucket_name(satellite, start)
    out = []
    hour = start.replace(minute=0, second=0, microsecond=0)
    while hour <= end:
        for key in _list_keys(f"ABI-L2-DMWF/{hour:%Y/%j/%H}/", bucket=host):
            if f"C{channel:02d}_" not in key:
                continue
            t = _start_time(key)
            if start <= t <= end:
                out.append((t, key))
        hour += dt.timedelta(hours=1)
    return sorted(out)


def read_winds(
    key: str,
    radius_km: float = 300.0,
    timeout: float = 300.0,
    *,
    bucket: str | None = None,
    satellite: str | None = None,
    lat0: float | None = None,
    lon0: float | None = None,
) -> dict:
    """Wind vectors within ``radius_km`` of the summit from one DMW file.

    Returns ``u``/``v`` in m/s (meteorological convention converted to eastward
    and northward components), plus pressure, distance and quality flag.
    """
    import uuid

    import netCDF4
    import requests

    site_lat = SITE_LAT if lat0 is None else float(lat0)
    site_lon = SITE_LON if lon0 is None else float(lon0)
    host = bucket or bucket_name(satellite, _start_time(key))
    resp = requests.get(f"https://{host}.s3.amazonaws.com/{key}", timeout=timeout)
    resp.raise_for_status()

    mem_name = f"inmem-{uuid.uuid4().hex}"
    with _HDF5_LOCK:
        with netCDF4.Dataset(mem_name, mode="r", memory=resp.content) as ds:
            lat = np.ma.filled(ds["lat"][:].astype(np.float64), np.nan)
            lon = np.ma.filled(ds["lon"][:].astype(np.float64), np.nan)
            dist = np.hypot(
                (lat - site_lat) * 111.0,
                (lon - site_lon) * 111.0 * np.cos(np.deg2rad(site_lat)),
            )
            sel = dist <= radius_km
            speed = np.ma.filled(ds["wind_speed"][:].astype(np.float64), np.nan)[sel]
            direction = np.ma.filled(ds["wind_direction"][:].astype(np.float64), np.nan)[sel]
            out = {
                "lat": lat[sel],
                "lon": lon[sel],
                "dist_km": dist[sel],
                "speed": speed,
                "direction": direction,
                "pressure": np.ma.filled(
                    ds["pressure"][:].astype(np.float64), np.nan
                )[sel],
                "dqf": np.ma.filled(ds["DQF"][:].astype(np.float64), np.nan)[sel],
            }

    # Meteorological direction is where the wind blows *from*, degrees clockwise
    # from north, so the vector it travels toward is rotated by 180 degrees.
    theta = np.deg2rad(direction)
    out["u"] = -speed * np.sin(theta)  # eastward
    out["v"] = -speed * np.cos(theta)  # northward
    n = len(out["u"])
    out["time"] = np.full(n, _start_time(key).timestamp(), dtype=np.float64)
    return out


def pressure_to_height_km(pressure_hpa: np.ndarray) -> np.ndarray:
    """Approximate geometric height from pressure, US standard atmosphere.

    Good to a few hundred metres in the troposphere, which is ample for
    converting a wind level into an angular rate on the sky.
    """
    p = np.asarray(pressure_hpa, dtype=np.float64)
    return 44.3308 - 4.94654 * np.power(p * 100.0, 0.190263)
