"""Geometry linking the all-sky sphere to the satellite ground grid.

The two modalities live in incompatible frames.

**The all-sky maps are in equatorial coordinates.** This was verified rather
than assumed: the observed footprint is a cap of ~70 deg radius (altitude
> 20 deg) whose centroid sits at declination -30.2 deg, the site latitude, and
whose right ascension tracks local sidereal time exactly across the night. A
tile therefore corresponds to a *different* direction above the observer at
every instant, and the Earth's rotation alone sweeps the field of view by
15 deg/hour.

The chain implemented here is:

    HEALPix tile -> (RA, Dec) -> (altitude, azimuth) at time t
                 -> sightline intersection with a cloud layer at height h
                 -> ground offset from the summit -> (lat, lon) -> GOES pixel

and its inverse. All of it is exact geometry with no fitted parameters.
"""

from __future__ import annotations

import numpy as np

from analysis.satellite.goes import SITE_LAT, SITE_LON

KM_PER_DEG_LAT = 111.195


def lst_deg(jd: float | np.ndarray) -> np.ndarray:
    """Local mean sidereal time at Cerro Pachon, in degrees."""
    from astropy.time import Time

    t = Time(np.atleast_1d(np.asarray(jd, dtype=np.float64)), format="jd")
    return np.asarray(
        t.sidereal_time("mean", longitude=SITE_LON).deg, dtype=np.float64
    )


def equatorial_to_altaz(
    ra_deg: np.ndarray, dec_deg: np.ndarray, lst: float
) -> tuple[np.ndarray, np.ndarray]:
    """(RA, Dec) to (altitude, azimuth) in degrees; azimuth from north, east +ve."""
    phi = np.deg2rad(SITE_LAT)
    ha = np.deg2rad(lst - np.asarray(ra_deg, dtype=np.float64))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=np.float64))

    sin_alt = np.sin(dec) * np.sin(phi) + np.cos(dec) * np.cos(phi) * np.cos(ha)
    sin_alt = np.clip(sin_alt, -1.0, 1.0)
    alt = np.arcsin(sin_alt)

    y = -np.cos(dec) * np.sin(ha)
    x = (np.sin(dec) - np.sin(phi) * sin_alt) / max(np.cos(phi), 1e-12)
    az = np.arctan2(y, x)
    return np.rad2deg(alt), np.rad2deg(az) % 360.0


def altaz_to_equatorial(
    alt_deg: np.ndarray, az_deg: np.ndarray, lst: float
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`equatorial_to_altaz`."""
    phi = np.deg2rad(SITE_LAT)
    alt = np.deg2rad(np.asarray(alt_deg, dtype=np.float64))
    az = np.deg2rad(np.asarray(az_deg, dtype=np.float64))

    sin_dec = np.sin(alt) * np.sin(phi) + np.cos(alt) * np.cos(phi) * np.cos(az)
    sin_dec = np.clip(sin_dec, -1.0, 1.0)
    dec = np.arcsin(sin_dec)

    y = -np.cos(alt) * np.sin(az)
    x = (np.sin(alt) - np.sin(phi) * sin_dec) / max(np.cos(phi), 1e-12)
    ha = np.arctan2(y, x)
    return (lst - np.rad2deg(ha)) % 360.0, np.rad2deg(dec)


def altaz_to_ground(
    alt_deg: np.ndarray, az_deg: np.ndarray, height_km: float, min_alt_deg: float = 10.0
) -> tuple[np.ndarray, np.ndarray]:
    """Where a sightline pierces a cloud layer, as km east/north of the summit.

    Sightlines below ``min_alt_deg`` are returned as NaN: the horizontal
    distance diverges as altitude goes to zero, and a plane-parallel cloud layer
    stops being a sensible model long before the horizon.
    """
    alt = np.deg2rad(np.asarray(alt_deg, dtype=np.float64))
    az = np.deg2rad(np.asarray(az_deg, dtype=np.float64))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = height_km / np.tan(alt)
    bad = np.asarray(alt_deg) < min_alt_deg
    east = np.where(bad, np.nan, r * np.sin(az))
    north = np.where(bad, np.nan, r * np.cos(az))
    return east, north


def ground_to_altaz(
    east_km: np.ndarray, north_km: np.ndarray, height_km: float
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`altaz_to_ground`."""
    r = np.hypot(east_km, north_km)
    alt = np.rad2deg(np.arctan2(height_km, r))
    az = np.rad2deg(np.arctan2(east_km, north_km)) % 360.0
    return alt, az


def ground_to_latlon(
    east_km: np.ndarray, north_km: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    lat = SITE_LAT + north_km / KM_PER_DEG_LAT
    lon = SITE_LON + east_km / (KM_PER_DEG_LAT * np.cos(np.deg2rad(SITE_LAT)))
    return lat, lon


def pix_radec(nside: int, nest: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """(RA, Dec) in degrees for every HEALPix tile."""
    import healpy as hp

    theta, phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), nest=nest)
    return np.rad2deg(phi), 90.0 - np.rad2deg(theta)


class SkyToGround:
    """Caches the tile -> sightline mapping for one HEALPix grid."""

    def __init__(self, nside: int = 16, nest: bool = True) -> None:
        self.nside = nside
        self.nest = nest
        self.ra, self.dec = pix_radec(nside, nest)

    def altaz(self, jd: float) -> tuple[np.ndarray, np.ndarray]:
        return equatorial_to_altaz(self.ra, self.dec, float(lst_deg(jd)[0]))

    def footprint(
        self, jd: float, height_km: float, min_alt_deg: float = 10.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Ground position (east, north in km) sampled by each tile at ``jd``."""
        alt, az = self.altaz(jd)
        return altaz_to_ground(alt, az, height_km, min_alt_deg=min_alt_deg)

    def ground_to_pix(
        self, east_km: np.ndarray, north_km: np.ndarray, jd: float, height_km: float
    ) -> np.ndarray:
        """HEALPix tile that observes a given ground position at ``jd``.

        Returns -1 where the position maps below the horizon.
        """
        import healpy as hp

        alt, az = ground_to_altaz(east_km, north_km, height_km)
        ra, dec = altaz_to_equatorial(alt, az, float(lst_deg(jd)[0]))
        usable = np.isfinite(alt) & np.isfinite(dec) & np.isfinite(ra) & (alt > 0)
        theta = np.deg2rad(90.0 - np.where(usable, dec, 0.0))
        phi = np.deg2rad(np.where(usable, ra, 0.0))
        pix = hp.ang2pix(self.nside, theta, phi, nest=self.nest)
        return np.where(usable, pix, -1)
