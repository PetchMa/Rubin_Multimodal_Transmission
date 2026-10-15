"""GOES-ABI cloud fields over Cerro Pachon, from the NOAA public archive.

The all-sky camera measures transmission along every line of sight from the
summit. A geostationary imager sees the same clouds from above, so GOES gives an
independent, absolutely-georeferenced view of the field the camera is looking
through. That makes it useful as ground truth for cloud advection, as a source
of predictors, and as a sanity check on the transmission maps themselves.

Access is anonymous HTTPS against the ``noaa-goes16`` S3 bucket; no credentials
and no AWS SDK are needed.

Two practical constraints shape this module:

- **Chile is only on the full disk.** CONUS and mesoscale sectors do not reach
  30 S, so the cadence is 10 min and each file covers the whole hemisphere.
- **The data is at night.** Reflectance-based products (notably cloud optical
  depth) are retrieved from visible channels and are unavailable or degraded in
  darkness. The products used here are the ones that work on thermal emission:
  the clear-sky mask, cloud-top height/temperature, and the 10.3 um window
  brightness temperature.

To avoid downloading 5424x5424 grids in full, the *forward* geolocation is used:
a target latitude/longitude box is converted to ABI scan angles, and only the
enclosing slice is read out of the file.
"""

from __future__ import annotations

import datetime as dt
import re
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Rubin Observatory / Cerro Pachon summit.
SITE_LAT = -30.24463
SITE_LON = -70.74942
SITE_ALT_M = 2647.0

#: Public NOAA Open Data buckets. GOES-19 became operational GOES-East on
#: 2025-04-07; DREAM nights (2025-06 → 2026-05) therefore live in ``goes19``.
SATELLITES = {
    "goes16": "noaa-goes16",
    "goes19": "noaa-goes19",
}
GOES19_EAST = dt.datetime(2025, 4, 7, 15, 10, tzinfo=dt.timezone.utc)
BUCKET = SATELLITES["goes16"]  # default: the 2024-02-20 FITS night

_HDF5_LOCK = threading.Lock()
_TLS = threading.local()


def _http():
    """Per-thread ``requests.Session`` so parallel GETs reuse connections."""
    import requests
    from requests.adapters import HTTPAdapter

    sess = getattr(_TLS, "session", None)
    if sess is None:
        sess = requests.Session()
        adapter = HTTPAdapter(pool_connections=8, pool_maxsize=8, max_retries=2)
        sess.mount("https://", adapter)
        sess.mount("http://", adapter)
        _TLS.session = sess
    return sess

#: 2 km ACHA2KMF cutout at ±1.5° is ~135×149. Legacy ACHAF is ~28×32.
ACHA_2KM_PRODUCT = "ABI-L2-ACHA2KMF"
ACHA_10KM_PRODUCT = "ABI-L2-ACHAF"
ACHA_2KM_MIN_AXIS = 80

#: Products that carry usable information at night, keyed by short name.
#: C07/C15 are the same CMIPF stream as C13 (channel number is in the key).
#: ``acha`` is the 2 km full-disk height product (same 10 min cadence as C13).
#: ``acha2km`` is a backward-compatible alias for the same files.
PRODUCTS = {
    "acm": ("ABI-L2-ACMF", "BCM"),      # binary cloud mask
    "acha": (ACHA_2KM_PRODUCT, "HT"),   # cloud top height (2 km)
    "acha2km": (ACHA_2KM_PRODUCT, "HT"),
    "actp": ("ABI-L2-ACTPF", "Phase"),  # 0 clear / 1 liquid / 2 S/C / 3 mixed / 4 ice
    "c07": ("ABI-L2-CMIPF", "CMI"),     # 3.9 um shortwave IR (night stratus)
    "c08": ("ABI-L2-CMIPF", "CMI"),     # 6.2 um upper-level water vapour
    "c09": ("ABI-L2-CMIPF", "CMI"),     # 6.9 um mid-level water vapour
    "c10": ("ABI-L2-CMIPF", "CMI"),     # 7.3 um low/mid water vapour
    "c13": ("ABI-L2-CMIPF", "CMI"),     # 10.3 um clean window brightness temp
    "c14": ("ABI-L2-CMIPF", "CMI"),     # 11.2 um IR window
    "c15": ("ABI-L2-CMIPF", "CMI"),     # 12.3 um dirty window (thin-ice proxy)
    "c16": ("ABI-L2-CMIPF", "CMI"),     # 13.3 um CO2
    "c11": ("ABI-L2-CMIPF", "CMI"),     # 8.4 um
    "c12": ("ABI-L2-CMIPF", "CMI"),     # 9.6 um ozone
    "lst": ("ABI-L2-LSTF", "LST"),      # land skin temperature (hourly)
    "lst2km": ("ABI-L2-LST2KMF", "LST"),  # 2 km land skin temperature
    "tpw": ("ABI-L2-TPWF", "TPW"),      # total precipitable water
    "acht": ("ABI-L2-ACHTF", "TEMP"),   # cloud-top temperature
    "ctp": ("ABI-L2-CTPF", "PRES"),     # cloud-top pressure (~10 km)
    "achp": ("ABI-L2-ACHP2KMF", "PRES"),  # cloud-top pressure (2 km)
    "sst": ("ABI-L2-SSTF", "SST"),      # sea-surface temperature
    "rrqpe": ("ABI-L2-RRQPEF", "RRQPE"),  # IR rainfall rate
    "dsi_li": ("ABI-L2-DSIF", "LI"),    # lifted index
    "dsi_cape": ("ABI-L2-DSIF", "CAPE"),
    "cod": ("ABI-L2-COD2KMF", "COD"),   # cloud optical depth (day/twilight)
    "cps": ("ABI-L2-CPSF", "CPS"),      # cloud particle size
}


def acha_npz_is_2km(path: str | Path) -> bool:
    """True if ``acha.npz`` is ACHA2KMF (~135×149), not legacy 10 km ACHAF."""
    path = Path(path)
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=True) as z:
            if "product" in z.files:
                prod = str(np.asarray(z["product"]).reshape(-1)[0])
                if prod == ACHA_2KM_PRODUCT:
                    return True
                if prod == ACHA_10KM_PRODUCT:
                    return False
            shape = tuple(z["values"].shape)
            return shape[-2] >= ACHA_2KM_MIN_AXIS and shape[-1] >= ACHA_2KM_MIN_AXIS
    except Exception:  # noqa: BLE001
        return False


def satellite_id(when: dt.datetime) -> str:
    """``goes16`` or ``goes19`` for a UTC time at the GOES-East slot."""
    if when.tzinfo is None:
        when = when.replace(tzinfo=dt.timezone.utc)
    return "goes19" if when >= GOES19_EAST else "goes16"


def bucket_name(
    satellite: str | None = None, when: dt.datetime | None = None
) -> str:
    if satellite is not None:
        return SATELLITES[satellite]
    if when is not None:
        return SATELLITES[satellite_id(when)]
    return BUCKET


def _list_keys(
    prefix: str, timeout: float = 60.0, *, bucket: str | None = None
) -> list[str]:
    """List every object under ``prefix``, following continuation tokens."""
    import urllib.parse

    host = bucket or BUCKET
    keys: list[str] = []
    token = None
    while True:
        params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        url = f"https://{host}.s3.amazonaws.com/?{urllib.parse.urlencode(params)}"
        text = _http().get(url, timeout=timeout).text
        keys.extend(re.findall(r"<Key>([^<]+)</Key>", text))
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", text)
        if not m:
            return keys
        token = m.group(1)


def _start_time(key: str) -> dt.datetime:
    """Parse the ``_sYYYYDDDHHMMSSm`` scan-start stamp out of an ABI filename."""
    m = re.search(r"_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})", key)
    if m is None:
        raise ValueError(f"no start time in {key}")
    year, doy, hh, mm, ss = (int(g) for g in m.groups())
    return dt.datetime(year, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(
        days=doy - 1, hours=hh, minutes=mm, seconds=ss
    )


def find_scans(
    product: str,
    start: dt.datetime,
    end: dt.datetime,
    channel: int | None = None,
    *,
    satellite: str | None = None,
    bucket: str | None = None,
) -> list[tuple[dt.datetime, str]]:
    """Keys for one product whose scan start falls in ``[start, end]``.

    Listings are per hour in the archive layout, so each covered hour is listed
    and the results filtered.
    """
    host = bucket or bucket_name(satellite, start)
    prod_dir = PRODUCTS[product][0]
    out: list[tuple[dt.datetime, str]] = []
    hour = start.replace(minute=0, second=0, microsecond=0)
    while hour <= end:
        prefix = f"{prod_dir}/{hour:%Y/%j/%H}/"
        for key in _list_keys(prefix, bucket=host):
            if channel is not None and f"C{channel:02d}_" not in key:
                continue
            t = _start_time(key)
            if start <= t <= end:
                out.append((t, key))
        hour += dt.timedelta(hours=1)
    return sorted(out)


def geodetic_to_scan(
    lat: np.ndarray, lon: np.ndarray, proj: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Geodetic lat/lon (deg) to ABI fixed-grid scan angles (rad).

    Implements the forward projection of the GOES-R Product User Guide vol. 3,
    section 5.1.2.8. Returns ``(x, y)`` matching the file's coordinate variables.
    """
    req = proj["semi_major_axis"]
    rpol = proj["semi_minor_axis"]
    h = proj["perspective_point_height"] + req
    lon0 = np.deg2rad(proj["longitude_of_projection_origin"])

    e2 = (req**2 - rpol**2) / req**2
    phi = np.deg2rad(np.asarray(lat, dtype=np.float64))
    lam = np.deg2rad(np.asarray(lon, dtype=np.float64))

    phi_c = np.arctan((rpol**2 / req**2) * np.tan(phi))
    rc = rpol / np.sqrt(1.0 - e2 * np.cos(phi_c) ** 2)

    sx = h - rc * np.cos(phi_c) * np.cos(lam - lon0)
    sy = -rc * np.cos(phi_c) * np.sin(lam - lon0)
    sz = rc * np.sin(phi_c)

    # Points on the far side of the globe have no line of sight to the satellite.
    visible = h * (h - sx) >= sy**2 + (req**2 / rpol**2) * sz**2
    y = np.arctan(sz / sx)
    x = np.arcsin(-sy / np.sqrt(sx**2 + sy**2 + sz**2))
    return np.where(visible, x, np.nan), np.where(visible, y, np.nan)


def scan_to_geodetic(
    x: np.ndarray, y: np.ndarray, proj: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`geodetic_to_scan`: scan angles (rad) to lat/lon (deg)."""
    req = proj["semi_major_axis"]
    rpol = proj["semi_minor_axis"]
    h = proj["perspective_point_height"] + req
    lon0 = np.deg2rad(proj["longitude_of_projection_origin"])

    sinx, cosx = np.sin(x), np.cos(x)
    siny, cosy = np.sin(y), np.cos(y)

    a = sinx**2 + cosx**2 * (cosy**2 + (req**2 / rpol**2) * siny**2)
    b = -2.0 * h * cosx * cosy
    c = h**2 - req**2

    disc = b**2 - 4.0 * a * c
    with np.errstate(invalid="ignore"):
        rs = (-b - np.sqrt(disc)) / (2.0 * a)

    sx = rs * cosx * cosy
    sy = -rs * sinx
    sz = rs * cosx * siny

    lat = np.rad2deg(
        np.arctan((req**2 / rpol**2) * sz / np.sqrt((h - sx) ** 2 + sy**2))
    )
    lon = np.rad2deg(lon0 - np.arctan(sy / (h - sx)))
    bad = disc < 0
    return np.where(bad, np.nan, lat), np.where(bad, np.nan, lon)


@dataclass
class Patch:
    """A georeferenced cutout of one ABI variable around a lat/lon site."""

    time: dt.datetime
    var: str
    values: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    units: str = ""
    long_name: str = ""
    site_lat: float = SITE_LAT
    site_lon: float = SITE_LON


def _box_slice(
    xs: np.ndarray,
    ys: np.ndarray,
    proj: dict,
    lat0: float,
    lon0: float,
    half_width_deg: float,
) -> tuple[int, int, int, int]:
    """Inclusive-exclusive index box for a lat/lon cutout on the ABI grid."""
    corner_lat = np.array([lat0 - half_width_deg, lat0 + half_width_deg] * 2)
    corner_lon = np.array([lon0 - half_width_deg] * 2 + [lon0 + half_width_deg] * 2)
    cx, cy = geodetic_to_scan(corner_lat, corner_lon, proj)
    ix = np.searchsorted(xs, [np.nanmin(cx), np.nanmax(cx)])
    iy = len(ys) - np.searchsorted(ys[::-1], [np.nanmax(cy), np.nanmin(cy)])
    i0, i1 = max(int(ix[0]) - 1, 0), min(int(ix[1]) + 1, len(xs))
    j0, j1 = max(int(iy[0]) - 1, 0), min(int(iy[1]) + 1, len(ys))
    return i0, i1, j0, j1


def _read_var_slice(ds, var: str, i0: int, i1: int, j0: int, j1: int):
    v = ds.variables[var]
    sl = v[j0:j1, i0:i1]
    if np.ma.isMaskedArray(sl):
        values = np.ma.filled(sl.astype(np.float64), np.nan)
    else:
        values = np.asarray(sl, dtype=np.float64)
    return values, getattr(v, "units", ""), getattr(v, "long_name", "")


def read_patch(
    key: str,
    var: str,
    *,
    half_width_deg: float = 1.5,
    timeout: float = 300.0,
    bucket: str | None = None,
    satellite: str | None = None,
    lat: float | None = None,
    lon: float | None = None,
) -> Patch:
    """Download one ABI file and cut out a box centred on ``lat``/``lon``.

    Defaults to Cerro Pachón. ``half_width_deg`` of 1.5 is about +/-165 km.
    """
    patches = read_patches(
        key,
        var,
        sites=[("site", float(SITE_LAT if lat is None else lat), float(SITE_LON if lon is None else lon))],
        half_width_deg=half_width_deg,
        timeout=timeout,
        bucket=bucket,
        satellite=satellite,
    )
    return patches["site"]


def read_patches(
    key: str,
    var: str,
    sites: list[tuple[str, float, float]],
    *,
    half_width_deg: float = 1.5,
    timeout: float = 300.0,
    bucket: str | None = None,
    satellite: str | None = None,
) -> dict[str, Patch]:
    """Download one ABI full-disk file once and cut a box for every site.

    ``sites`` is ``(name, lat_deg, lon_deg)``. Used by the multi-site
    pretrain fetch so nearby Andes/Pacific boxes share the same scan.
    """
    import netCDF4

    host = bucket or bucket_name(satellite, _start_time(key))
    url = f"https://{host}.s3.amazonaws.com/{key}"
    resp = _http().get(url, timeout=timeout)
    resp.raise_for_status()

    mem_name = f"inmem-{uuid.uuid4().hex}"
    out: dict[str, Patch] = {}
    when = _start_time(key)
    with _HDF5_LOCK:
        with netCDF4.Dataset(mem_name, mode="r", memory=resp.content) as ds:
            _fill_patches(ds, when, var, sites, half_width_deg, out)
    return out


def read_patches_vars(
    key: str,
    variables: list[str],
    sites: list[tuple[str, float, float]],
    *,
    half_width_deg: float = 1.5,
    timeout: float = 300.0,
    bucket: str | None = None,
    satellite: str | None = None,
) -> dict[str, dict[str, Patch]]:
    """One full-disk download, several variables, several site cutouts.

    Returns ``{var: {site: Patch}}``. Used by the rescue archive so
    multi-field products (DSIF) are not downloaded twice.
    """
    import netCDF4

    host = bucket or bucket_name(satellite, _start_time(key))
    url = f"https://{host}.s3.amazonaws.com/{key}"
    resp = _http().get(url, timeout=timeout)
    resp.raise_for_status()

    mem_name = f"inmem-{uuid.uuid4().hex}"
    when = _start_time(key)
    nested: dict[str, dict[str, Patch]] = {v: {} for v in variables}
    with _HDF5_LOCK:
        with netCDF4.Dataset(mem_name, mode="r", memory=resp.content) as ds:
            for var in variables:
                if var not in ds.variables:
                    continue
                _fill_patches(ds, when, var, sites, half_width_deg, nested[var])
    return nested


def _fill_patches(ds, when, var, sites, half_width_deg, out):
    gip = ds.variables["goes_imager_projection"]
    proj = {
        "semi_major_axis": float(gip.semi_major_axis),
        "semi_minor_axis": float(gip.semi_minor_axis),
        "perspective_point_height": float(gip.perspective_point_height),
        "longitude_of_projection_origin": float(
            gip.longitude_of_projection_origin
        ),
    }
    xs = ds.variables["x"][:].astype(np.float64)
    ys = ds.variables["y"][:].astype(np.float64)
    for name, lat0, lon0 in sites:
        i0, i1, j0, j1 = _box_slice(xs, ys, proj, lat0, lon0, half_width_deg)
        values, units, long_name = _read_var_slice(ds, var, i0, i1, j0, j1)
        gx, gy = np.meshgrid(xs[i0:i1], ys[j0:j1])
        plat, plon = scan_to_geodetic(gx, gy, proj)
        out[name] = Patch(
            time=when,
            var=var,
            values=values,
            lat=plat,
            lon=plon,
            units=units,
            long_name=long_name,
            site_lat=float(lat0),
            site_lon=float(lon0),
        )
