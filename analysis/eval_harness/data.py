"""Dataset loading for the all-sky evaluation protocol.

Primary quantity is **transmission** over the full sky:
    T = 10 ** (-0.4 * A)
where A is extinction [mag] from the FITS ``clouds`` column.

Maps are HEALPix on the sphere (nested, equal-area). Loaders validate
``NSIDE``, ``SCHEME``, and ``NPIX`` against the FITS headers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[2]
NIGHT = ROOT / "data/fits/20240219_hp4"

# all-sky defaults: 1 h history → predict +20 min
HISTORY_SEC = 3600
HORIZON_SEC = 1200

# Expected spherical discretization for this dataset
EXPECTED_NSIDE = 16
EXPECTED_SCHEME = "NEST"
EXPECTED_NPIX = 12 * EXPECTED_NSIDE * EXPECTED_NSIDE  # 3072

# Legacy multi-horizon frame eval (optional / secondary)
HORIZONS_SEC: tuple[int, ...] = (60, 300, 600, 900, 1200, 1800, 3600)
HORIZON_LABELS: dict[int, str] = {
    60: "1 min",
    300: "5 min",
    600: "10 min",
    900: "15 min",
    1200: "20 min",
    1800: "30 min",
    3600: "1 hour",
}
MIN_NOBS = 20

# Fractional-error metrics divide by the measured transmission, which tends to
# zero under thick cloud. Tiles below this floor are excluded from those
# metrics only (absolute metrics such as RMSE still use every tile).
TRUTH_FLOOR = 0.05


def extinction_to_transmission(A: np.ndarray) -> np.ndarray:
    """Convert extinction [mag] to flux transmission fraction."""
    T = np.power(10.0, -0.4 * A)
    T = T.astype(np.float64, copy=False)
    T[~np.isfinite(A)] = np.nan
    return T


def _validate_healpix_header(header, path: Path) -> tuple[int, str, int]:
    """Return (nside, scheme, npix) after checking all-sky geometry."""
    if "NSIDE" not in header or "NPIX" not in header or "SCHEME" not in header:
        raise ValueError(
            f"{path.name}: missing HEALPix geometry keywords "
            f"(need NSIDE, NPIX, SCHEME); got {list(header.keys())}"
        )
    nside = int(header["NSIDE"])
    npix = int(header["NPIX"])
    scheme = str(header["SCHEME"]).strip().upper()
    if scheme.startswith("NEST"):
        scheme = "NEST"
    elif scheme.startswith("RING"):
        scheme = "RING"
    else:
        raise ValueError(f"{path.name}: unrecognized SCHEME={header['SCHEME']!r}")

    if nside != EXPECTED_NSIDE:
        raise ValueError(
            f"{path.name}: NSIDE={nside} != expected {EXPECTED_NSIDE}"
        )
    if npix != EXPECTED_NPIX:
        raise ValueError(
            f"{path.name}: NPIX={npix} != expected {EXPECTED_NPIX}"
        )
    if scheme != EXPECTED_SCHEME:
        raise ValueError(
            f"{path.name}: SCHEME={scheme} != expected {EXPECTED_SCHEME}"
        )
    if npix != 12 * nside * nside:
        raise ValueError(
            f"{path.name}: NPIX={npix} inconsistent with NSIDE={nside}"
        )
    return nside, scheme, npix


def load_split(split: str, night_dir: Path | None = None) -> dict:
    """Load a chronological all-sky map split (``train`` or ``validation``).

    Returns
    -------
    dict with keys:
        T : (n_times, npix) transmission field (all-sky quantity)
        A : (n_times, npix) extinction [mag]
        nobs : (n_times, npix) star counts
        jd : (n_times,) Julian Date
        names : list[str]
        split : str
        nside, scheme, npix : HEALPix geometry (validated)
    """
    folder = (night_dir or NIGHT) / split
    files = sorted(folder.glob("clouds*_weighted.fits.gz"))
    if not files:
        raise FileNotFoundError(f"No maps found in {folder}")

    As, Ns, jds, names = [], [], [], []
    nside = scheme = npix = None
    for path in files:
        with fits.open(path) as hdul:
            hdr = hdul[1].header
            nside_i, scheme_i, npix_i = _validate_healpix_header(hdr, path)
            if nside is None:
                nside, scheme, npix = nside_i, scheme_i, npix_i
            elif (nside_i, scheme_i, npix_i) != (nside, scheme, npix):
                raise ValueError(
                    f"{path.name}: geometry {(nside_i, scheme_i, npix_i)} "
                    f"differs from {(nside, scheme, npix)}"
                )

            tab = hdul[1].data
            A = np.asarray(tab["clouds"], dtype=np.float64).copy()
            nobs = np.asarray(tab["nobs"], dtype=np.int32)
            if A.shape[0] != npix:
                raise ValueError(
                    f"{path.name}: table length {A.shape[0]} != NPIX={npix}"
                )
            A[(nobs <= 0) | ~np.isfinite(A)] = np.nan
            As.append(A)
            Ns.append(nobs)
            jds.append(float(hdr["JD"]))
            names.append(path.name)

    jd = np.asarray(jds, dtype=np.float64)
    if not np.all(np.diff(jd) > 0):
        raise ValueError(f"{split} Julian Dates are not strictly increasing")

    A = np.stack(As)
    return {
        "T": extinction_to_transmission(A),
        "A": A,
        "nobs": np.stack(Ns),
        "jd": jd,
        "names": names,
        "split": split,
        "nside": int(nside),
        "scheme": str(scheme),
        "npix": int(npix),
    }
