"""Dense C13 optical flow from adjacent ABI frames. No network.

Phase-correlation translation between consecutive native C13 cutouts,
converted to east/north m/s. Written as ``of.npz`` next to ``c13.npz``.
Used as a DMW wind fallback when the point cloud is empty.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

OF_NAME = "of.npz"
DEFAULT_HEIGHT_KM = 8.0


def fill_nan(frame: np.ndarray) -> np.ndarray:
    x = np.asarray(frame, dtype=np.float64)
    ok = np.isfinite(x)
    if not ok.any():
        return np.zeros_like(x)
    if ok.all():
        return x
    x = x.copy()
    x[~ok] = float(np.median(x[ok]))
    return x


def phase_shift_px(prev: np.ndarray, last: np.ndarray) -> tuple[float, float]:
    """Translation of ``last`` relative to ``prev`` (drow, dcol), sub-pixel."""
    a = fill_nan(prev)
    b = fill_nan(last)
    a = a - a.mean()
    b = b - b.mean()
    fa = np.fft.rfft2(a)
    fb = np.fft.rfft2(b)
    r = fa * np.conj(fb)
    r /= np.abs(r) + 1e-9
    c = np.fft.irfft2(r, s=a.shape)
    peak = np.unravel_index(int(np.argmax(c)), c.shape)
    pr, pc = int(peak[0]), int(peak[1])
    h, w = c.shape

    def quad(axis_val: int, neigh: tuple[float, float, float]) -> float:
        left, mid, right = neigh
        denom = left - 2.0 * mid + right
        if abs(denom) < 1e-9:
            return float(axis_val)
        return float(axis_val) + 0.5 * (left - right) / denom

    def wrap(i: int, n: int) -> int:
        return int(i) % n

    drow = quad(
        pr,
        (float(c[wrap(pr - 1, h), pc]), float(c[pr, pc]), float(c[wrap(pr + 1, h), pc])),
    )
    dcol = quad(
        pc,
        (float(c[pr, wrap(pc - 1, w)]), float(c[pr, pc]), float(c[pr, wrap(pc + 1, w)])),
    )
    if drow > h / 2:
        drow -= h
    if dcol > w / 2:
        dcol -= w
    return float(drow), float(dcol)


def px_to_ms(
    drow: float,
    dcol: float,
    lat: np.ndarray,
    lon: np.ndarray,
    dt_sec: float,
) -> tuple[float, float]:
    """Pixel shift → east/north m/s. Row 0 is north."""
    if dt_sec <= 1:
        return float("nan"), float("nan")
    cy, cx = lat.shape[0] // 2, lat.shape[1] // 2
    km_col = float((lon[cy, min(cx + 1, lon.shape[1] - 1)] - lon[cy, max(cx - 1, 0)]) / 2.0)
    km_row = float((lat[max(cy - 1, 0), cx] - lat[min(cy + 1, lat.shape[0] - 1), cx]) / 2.0)
    km_col *= 111.195 * np.cos(np.deg2rad(float(lat[cy, cx])))
    km_row *= 111.195
    # +dcol = east; +drow = south (row 0 is north).
    u = (dcol * km_col) * 1000.0 / dt_sec
    v = (-drow * km_row) * 1000.0 / dt_sec
    return float(u), float(v)


def _acha_height_km(acha: np.ndarray | None, t: int) -> float:
    if acha is None or t >= int(acha.shape[0]):
        return DEFAULT_HEIGHT_KM
    frame = np.asarray(acha[t], dtype=np.float64)
    ok = np.isfinite(frame) & (frame > 100.0)
    if not ok.any():
        return DEFAULT_HEIGHT_KM
    return float(np.median(frame[ok]) / 1000.0)


def flow_from_c13(
    values: np.ndarray,
    unix: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    *,
    acha: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Per-scan C13 flow. Frame 0 is NaN (no previous scan)."""
    n = int(values.shape[0])
    u = np.full(n, np.nan, dtype=np.float32)
    v = np.full(n, np.nan, dtype=np.float32)
    drow = np.full(n, np.nan, dtype=np.float32)
    dcol = np.full(n, np.nan, dtype=np.float32)
    height = np.full(n, DEFAULT_HEIGHT_KM, dtype=np.float32)
    for j in range(1, n):
        dt_sec = float(unix[j] - unix[j - 1])
        dr, dc = phase_shift_px(values[j - 1], values[j])
        uu, vv = px_to_ms(dr, dc, lat, lon, dt_sec)
        drow[j] = dr
        dcol[j] = dc
        u[j] = uu
        v[j] = vv
        height[j] = _acha_height_km(acha, j)
    height[0] = _acha_height_km(acha, 0)
    speed = np.hypot(u, v)
    return {
        "unix": np.asarray(unix, dtype=np.float64),
        "u": u,
        "v": v,
        "speed": speed,
        "drow": drow,
        "dcol": dcol,
        "height_km": height,
    }


def night_of_path(night_dir: Path) -> Path:
    return Path(night_dir) / OF_NAME


def of_is_current(night_dir: Path, unix: np.ndarray) -> bool:
    path = night_of_path(night_dir)
    if not path.is_file():
        return False
    try:
        z = np.load(path)
        got = np.asarray(z["unix"], dtype=np.float64)
    except (OSError, KeyError, ValueError):
        return False
    return got.shape == unix.shape and np.allclose(got, unix, atol=0.5, equal_nan=True)


def write_night_of(night_dir: Path, *, force: bool = False) -> dict:
    """Compute ``of.npz`` for one archive evening. Returns a small status dict."""
    night_dir = Path(night_dir)
    c13_path = night_dir / "c13.npz"
    if not c13_path.is_file():
        return {"night": night_dir.name, "skipped": "no_c13"}
    z = np.load(c13_path)
    unix = np.asarray(z["unix"], dtype=np.float64)
    if not force and of_is_current(night_dir, unix):
        return {"night": night_dir.name, "skipped": True, "n": int(unix.size)}
    acha = None
    acha_path = night_dir / "acha.npz"
    if acha_path.is_file():
        za = np.load(acha_path)
        au = np.asarray(za["unix"], dtype=np.float64)
        av = np.asarray(za["values"], dtype=np.float32)
        if au.shape == unix.shape and np.allclose(au, unix, atol=30.0):
            acha = av
    payload = flow_from_c13(
        np.asarray(z["values"], dtype=np.float32),
        unix,
        np.asarray(z["lat"], dtype=np.float32),
        np.asarray(z["lon"], dtype=np.float32),
        acha=acha,
    )
    out = night_of_path(night_dir)
    np.savez_compressed(out, **payload)
    finite = np.isfinite(payload["u"])
    return {
        "night": night_dir.name,
        "n": int(unix.size),
        "n_finite": int(finite.sum()),
        "path": str(out),
        "size_kb": round(out.stat().st_size / 1e3, 2),
    }
