"""GOES channel layout and cube encoder.

Channel order, fill values, and the native ABI cutout size used when the
forecaster reads a night-time GOES cube over the site.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from analysis.satellite.fetch_pretrain import DEFAULT_OUT, MODEL_PRODUCTS
from analysis.satellite.sites import SITE_BY_NAME

N_FRAMES = 8
LEAD_FRAMES = 2  # 20 min at ABI 10-min cadence (nearest to +15)
GRID = 48
# Native ABI full-disk cutout at Pachón (±1.5°). IR/ACM/phase/WV/ACHA are
# 135×149 @ ~2 km; LST/TPW are 28×32 @ 10 km and get lifted to this.
NATIVE_IR_HW = (135, 149)

# Align each product onto the C13 timeline. LST is hourly.
ALIGN_SLOP_SEC = {
    "lst": 40 * 60,
}

C13_FILL = 270.0
C07_FILL = 270.0
C15_FILL = 270.0
ACHA_FILL = 0.0
LST_FILL = 272.0
TPW_FILL = 2.0

# Channel order of ``x`` (T, C, H, W). Keep stable — training jobs key on this.
CHANNEL_NAMES = (
    "c13",
    "c07",
    "c15",
    "btd_fog",  # C13 − C07  (night water/stratus)
    "btd_split",  # C13 − C15  (thin-ice / IR OD proxy)
    "acm",
    "acha",
    "phase",
    "liquid",
    "ice",
    "lst",
    "tpw",
    "dc13",
    "dacm",
    "c08",
    "c09",
    "c10",
    "is_missing",
)


IR_K_LO, IR_K_HI = 180.0, 330.0
BTD_CLIP_K = 15.0
ACHA_HI = 20000.0
LST_LO, LST_HI = 200.0, 330.0
TPW_HI = 80.0


def _as_hw(size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(size, int):
        return (int(size), int(size))
    return (int(size[0]), int(size[1]))


def _resize(
    arr: np.ndarray,
    size: int | tuple[int, int] = GRID,
    *,
    nearest: bool = False,
) -> np.ndarray:
    import torch.nn.functional as F

    hw = _as_hw(size)
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 2 and a.shape == hw:
        return a
    if a.ndim == 3 and a.shape[-2:] == hw:
        return a
    t = torch.from_numpy(a)
    mode = "nearest" if nearest else "bilinear"
    kw = {} if nearest else {"align_corners": False}
    if t.ndim == 2:
        t = t[None, None]
        out = F.interpolate(t, size=hw, mode=mode, **kw)
        return out[0, 0].numpy()
    if t.ndim == 3:
        t = t[None]
        out = F.interpolate(t, size=hw, mode=mode, **kw)
        return out[0].numpy()
    raise ValueError(arr.shape)


def _fill_nan(arr: np.ndarray, fill: float, *, mode: str = "constant") -> np.ndarray:
    """Replace NaNs. ``median`` uses that frame's finite pixels (LST/TPW)."""
    out = np.array(arr, dtype=np.float32, copy=True)
    for t in range(out.shape[0]):
        frame = out[t]
        miss = ~np.isfinite(frame)
        if not bool(miss.any()):
            continue
        if mode == "median":
            fin = frame[~miss]
            f = float(np.median(fin)) if fin.size else float(fill)
        else:
            f = float(fill)
        frame[miss] = f
    return out


def ir_is_missing(
    c13: np.ndarray,
    c07: np.ndarray,
    c15: np.ndarray,
    *,
    size: int | tuple[int, int] = GRID,
    c08: np.ndarray | None = None,
    c09: np.ndarray | None = None,
    c10: np.ndarray | None = None,
) -> np.ndarray:
    """Binary ``(T, H, W)``: 1 where IR was NaN before the 270 K fill.

    C13/C07/C15 (and C08–C10 when present) share that fill. After fill the
    cube cannot tell a missing pixel from a real ~270 K cloud or ground.
    Nearest resize keeps the mask binary on the 48×48 grid.
    """
    miss = ~np.isfinite(c13) | ~np.isfinite(c07) | ~np.isfinite(c15)
    for wv in (c08, c09, c10):
        if wv is not None:
            miss = miss | ~np.isfinite(wv)
    miss = _resize(miss.astype(np.float32), size, nearest=True)
    return (miss >= 0.5).astype(np.float32)


def encode_cube(
    c13: np.ndarray,
    c07: np.ndarray,
    c15: np.ndarray,
    acm: np.ndarray,
    acha: np.ndarray,
    phase: np.ndarray,
    lst: np.ndarray,
    tpw: np.ndarray,
    *,
    size: int | tuple[int, int] = GRID,
    c08: np.ndarray | None = None,
    c09: np.ndarray | None = None,
    c10: np.ndarray | None = None,
) -> np.ndarray:
    """``(T, 15|18, H, W)`` night cube. BTDs use pre-fill Kelvin.

    C08/C09/C10 append after the original 14 slots when supplied (same
    Kelvin z-score as C13). Missing WV is filled at 270 K. Last channel
    is ``is_missing``: 1 where any of those IR planes was NaN before fill.

    LST/TPW NaNs get that frame's spatial median so the constant fills
    (272 K / 2 mm) do not collide with the real night distribution.
    ACHA NaNs stay 0 (no cloud-top; that 0 is the no-retrieval convention,
    not a 270 K-style collision with a real height). IR/BTD/height/TPW
    are clipped to physical ranges after fill.
    """
    is_missing = ir_is_missing(
        c13, c07, c15, size=size, c08=c08, c09=c09, c10=c10
    )
    both_fog = np.isfinite(c13) & np.isfinite(c07)
    both_split = np.isfinite(c13) & np.isfinite(c15)
    fog = np.where(both_fog, c13 - c07, np.nan)
    split = np.where(both_split, c13 - c15, np.nan)

    c13 = _resize(_fill_nan(c13, C13_FILL), size)
    c07 = _resize(_fill_nan(c07, C07_FILL), size)
    c15 = _resize(_fill_nan(c15, C15_FILL), size)
    fog = _resize(_fill_nan(fog, 0.0), size)
    split = _resize(_fill_nan(split, 0.0), size)
    acm = _resize(_fill_nan(acm, 0.0), size, nearest=True)
    acha = _resize(_fill_nan(acha, ACHA_FILL), size)
    phase = _resize(_fill_nan(phase, 0.0), size, nearest=True)
    lst = _resize(_fill_nan(lst, LST_FILL, mode="median"), size)
    tpw = _resize(_fill_nan(tpw, TPW_FILL, mode="median"), size)

    c13 = np.clip(c13, IR_K_LO, IR_K_HI)
    c07 = np.clip(c07, IR_K_LO, IR_K_HI)
    c15 = np.clip(c15, IR_K_LO, IR_K_HI)
    fog = np.clip(fog, -BTD_CLIP_K, BTD_CLIP_K)
    split = np.clip(split, -BTD_CLIP_K, BTD_CLIP_K)
    acm = np.clip(acm, 0.0, 1.0)
    acha = np.clip(acha, 0.0, ACHA_HI)
    phase = np.clip(phase, 0.0, 4.0)
    lst = np.clip(lst, LST_LO, LST_HI)
    tpw = np.clip(tpw, 0.0, TPW_HI)

    liquid = ((phase >= 0.5) & (phase < 2.5)).astype(np.float32)
    ice = (phase >= 3.5).astype(np.float32)
    dc13 = np.zeros_like(c13)
    dacm = np.zeros_like(acm)
    if c13.shape[0] > 1:
        dc13[1:] = c13[1:] - c13[:-1]
        dacm[1:] = acm[1:] - acm[:-1]
    layers = [
        (c13 - 278.0) / 15.0,
        (c07 - 278.0) / 15.0,
        (c15 - 278.0) / 15.0,
        fog / 5.0,
        split / 5.0,
        acm,
        (acha - 62.0) / 1639.0,
        phase / 4.0,
        liquid,
        ice,
        (lst - 272.0) / 15.0,
        (tpw - 2.0) / 4.0,
        np.clip(dc13 / 15.0, -2.0, 2.0),
        np.clip(dacm, -1.0, 1.0),
    ]
    for wv in (c08, c09, c10):
        if wv is None:
            filled = np.full_like(c13, (C13_FILL - 278.0) / 15.0)
        else:
            filled = np.clip(_resize(_fill_nan(wv, C13_FILL), size), IR_K_LO, IR_K_HI)
            filled = (filled - 278.0) / 15.0
        layers.append(filled)
    layers.append(is_missing)
    x = np.stack(layers, axis=1)
    return np.ascontiguousarray(x, dtype=np.float32)


def _align(unix: np.ndarray, unix0: np.ndarray, val: np.ndarray, slop: float) -> np.ndarray:
    idx = np.abs(unix[None, :] - unix0[:, None]).argmin(axis=1)
    out = val[idx]
    ok = np.abs(unix[idx] - unix0) <= slop
    out = out.copy()
    out[~ok] = np.nan
    return out


def _load_night(root: Path, site: str, date: str) -> dict[str, np.ndarray] | None:
    d = root / site / date
    paths = {p: d / f"{p}.npz" for p in MODEL_PRODUCTS}
    if not all(p.is_file() for p in paths.values()):
        return None
    # C13 is the time spine (10 min ABI).
    z0 = np.load(paths["c13"])
    unix0 = np.asarray(z0["unix"], dtype=np.float64)
    out: dict[str, np.ndarray] = {
        "unix": unix0,
        "c13": np.asarray(z0["values"], dtype=np.float32),
    }
    for name, path in paths.items():
        if name == "c13":
            continue
        z = np.load(path)
        unix = np.asarray(z["unix"], dtype=np.float64)
        val = np.asarray(z["values"], dtype=np.float32)
        slop = float(ALIGN_SLOP_SEC.get(name, 8 * 60))
        out[name] = _align(unix, unix0, val, slop)
    for name in ("c08", "c09", "c10"):
        path = d / f"{name}.npz"
        if not path.is_file():
            continue
        z = np.load(path)
        unix = np.asarray(z["unix"], dtype=np.float64)
        val = np.asarray(z["values"], dtype=np.float32)
        out[name] = _align(unix, unix0, val, 8 * 60)
    return out


def build_index(
    root: Path | None = None,
    *,
    n_frames: int = N_FRAMES,
    lead: int = LEAD_FRAMES,
    split: str | None = None,
) -> list[dict]:
    dest = Path(root) if root is not None else DEFAULT_OUT
    rows: list[dict] = []
    if not dest.is_dir():
        return rows
    for site_dir in sorted(dest.iterdir()):
        if not site_dir.is_dir() or site_dir.name.startswith("."):
            continue
        site = SITE_BY_NAME.get(site_dir.name)
        if site is None:
            continue
        if split is not None and site.pretrain_split != split:
            continue
        for night in sorted(p for p in site_dir.iterdir() if p.is_dir()):
            packed = _load_night(dest, site.name, night.name)
            if packed is None:
                continue
            unix = packed["unix"]
            acm = packed["acm"]
            n = len(unix)
            need = n_frames + lead
            for i0 in range(0, n - need + 1):
                sl = slice(i0, i0 + n_frames)
                it = i0 + n_frames - 1 + lead
                dt = np.diff(unix[sl])
                if np.any(dt > 15 * 60) or (unix[it] - unix[i0 + n_frames - 1]) > 25 * 60:
                    continue
                hist = acm[sl]
                tgt = acm[it]
                last = acm[i0 + n_frames - 1]
                if not (np.isfinite(hist).mean() > 0.8 and np.isfinite(tgt).mean() > 0.8):
                    continue
                gold = tgt >= 0.5
                persist = last >= 0.5
                inter = np.logical_and(gold, persist).sum()
                union = np.logical_or(gold, persist).sum()
                rows.append(
                    {
                        "site": site.name,
                        "date": night.name,
                        "i0": i0,
                        "it": int(it),
                        "split": site.pretrain_split,
                        "persist_iou": float(inter / union) if union else 1.0,
                    }
                )
    return rows


def save_index(rows: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + ("\n" if rows else ""))
    return path


def split_rows(rows: list[dict]) -> dict[str, list[dict]]:
    out = {"train": [], "val": [], "test": []}
    for r in rows:
        out.setdefault(r.get("split", "train"), []).append(r)
    return out


class SatNightDataset(Dataset):
    def __init__(self, rows: list[dict], root: Path | None = None, cache: bool = True):
        self.rows = list(rows)
        self.root = Path(root) if root is not None else DEFAULT_OUT
        self._nights: dict[tuple[str, str], dict] = {}
        self.cache = cache

    def _night(self, site: str, date: str) -> dict:
        key = (site, date)
        if key not in self._nights:
            packed = _load_night(self.root, site, date)
            if packed is None:
                raise FileNotFoundError(f"{site}/{date}")
            if self.cache:
                self._nights[key] = packed
            else:
                return packed
        return self._nights[key]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        row = self.rows[idx]
        night = self._night(row["site"], row["date"])
        i0, it = int(row["i0"]), int(row["it"])
        sl = slice(i0, i0 + N_FRAMES)

        def raw(name: str) -> np.ndarray:
            return np.asarray(night[name][sl], dtype=np.float32)

        x = encode_cube(
            raw("c13"),
            raw("c07"),
            raw("c15"),
            raw("acm"),
            raw("acha"),
            raw("actp"),
            raw("lst"),
            raw("tpw"),
            c08=night["c08"][sl] if "c08" in night else None,
            c09=night["c09"][sl] if "c09" in night else None,
            c10=night["c10"][sl] if "c10" in night else None,
        )
        tgt = _resize(np.nan_to_num(night["acm"][it], nan=0.0), nearest=True)
        last = x[-1, CHANNEL_NAMES.index("acm")]
        return {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy((tgt >= 0.5).astype(np.float32)),
            "last": torch.from_numpy((last >= 0.5).astype(np.float32)),
            "site": row["site"],
            "date": row["date"],
        }
