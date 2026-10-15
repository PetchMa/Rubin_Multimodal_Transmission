"""Encoded training pairs for the transmission forecaster.

Reads the parquet bundles from ``analysis.dataset_pairs`` and the memmap
cache used by training. The network itself lives in ``analysis.forecaster``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import multiprocessing as mp
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler, Subset

from analysis.torch_util import resolve_device, seed_everything
from analysis.dataset_pairs import DEFAULT_OUT, iter_samples
from analysis.eval_harness.data import HORIZON_SEC, ROOT, TRUTH_FLOOR
from analysis.eval_harness.dream_data import DREAM_MIN_NOBS, DREAM_NSIDE
from analysis.eval_harness.horizons import debias, mae, rmse
from analysis.satellite.goes import SITE_LAT, SITE_LON

JD_UNIX_EPOCH = 2440587.5
PAIRS_LINK = ROOT / "data/pairs/dream_20min"
CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"
LOG_DIR = Path(__file__).resolve().parent / "logs"

# C13 brightness temperature (K) and ACHA cloud-top height (m). Used only as
# fill values before train statistics exist (e.g. --dry-run).
C13_FILL = 270.0
ACHA_FILL = 8000.0
META_DIM = 9  # hour sin/cos, doy sin/cos (held at 0), winds, scan age, lead
# Day-of-year occupies these slots but is zeroed: the even/odd split is by
# night, so doy is a near-unique barcode and the net memorizes weather.
DOY_SLICE = slice(2, 4)
#: Padded DMW point-cloud size. Features: u, v, height_km, east_km, north_km, age_hr.
MAX_DMW = 400
WIND_FEAT = 6
KM_PER_DEG_LAT = 111.195
# Residual quantile head. Point forecast / gallery / skip story use the median.
QUANTILE_TAUS = (0.1, 0.25, 0.5, 0.75, 0.9)
MEDIAN_Q = QUANTILE_TAUS.index(0.5)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def resolve_pairs(path: str | Path | None = None) -> Path:
    """Prefer the repo symlink, then the large-disk default from packing."""
    if path is not None:
        return Path(path)
    if PAIRS_LINK.exists():
        return PAIRS_LINK
    return DEFAULT_OUT


def list_bundles(pairs_dir: Path, split: str | None = None) -> list[Path]:
    root = Path(pairs_dir)
    if not root.exists():
        return []
    if split is not None:
        # Empty split folder (or a missing one) is empty, not "all bundles".
        return sorted((root / split).rglob("*.parquet"))
    return sorted(p for p in root.rglob("*.parquet") if p.name != "manifest.json")


# ---------------------------------------------------------------------------
# Metadata: time of night, day of year
# ---------------------------------------------------------------------------


def _jd_to_utc(jd: float) -> dt.datetime:
    return dt.datetime.fromtimestamp(
        (float(jd) - JD_UNIX_EPOCH) * 86400.0, tz=dt.timezone.utc
    )


def _cyclic(value: float, period: float) -> tuple[float, float]:
    ang = 2.0 * math.pi * (value / period)
    return math.sin(ang), math.cos(ang)


def time_features(jd: float) -> np.ndarray:
    """Local solar hour as (sin, cos). Day-of-year slots stay zero.

    Hour-of-night is a physical prior (seeing, airmass, typical cloud
    evolution). Day-of-year is not: held-out nights have unseen doy
    values, so a nonzero encoding lets the model key on calendar date.
    """
    utc = _jd_to_utc(jd)
    utc_hour = utc.hour + utc.minute / 60.0 + utc.second / 3600.0
    local_hour = (utc_hour + SITE_LON / 15.0) % 24.0
    hs, hc = _cyclic(local_hour, 24.0)
    return np.array([hs, hc, 0.0, 0.0], dtype=np.float32)


def frame_metadata(
    jd: float,
    *,
    u: float,
    v: float,
    height_km: float,
    scan_age_hour: float,
    lead_hours: float,
) -> np.ndarray:
    """Per-token context vector: cyclic calendar + winds + timing."""
    return np.concatenate(
        [
            time_features(jd),
            np.array(
                [
                    np.tanh(u / 30.0),
                    np.tanh(v / 30.0),
                    np.tanh((height_km - 8.0) / 8.0),
                    np.tanh(scan_age_hour / 2.0),
                    np.tanh(lead_hours / 6.0),
                ],
                dtype=np.float32,
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Pair → tensors
# ---------------------------------------------------------------------------


def _subsample_idx(n: int, k: int) -> np.ndarray:
    if n <= 0:
        return np.zeros(k, dtype=int)
    return np.round(np.linspace(0, n - 1, k)).astype(int)


def _causal_index(times: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Latest scan at or before each query time, clamped to the first scan."""
    times = np.asarray(times, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64)
    if times.size == 0:
        return np.zeros(query.shape, dtype=int)
    idx = np.searchsorted(times, query, side="right") - 1
    return np.clip(idx, 0, times.size - 1)


def _resize_maps(maps: np.ndarray, size: int, fill_value: float) -> np.ndarray:
    """``(T, H, W)`` → ``(T, size, size)``. NaNs filled per-frame before resize."""
    maps = np.asarray(maps, dtype=np.float32)
    t = int(maps.shape[0])
    out = np.zeros((t, size, size), dtype=np.float32)
    for i in range(t):
        frame = maps[i]
        finite = np.isfinite(frame)
        filled = np.where(finite, frame, fill_value)[None, None]
        out[i] = (
            F.interpolate(
                torch.from_numpy(filled),
                size=(size, size),
                mode="bilinear",
                align_corners=False,
            )
            .numpy()[0, 0]
        )
    return out


def _maps_at(arr: np.ndarray, idx: np.ndarray, size: int, fill_value: float) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.size == 0 or arr.shape[0] == 0:
        return np.full((len(idx), size, size), fill_value, dtype=np.float32)
    return _resize_maps(arr[idx], size, fill_value)


def _xy_km(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    east = (lon - SITE_LON) * KM_PER_DEG_LAT * np.cos(np.deg2rad(SITE_LAT))
    north = (lat - SITE_LAT) * KM_PER_DEG_LAT
    return east, north


def _pad_points(points: np.ndarray, max_n: int = MAX_DMW) -> tuple[np.ndarray, np.ndarray]:
    """Pad / subsample ``(n, WIND_FEAT)`` to ``(max_n, WIND_FEAT)`` with a 0/1 mask."""
    out = np.zeros((max_n, WIND_FEAT), dtype=np.float32)
    mask = np.zeros((max_n,), dtype=np.float32)
    n = int(points.shape[0]) if points.ndim == 2 else 0
    if n <= 0:
        return out, mask
    if n > max_n:
        idx = np.round(np.linspace(0, n - 1, max_n)).astype(int)
        points = points[idx]
        n = max_n
    out[:n] = points[:n]
    mask[:n] = 1.0
    return out, mask


def _wind_points_at(dmw: dict | None, query_unix: float) -> np.ndarray:
    """Causal DMW points at ``query_unix``: u, v, height, east, north, age_hr."""
    empty = np.zeros((0, WIND_FEAT), dtype=np.float32)
    if not dmw:
        return empty
    t = np.asarray(dmw.get("time", []), dtype=np.float64)
    if t.size == 0:
        return empty
    keep = np.isfinite(t) & (t <= float(query_unix))
    if not np.any(keep):
        return empty
    lat = np.asarray(dmw["lat"], dtype=np.float64)[keep]
    lon = np.asarray(dmw["lon"], dtype=np.float64)[keep]
    east, north = _xy_km(lat, lon)
    age_hr = (float(query_unix) - t[keep]) / 3600.0
    pts = np.stack(
        [
            np.asarray(dmw["u"], dtype=np.float64)[keep],
            np.asarray(dmw["v"], dtype=np.float64)[keep],
            np.asarray(dmw["height_km"], dtype=np.float64)[keep],
            east,
            north,
            age_hr,
        ],
        axis=1,
    )
    finite = np.isfinite(pts).all(axis=1)
    return pts[finite].astype(np.float32)


def encode_bundle(
    sample: dict,
    *,
    n_frames: int,
    sat_size: int,
    min_nobs: int,
) -> dict[str, np.ndarray]:
    """Turn one decoded parquet bundle into fixed-shape model tensors."""
    sky = np.asarray(sample["field_window"], dtype=np.float32)
    nobs = np.asarray(sample["nobs_window"], dtype=np.float32)
    jd = np.asarray(sample["jd_window"], dtype=np.float64)
    pick = _subsample_idx(sky.shape[0], n_frames)
    sky = sky[pick]
    nobs = nobs[pick]
    jd = jd[pick]
    sky_mask = (np.isfinite(sky) & (nobs >= min_nobs)).astype(np.float32)
    sky = np.where(sky_mask > 0, sky, 0.0).astype(np.float32)

    frame_unix = (jd - JD_UNIX_EPOCH) * 86400.0
    c13_idx = _causal_index(sample["sat_unix"], frame_unix)
    acha_idx = _causal_index(sample["sat_acha_unix"], frame_unix)
    c13_maps = _maps_at(sample["sat_c13"], c13_idx, sat_size, fill_value=C13_FILL)
    acha_maps = _maps_at(sample["sat_acha"], acha_idx, sat_size, fill_value=0.0)
    sat = np.stack([c13_maps, acha_maps], axis=1)  # (T, 2, H, W)

    sat_unix = np.asarray(sample["sat_unix"], dtype=np.float64)
    ages = np.zeros(n_frames, dtype=np.float32)
    if sat_unix.size:
        ages = ((frame_unix - sat_unix[c13_idx]) / 3600.0).astype(np.float32)

    u = float(sample["sat_u"])
    v = float(sample["sat_v"])
    hkm = float(sample["sat_height_km"])
    lead_h = float(sample["dt_ahead_days"]) * 24.0
    meta = np.stack(
        [
            frame_metadata(
                float(jd[t]),
                u=u,
                v=v,
                height_km=hkm,
                scan_age_hour=float(ages[t]),
                lead_hours=lead_h,
            )
            for t in range(n_frames)
        ]
    )
    meta_q = frame_metadata(
        float(sample["t_target_jd"]),
        u=u,
        v=v,
        height_km=hkm,
        scan_age_hour=float(ages[-1]) + lead_h,
        lead_hours=lead_h,
    )

    dmw = sample.get("dmw")
    wind_set = np.zeros((n_frames, MAX_DMW, WIND_FEAT), dtype=np.float32)
    wind_mask = np.zeros((n_frames, MAX_DMW), dtype=np.float32)
    for t in range(n_frames):
        pts = _wind_points_at(dmw, float(frame_unix[t]))
        wind_set[t], wind_mask[t] = _pad_points(pts)
    # Persist last-frame winds. Querying DMW at t_target would include
    # ABI vectors issued after T=0 (a leak). Age is not rewritten; the
    # lead slot in meta_q already says +horizon.
    wind_set_q, wind_mask_q = wind_set[-1].copy(), wind_mask[-1].copy()

    target = np.asarray(sample["target"], dtype=np.float32)
    t_nobs = np.asarray(sample["target_nobs"], dtype=np.float32)
    y_mask = (np.isfinite(target) & (t_nobs >= min_nobs)).astype(np.float32)
    target = np.where(y_mask > 0, target, 0.0).astype(np.float32)

    return {
        "sky": sky,
        "sky_mask": sky_mask,
        "sat": sat,
        "meta": meta,
        "meta_q": meta_q,
        "wind_set": wind_set,
        "wind_mask": wind_mask,
        "wind_set_q": wind_set_q,
        "wind_mask_q": wind_mask_q,
        "y": target,
        "y_mask": y_mask,
        "last": sky[-1].copy(),
        "last_mask": sky_mask[-1].copy(),
    }


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


@dataclass
class ChannelNorm:
    """Robust (median / IQR) scale, fitted on train only."""

    mean: float
    std: float
    clip_min: float | None = None
    clip_max: float | None = None

    @classmethod
    def fit(cls, values: np.ndarray, clip_min: float | None = None, clip_max: float | None = None) -> ChannelNorm:
        v = np.asarray(values, dtype=np.float64)
        v = v[np.isfinite(v)]
        if clip_min is not None or clip_max is not None:
            v = np.clip(v, clip_min, clip_max)
        if v.size == 0:
            return cls(0.0, 1.0, clip_min, clip_max)
        med = float(np.median(v))
        iqr = float(np.percentile(v, 75) - np.percentile(v, 25))
        return cls(med, max(iqr, 1e-3), clip_min, clip_max)

    def encode(self, x: np.ndarray) -> np.ndarray:
        if self.clip_min is not None or self.clip_max is not None:
            x = np.clip(x, self.clip_min, self.clip_max)
        return ((x - self.mean) / self.std).astype(np.float32)

    def decode(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean

    def encode_t(self, x: torch.Tensor) -> torch.Tensor:
        if self.clip_min is not None or self.clip_max is not None:
            x = torch.clamp(x, min=self.clip_min, max=self.clip_max)
        return (x - self.mean) / self.std

    def decode_t(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean


@dataclass
class BundleNorm:
    sky: ChannelNorm
    c13: ChannelNorm
    acha: ChannelNorm
    wind: tuple[ChannelNorm, ChannelNorm, ChannelNorm, ChannelNorm, ChannelNorm, ChannelNorm]

    @classmethod
    def fit(cls, rows: list[dict]) -> BundleNorm:
        sky_vals, c13_vals, acha_vals = [], [], []
        wind_vals: list[list[np.ndarray]] = [[] for _ in range(WIND_FEAT)]
        for r in rows:
            m = r["sky_mask"] > 0
            sky_vals.append(r["sky"][m])
            sat = r["sat"]
            c13_vals.append(sat[:, 0].ravel())
            acha_vals.append(sat[:, 1].ravel())
            wm = r["wind_mask"] > 0
            if np.any(wm):
                pts = r["wind_set"][wm]
                for i in range(WIND_FEAT):
                    wind_vals[i].append(pts[:, i])
            wmq = r["wind_mask_q"] > 0
            if np.any(wmq):
                pts_q = r["wind_set_q"][wmq]
                for i in range(WIND_FEAT):
                    wind_vals[i].append(pts_q[:, i])
        wind = tuple(
            ChannelNorm.fit(np.concatenate(v) if v else [0.0]) for v in wind_vals
        )
        return cls(
            sky=ChannelNorm.fit(np.concatenate(sky_vals) if sky_vals else [0.0], clip_min=0.0, clip_max=3.0),
            c13=ChannelNorm.fit(np.concatenate(c13_vals) if c13_vals else [C13_FILL]),
            acha=ChannelNorm.fit(np.concatenate(acha_vals) if acha_vals else [0.0]),
            wind=wind,  # type: ignore[arg-type]
        )

    @classmethod
    def fit_files(
        cls,
        files: list[Path],
        *,
        n_frames: int,
        sat_size: int,
        min_nobs: int,
        max_files: int = 4096,
        verbose: bool = True,
    ) -> BundleNorm:
        if not files:
            return cls.fit([])
        n_pick = min(max_files, len(files))
        pick = [files[i] for i in np.round(np.linspace(0, len(files) - 1, n_pick)).astype(int)]
        rows = []
        for path in _progress(pick, desc="fit norm", total=n_pick, disable=not verbose):
            rows.append(
                encode_file(
                    path,
                    "train",
                    n_frames=n_frames,
                    sat_size=sat_size,
                    min_nobs=min_nobs,
                )
            )
        return cls.fit(rows)

    def apply_row(self, row: dict) -> dict:
        out = dict(row)
        out["sky"] = self.sky.encode(row["sky"]) * row["sky_mask"]
        out["last"] = self.sky.encode(row["last"]) * row["last_mask"]
        out["y"] = self.sky.encode(row["y"]) * row["y_mask"]
        sat = row["sat"].copy()
        sat[:, 0] = self.c13.encode(sat[:, 0])
        sat[:, 1] = self.acha.encode(sat[:, 1])
        out["sat"] = np.nan_to_num(sat, nan=0.0).astype(np.float32)
        wind = row["wind_set"].copy()
        wind_q = row["wind_set_q"].copy()
        for i, nrm in enumerate(self.wind):
            wind[..., i] = nrm.encode(wind[..., i])
            wind_q[..., i] = nrm.encode(wind_q[..., i])
        out["wind_set"] = (wind * row["wind_mask"][..., None]).astype(np.float32)
        out["wind_set_q"] = (wind_q * row["wind_mask_q"][..., None]).astype(np.float32)
        out["wind_mask"] = row["wind_mask"].astype(np.float32)
        out["wind_mask_q"] = row["wind_mask_q"].astype(np.float32)
        return out


def _stack(rows: list[dict], key: str) -> np.ndarray:
    return np.stack([r[key] for r in rows])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


ARRAY_KEYS = (
    "sky",
    "sky_mask",
    "sat",
    "meta",
    "meta_q",
    "wind_set",
    "wind_mask",
    "wind_set_q",
    "wind_mask_q",
    "y",
    "y_mask",
    "last",
    "last_mask",
)


def default_cache_dir(pairs_dir: Path) -> Path:
    root = Path(pairs_dir).resolve()
    return root.parent / f"{root.name}_encoded"


def encode_file(
    path: Path,
    split: str,
    *,
    n_frames: int,
    sat_size: int,
    min_nobs: int,
) -> dict[str, np.ndarray]:
    sample = next(iter_samples(path))
    sample["split"] = split
    return encode_bundle(
        sample, n_frames=n_frames, sat_size=sat_size, min_nobs=min_nobs
    )


def _array_shapes(
    n: int, n_frames: int, npix: int, sat_size: int
) -> dict[str, tuple[int, ...]]:
    return {
        "sky": (n, n_frames, npix),
        "sky_mask": (n, n_frames, npix),
        "sat": (n, n_frames, 2, sat_size, sat_size),
        "meta": (n, n_frames, META_DIM),
        "meta_q": (n, META_DIM),
        "wind_set": (n, n_frames, MAX_DMW, WIND_FEAT),
        "wind_mask": (n, n_frames, MAX_DMW),
        "wind_set_q": (n, MAX_DMW, WIND_FEAT),
        "wind_mask_q": (n, MAX_DMW),
        "y": (n, npix),
        "y_mask": (n, npix),
        "last": (n, npix),
        "last_mask": (n, npix),
    }


def _open_memmaps(
    dest: Path, shapes: dict[str, tuple[int, ...]], mode: str
) -> dict[str, np.memmap]:
    dest.mkdir(parents=True, exist_ok=True)
    return {
        key: np.memmap(dest / f"{key}.npy", mode=mode, dtype=np.float32, shape=shape)
        for key, shape in shapes.items()
    }


def _norm_from_dict(raw: dict) -> BundleNorm:
    wind_raw = raw.get("wind")
    if not wind_raw:
        dummy = ChannelNorm(0.0, 1.0)
        wind = tuple(dummy for _ in range(WIND_FEAT))
    else:
        wind = tuple(ChannelNorm(**w) for w in wind_raw)
    return BundleNorm(
        sky=ChannelNorm(**raw["sky"]),
        c13=ChannelNorm(**raw["c13"]),
        acha=ChannelNorm(**raw["acha"]),
        wind=wind,  # type: ignore[arg-type]
    )


class PairsDataset(Dataset):
    """Memmap (or in-RAM) encoded bundles. ``__getitem__`` returns CPU tensors.

    Workers reopen the memmaps so the full set never needs to be resident.
    """

    def __init__(
        self,
        *,
        arrays: dict[str, np.ndarray] | None = None,
        dest: Path | None = None,
        shapes: dict[str, tuple[int, ...]] | None = None,
        device: torch.device,
        norm: BundleNorm | None = None,
        n: int | None = None,
    ) -> None:
        self._arrays = arrays
        self._dest = Path(dest) if dest is not None else None
        self._shapes = shapes
        self.device = torch.device(device)
        self.norm = norm
        if n is not None:
            self.n = int(n)
        elif arrays is not None:
            self.n = int(next(iter(arrays.values())).shape[0])
        elif shapes is not None:
            self.n = int(next(iter(shapes.values()))[0])
        else:
            raise ValueError("PairsDataset needs arrays, shapes, or n")

    @classmethod
    def from_rows(
        cls,
        rows: list[dict],
        norm: BundleNorm,
        device: torch.device,
    ) -> PairsDataset:
        encoded = [norm.apply_row(r) for r in rows]
        arrays = {key: _stack(encoded, key) for key in ARRAY_KEYS}
        return cls(arrays=arrays, device=device, norm=norm)

    @classmethod
    def from_memmap(cls, dest: Path, device: torch.device) -> PairsDataset:
        dest = Path(dest)
        meta = json.loads((dest / "meta.json").read_text())
        shapes = {k: tuple(v) for k, v in meta["shapes"].items()}
        return cls(
            dest=dest,
            shapes=shapes,
            device=device,
            norm=_norm_from_dict(meta["norm"]),
            n=int(meta["n"]),
        )

    def _ensure_arrays(self) -> dict[str, np.ndarray]:
        if self._arrays is None:
            if self._dest is None or self._shapes is None:
                raise RuntimeError("memmap dataset has no path to reopen")
            self._arrays = _open_memmaps(self._dest, self._shapes, mode="r")
        return self._arrays

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        arrays = self._ensure_arrays()
        i = int(idx)
        return {
            key: torch.from_numpy(np.array(arr[i], dtype=np.float32, copy=True))
            for key, arr in arrays.items()
        }


def _dataset_root(dataset: Dataset) -> PairsDataset:
    cur: object = dataset
    while True:
        if isinstance(cur, Subset):
            cur = cur.dataset
            continue
        inner = getattr(cur, "_inner", None)
        if inner is not None and not isinstance(cur, PairsDataset):
            cur = inner
            continue
        break
    if not isinstance(cur, PairsDataset):
        raise TypeError(f"expected PairsDataset, got {type(cur)}")
    return cur


def _worker_init(_worker_id: int) -> None:
    """Reopen memmaps in each worker so they do not share the parent's maps."""
    info = torch.utils.data.get_worker_info()
    if info is None:
        return
    root = _dataset_root(info.dataset)
    root._arrays = None
    if root._dest is not None:
        root._ensure_arrays()
    cur: object = info.dataset
    seen: set[int] = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        reopen = getattr(cur, "reopen_extra", None)
        if reopen is not None:
            reopen()
        if isinstance(cur, Subset):
            cur = cur.dataset
            continue
        inner = getattr(cur, "_inner", None)
        cur = inner if inner is not None and inner is not cur else None


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


_LABEL_KEYS = ("y", "y_mask", "last", "last_mask")


def apply_batch_preprocess(
    model: nn.Module, batch: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Optional per-batch input transform on already-cached tensors.

    ``model.preprocess_batch`` may rewrite sky/sat/meta/wind features. Loss
    targets (``y``, ``last``, masks) are restored so the objective cannot
    be changed. No recache: this runs after the memmap DataLoader.
    """
    fn = getattr(model, "preprocess_batch", None)
    if fn is None:
        return batch
    labels = {k: batch[k].clone() for k in _LABEL_KEYS if k in batch}
    out = fn(dict(batch))
    if not isinstance(out, dict):
        raise TypeError("preprocess_batch must return a dict of tensors")
    out.update(labels)
    return out


class ContiguousBlockRandomSampler(Sampler[int]):
    """Shuffle memmap runs; walk each run in row order.

    Global window shuffle random-reads a working set larger than RAM.
    Keep-sets are night-contiguous; a 10% day-stratified draw is not, so
    prefer ``runs=`` grouped by night. Shuffle nights, stream each night
    in memmap order so the page cache stays hot.
    """

    def __init__(
        self,
        row_ids: np.ndarray | None = None,
        generator: torch.Generator | None = None,
        *,
        runs: list[np.ndarray] | None = None,
    ):
        self.generator = generator
        if runs is not None:
            self.runs = [np.asarray(r, dtype=np.int64) for r in runs if len(r)]
            self.n = int(sum(int(r.size) for r in self.runs))
            return
        if row_ids is None:
            raise ValueError("ContiguousBlockRandomSampler needs row_ids or runs")
        ids = np.asarray(row_ids, dtype=np.int64)
        if ids.ndim != 1:
            raise ValueError(f"row_ids must be 1-d, got {ids.shape}")
        self.n = int(ids.size)
        if self.n == 0:
            self.runs = []
        else:
            breaks = np.flatnonzero(np.diff(ids) != 1) + 1
            bounds = np.concatenate(([0], breaks, [self.n]))
            self.runs = [
                np.arange(int(bounds[i]), int(bounds[i + 1]), dtype=np.int64)
                for i in range(len(bounds) - 1)
            ]

    @property
    def n_runs(self) -> int:
        return len(self.runs)

    def __iter__(self):
        n_runs = len(self.runs)
        if n_runs == 0:
            return
        order = torch.randperm(n_runs, generator=self.generator).tolist()
        for i in order:
            yield from self.runs[i].tolist()

    def __len__(self) -> int:
        return self.n


def _memmap_row_ids(dataset: Dataset) -> np.ndarray:
    _, rows = _compose_base_indices(dataset)
    return rows


def _night_runs(dataset: Dataset) -> list[np.ndarray] | None:
    """Local indices grouped by pair date, each group in memmap order."""
    dates = _pair_dates_for(dataset)
    if dates is None:
        return None
    _, rows = _compose_base_indices(dataset)
    by: dict[str, list[int]] = {}
    for i, d in enumerate(dates):
        by.setdefault(str(d), []).append(int(i))
    runs: list[np.ndarray] = []
    for loc in by.values():
        idx = np.asarray(loc, dtype=np.int64)
        runs.append(idx[np.argsort(rows[idx], kind="stable")])
    return runs


def order_indices_by_memmap(dataset: Dataset, indices: list[int]) -> list[int]:
    """Sort dataset-local indices so the loader streams memmap rows."""
    _, rows = _compose_base_indices(dataset)
    idx = np.asarray(indices, dtype=np.int64)
    return idx[np.argsort(rows[idx], kind="stable")].tolist()


def make_loader(
    dataset: Dataset,
    cfg: TrainConfig,
    *,
    shuffle: bool,
    device: torch.device,
    block_shuffle: bool = False,
) -> DataLoader:
    kwargs: dict = {
        "batch_size": cfg.batch_size,
        "shuffle": False,
        "num_workers": int(cfg.num_workers),
        "pin_memory": device.type == "cuda",
        "drop_last": False,
    }
    if cfg.num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = int(cfg.prefetch_factor)
        kwargs["worker_init_fn"] = _worker_init
        # Fork so workers share the encoded cache instead of re-importing it.
        kwargs["multiprocessing_context"] = mp.get_context("fork")
    gen = None
    if shuffle:
        gen = torch.Generator()
        gen.manual_seed(int(cfg.seed))
        kwargs["generator"] = gen
        sampler = None
        kind = "block"
        if block_shuffle:
            night_runs = _night_runs(dataset)
            if night_runs is not None and len(night_runs) > 1:
                sampler = ContiguousBlockRandomSampler(
                    runs=night_runs, generator=gen
                )
                kind = "night"
            else:
                sampler = ContiguousBlockRandomSampler(
                    _memmap_row_ids(dataset), generator=gen
                )
        if sampler is not None and sampler.n_runs > 1:
            kwargs["sampler"] = sampler
            print(
                f"  loader {kind}-shuffle  {sampler.n_runs} runs  n={len(sampler)}",
                flush=True,
            )
        else:
            kwargs["shuffle"] = True
    return DataLoader(dataset, **kwargs)


def _compose_base_indices(dataset: Dataset) -> tuple[object, np.ndarray]:
    """Walk nested Subset / keep-index wrappers back to the memmap root."""
    idx = np.arange(len(dataset), dtype=np.int64)
    cur: object = dataset
    while True:
        if isinstance(cur, Subset):
            idx = np.asarray(cur.indices, dtype=np.int64)[idx]
            cur = cur.dataset
            continue
        inner = getattr(cur, "_inner", None)
        remap = getattr(cur, "_idx", None)
        if inner is not None and not isinstance(cur, PairsDataset):
            if remap is not None:
                idx = np.asarray(remap, dtype=np.int64)[idx]
            cur = inner
            continue
        break
    return cur, idx


def _pair_dates_for(dataset: Dataset) -> np.ndarray | None:
    """Pair-evening dates aligned to ``dataset`` items, or None if unknown."""
    dates = getattr(dataset, "dates", None)
    if dates is not None and len(np.asarray(dates)) == len(dataset):
        return np.array([str(d) for d in np.asarray(dates)], dtype=object)
    root, rows = _compose_base_indices(dataset)
    dates = getattr(root, "dates", None)
    if dates is None:
        dest = getattr(root, "_dest", None)
        if dest is None:
            return None
        npz = Path(dest) / "times.npz"
        if not npz.is_file():
            return None
        dates = np.load(npz)["dates"]
    dates = np.asarray(dates)
    if rows.size and int(rows.max()) >= dates.shape[0]:
        raise ValueError(
            f"pair dates n={dates.shape[0]} cannot index row {int(rows.max())}"
        )
    return np.array([str(d) for d in dates[rows]], dtype=object)


def _stratified_choice(
    groups: list[np.ndarray],
    n_keep: int,
    rng: np.random.Generator,
) -> list[int]:
    """Sample ``n_keep`` indices with at least one from each group.

    Leftover slots follow group size (largest remainder). This is not
    "keep a fraction of the groups".
    """
    sizes = np.array([int(g.size) for g in groups], dtype=np.int64)
    n_days = int(len(groups))
    n = int(sizes.sum())
    n_keep = int(min(n, max(n_days, n_keep)))
    leftover = n_keep - n_days
    extra = np.zeros(n_days, dtype=np.int64)
    cap = np.maximum(sizes - 1, 0)
    if leftover > 0 and int(sizes.sum()) > n_days:
        weights = sizes.astype(np.float64)
        raw = leftover * (weights / weights.sum())
        extra = np.floor(raw).astype(np.int64)
        rem = leftover - int(extra.sum())
        order = np.argsort(-(raw - extra), kind="stable")
        for i in order:
            if rem <= 0:
                break
            extra[i] += 1
            rem -= 1
        extra = np.minimum(extra, cap)
        deficit = leftover - int(extra.sum())
        if deficit > 0:
            room = cap - extra
            for i in np.argsort(-sizes, kind="stable"):
                if deficit <= 0:
                    break
                take = min(int(room[i]), deficit)
                extra[i] += take
                deficit -= take
    quotas = np.minimum(sizes, 1 + extra)
    short = n_keep - int(quotas.sum())
    if short > 0:
        room = sizes - quotas
        for i in np.argsort(-sizes, kind="stable"):
            if short <= 0:
                break
            take = min(int(room[i]), short)
            quotas[i] += take
            short -= take
    kept: list[np.ndarray] = []
    for grp, q in zip(groups, quotas, strict=True):
        q = int(q)
        if q >= grp.size:
            chosen = grp
        else:
            chosen = rng.choice(grp, size=q, replace=False)
        kept.append(np.asarray(chosen, dtype=np.int64))
    out = np.concatenate(kept) if kept else np.zeros(0, dtype=np.int64)
    rng.shuffle(out)
    return out.tolist()


def _frac_indices(
    n: int,
    frac: float,
    seed: int,
    *,
    strata: np.ndarray | None = None,
) -> list[int]:
    """Seeded sample of about ``frac * n`` indices, without replacement.

    When ``strata`` (pair dates) is given, every day is represented and the
    draw is taken inside each day — not by dropping days.
    """
    frac = float(frac)
    if n <= 0 or frac >= 1.0:
        return list(range(n))
    if frac <= 0.0:
        raise ValueError(f"train_frac must be > 0, got {frac}")
    n_keep = min(n, max(1, int(round(frac * n))))
    rng = np.random.default_rng(int(seed))
    if strata is None:
        return rng.choice(n, size=n_keep, replace=False).tolist()
    labels = np.asarray(strata)
    if labels.shape[0] != n:
        raise ValueError(f"strata length {labels.shape[0]} != n={n}")
    by: dict[str, list[int]] = {}
    for i, lab in enumerate(labels):
        by.setdefault(str(lab), []).append(i)
    keys = sorted(by)
    groups = [np.asarray(by[k], dtype=np.int64) for k in keys]
    return _stratified_choice(groups, n_keep, rng)


def train_frac_indices(
    dataset: Dataset, frac: float, seed: int
) -> tuple[list[int], dict]:
    """``frac`` of windows, uniform across every training day.

    Requires pair dates so a 10% draw cannot collapse to 10% of nights.
    """
    n = len(dataset)
    strata = _pair_dates_for(dataset)
    if float(frac) < 1.0 and strata is None:
        raise ValueError(
            "train_frac < 1 needs pair dates (times.npz or dataset.dates) "
            "so the draw covers every training day instead of dropping nights"
        )
    keep = order_indices_by_memmap(dataset, _frac_indices(n, frac, seed, strata=strata))
    info = {
        "n": n,
        "n_keep": len(keep),
        "frac": float(frac),
        "seed": int(seed),
        "stratified": strata is not None,
        "n_days": 0,
        "n_days_keep": 0,
    }
    if strata is not None:
        src_days = {str(d) for d in strata}
        keep_days = {str(strata[i]) for i in keep}
        info["n_days"] = len(src_days)
        info["n_days_keep"] = len(keep_days)
        if keep_days != src_days:
            missing = sorted(src_days - keep_days)
            raise RuntimeError(
                f"day-stratified train_frac missed nights: {missing[:8]}"
            )
    return keep, info


def _cache_signature(
    files: list[Path],
    *,
    n_frames: int,
    sat_size: int,
    min_nobs: int,
) -> dict:
    return {
        "n": len(files),
        "n_frames": int(n_frames),
        "sat_size": int(sat_size),
        "min_nobs": int(min_nobs),
        "max_dmw": int(MAX_DMW),
        "wind_feat": int(WIND_FEAT),
        "first": files[0].name if files else "",
        "last": files[-1].name if files else "",
    }


def _cache_matches(dest: Path, signature: dict) -> dict | None:
    meta_path = dest / "meta.json"
    if not meta_path.exists():
        return None
    old = json.loads(meta_path.read_text())
    if all(old.get(k) == signature[k] for k in signature):
        return old
    return None


def encode_split_cache(
    files: list[Path],
    dest: Path,
    *,
    split: str,
    n_frames: int,
    sat_size: int,
    min_nobs: int,
    norm: BundleNorm,
    verbose: bool = True,
) -> Path:
    """Write normalized tensors to memmaps. Reuses dest if the signature matches."""
    dest = Path(dest)
    signature = _cache_signature(
        files, n_frames=n_frames, sat_size=sat_size, min_nobs=min_nobs
    )
    if _cache_matches(dest, signature) is not None:
        if verbose:
            print(f"  {split}: reusing cache {dest}  n={len(files)}", flush=True)
        return dest

    if not files:
        raise ValueError(f"no {split} bundles to encode")
    probe = encode_file(
        files[0], split, n_frames=n_frames, sat_size=sat_size, min_nobs=min_nobs
    )
    npix = int(probe["sky"].shape[-1])
    shapes = _array_shapes(len(files), n_frames, npix, sat_size)
    maps = _open_memmaps(dest, shapes, mode="w+")
    for i, path in enumerate(
        _progress(files, desc=f"encode {split}", total=len(files), disable=not verbose)
    ):
        enc = norm.apply_row(
            encode_file(
                path, split, n_frames=n_frames, sat_size=sat_size, min_nobs=min_nobs
            )
        )
        for key in ARRAY_KEYS:
            maps[key][i] = enc[key]
    for arr in maps.values():
        arr.flush()
    meta = {
        **signature,
        "npix": npix,
        "nside": int(round(math.sqrt(npix / 12.0))),
        "shapes": {k: list(v) for k, v in shapes.items()},
        "norm": asdict(norm),
    }
    (dest / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    if verbose:
        print(f"  {split}: wrote cache {dest}  n={len(files)}", flush=True)
    return dest


def load_split(
    pairs_dir: Path,
    split: str,
    *,
    n_frames: int,
    sat_size: int,
    min_nobs: int,
    verbose: bool = True,
) -> list[dict]:
    files = list_bundles(pairs_dir, split)
    rows: list[dict] = []
    if not files:
        if verbose:
            print(f"  {split}: 0 bundles", flush=True)
        return rows
    samples = iter_samples(pairs_dir / split)
    for sample in _progress(
        samples, desc=f"load {split}", total=len(files), disable=not verbose
    ):
        # Folder is the split source of truth (parquet may still say "test"
        # after a night was carved into validation/).
        sample["split"] = split
        rows.append(
            encode_bundle(
                sample, n_frames=n_frames, sat_size=sat_size, min_nobs=min_nobs
            )
        )
    if verbose:
        print(f"  {split}: {len(rows)} bundles", flush=True)
    return rows


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------


def point_residual(pred: torch.Tensor) -> torch.Tensor:
    """Median residual ``(B, npix)`` from a point map or quantile stack."""
    if pred.ndim == 2:
        return pred
    if pred.ndim != 3:
        raise ValueError(
            f"expected residual (B, npix) or (B, Q, npix), got {tuple(pred.shape)}"
        )
    q = pred.shape[1]
    idx = MEDIAN_Q if q == len(QUANTILE_TAUS) else q // 2
    return pred[:, idx]


def huber_elem(
    pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0
) -> torch.Tensor:
    err = (pred - target).abs()
    quad = err.clamp(max=delta)
    return 0.5 * quad**2 + delta * (err - quad)


def masked_per_sample_sum(
    elem: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sum of per-sample means over valid tiles, plus how many samples had mass."""
    e = elem.reshape(elem.shape[0], -1)
    m = mask.reshape(mask.shape[0], -1).to(dtype=e.dtype)
    if weights is not None:
        m = m * weights.reshape(weights.shape[0], -1).to(dtype=e.dtype)
    den = m.sum(dim=-1)
    ok = den > 0
    n = ok.sum()
    if int(n.item()) == 0:
        return e.new_zeros(()), n
    per = (e * m).sum(dim=-1) / den.clamp_min(1.0)
    return per[ok].sum(), n


def masked_reduce(
    elem: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-sample mean over valid tiles, then mean over those samples."""
    total, n = masked_per_sample_sum(elem, mask, weights)
    if int(n.item()) == 0:
        return total
    return total / n.to(dtype=total.dtype)


def masked_huber(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    delta: float = 1.0,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Masked Huber: mean over valid tiles *per sample*, then over the batch."""
    return masked_reduce(huber_elem(pred, target, delta), mask, weights)


def masked_pinball(
    pred_q: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    taus: tuple[float, ...] = QUANTILE_TAUS,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pinball on a ``(B, Q, npix)`` head. Falls back to Huber on the median."""
    if pred_q.ndim != 3 or pred_q.shape[1] != len(taus):
        return masked_huber(point_residual(pred_q), target, mask, weights=weights)
    err = target.unsqueeze(1) - pred_q
    t = pred_q.new_tensor(list(taus)).view(1, -1, 1)
    elem = torch.maximum(t * err, (t - 1.0) * err).mean(dim=1)
    return masked_reduce(elem, mask, weights)


def event_tile_weights(
    resid: torch.Tensor, alpha: float, w_max: float
) -> torch.Tensor:
    """``w = min(1 + α|r|, w_max)`` on each tile (normalized residual)."""
    if alpha <= 0:
        return torch.ones_like(resid)
    return (1.0 + alpha * resid.abs()).clamp(max=w_max)


def stratum_masks(
    resid: torch.Tensor,
    last: torch.Tensor,
    mask: torch.Tensor,
    sky_norm: ChannelNorm | None,
    resid_thresh: float,
    trans_thresh: float,
) -> dict[str, torch.Tensor]:
    valid = mask > 0
    change = valid & (resid.abs() > resid_thresh)
    last_phys = sky_norm.decode_t(last) if sky_norm is not None else last
    cloud = valid & (last_phys < trans_thresh)
    return {"all": valid, "change": change, "cloud": cloud}


class _Moments:
    """Running population variance over masked tiles."""

    def __init__(self) -> None:
        self.n = 0
        self.sx = 0.0
        self.sx2 = 0.0

    def add(self, x: torch.Tensor, m: torch.Tensor) -> None:
        v = x.detach().reshape(-1)[m.reshape(-1) > 0]
        if v.numel() == 0:
            return
        v = v.double()
        self.n += int(v.numel())
        self.sx += float(v.sum())
        self.sx2 += float((v * v).sum())

    def var(self) -> float:
        if self.n < 2:
            return float("nan")
        mean = self.sx / self.n
        return max(self.sx2 / self.n - mean * mean, 0.0)


def _var_ratio(pred_m: _Moments, resid_m: _Moments) -> float:
    vr = resid_m.var()
    vp = pred_m.var()
    if not np.isfinite(vr) or not np.isfinite(vp) or vr <= 1e-12:
        return float("nan")
    return float(vp / vr)


class _SplitDiag:
    """Unweighted Huber / persistence / sharpness, stratified on a split."""

    keys = ("all", "change", "cloud")

    def __init__(self) -> None:
        self.huber = {k: 0.0 for k in self.keys}
        self.persist = {k: 0.0 for k in self.keys}
        self.n = {k: 0 for k in self.keys}
        self.pred_m = {k: _Moments() for k in self.keys}
        self.resid_m = {k: _Moments() for k in self.keys}
        self.pinball_sum = 0.0
        self.pinball_n = 0
        self.pct_sum = 0.0
        self.pct_n = 0

    def add(
        self,
        pred: torch.Tensor,
        resid: torch.Tensor,
        last: torch.Tensor,
        mask: torch.Tensor,
        y: torch.Tensor,
        sky_norm: ChannelNorm | None,
        cfg: TrainConfig,
        pred_q: torch.Tensor | None = None,
    ) -> None:
        he = huber_elem(pred, resid)
        pe = huber_elem(torch.zeros_like(resid), resid)
        strata = stratum_masks(
            resid,
            last,
            mask,
            sky_norm,
            cfg.event_resid_thresh,
            cfg.cloud_trans_thresh,
        )
        for k, m in strata.items():
            hs, hn = masked_per_sample_sum(he, m)
            ps, _pn = masked_per_sample_sum(pe, m)
            self.huber[k] += float(hs)
            self.persist[k] += float(ps)
            self.n[k] += int(hn)
            self.pred_m[k].add(pred, m)
            self.resid_m[k].add(resid, m)
        if (
            pred_q is not None
            and pred_q.ndim == 3
            and pred_q.shape[1] == len(QUANTILE_TAUS)
        ):
            err = resid.unsqueeze(1) - pred_q
            t = pred_q.new_tensor(list(QUANTILE_TAUS)).view(1, -1, 1)
            tile = torch.maximum(t * err, (t - 1.0) * err).mean(dim=1)
            pbs, pbn = masked_per_sample_sum(tile, mask)
            self.pinball_sum += float(pbs)
            self.pinball_n += int(pbn)
        frac_sum, n_bright = masked_pct_error(pred, last, y, mask, sky_norm)
        self.pct_sum += float(frac_sum)
        self.pct_n += int(n_bright)

    def as_metrics(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for k in self.keys:
            n = self.n[k]
            h = self.huber[k] / n if n else float("nan")
            p = self.persist[k] / n if n else float("nan")
            out[f"huber_{k}"] = h
            out[f"persist_{k}"] = p
            out[f"imp_{k}"] = improvement_pct(h, p)
            out[f"sharpness_{k}"] = _var_ratio(self.pred_m[k], self.resid_m[k])
        out["pinball"] = (
            self.pinball_sum / self.pinball_n if self.pinball_n else float("nan")
        )
        out["pct_error"] = (
            100.0 * self.pct_sum / self.pct_n if self.pct_n else float("nan")
        )
        return out


def persistence_huber(
    resid: torch.Tensor, mask: torch.Tensor, delta: float = 1.0
) -> torch.Tensor:
    """Huber of the zero residual — persistence assumes the sky does not change."""
    return masked_huber(torch.zeros_like(resid), resid, mask, delta)


def _json_num(x: float) -> float | None:
    x = float(x)
    return x if np.isfinite(x) else None


def improvement_pct(model_loss: float, persist_loss: float) -> float:
    """``100 * (1 - model / persist)``. NaN if persistence Huber is ~0."""
    if not np.isfinite(model_loss) or not np.isfinite(persist_loss) or persist_loss <= 1e-12:
        return float("nan")
    return float(100.0 * (1.0 - model_loss / persist_loss))


def truth_pct_errors(
    err: np.ndarray, truth: np.ndarray, floor: float = TRUTH_FLOOR
) -> tuple[float, float]:
    """Mean and median ``100 * |pred - y| / |y|`` on tiles with ``|y| >= floor``."""
    bright = np.abs(truth) >= floor
    if not np.any(bright):
        return float("nan"), float("nan")
    frac = np.abs(err[bright] / truth[bright])
    return float(100.0 * np.mean(frac)), float(100.0 * np.median(frac))


def masked_pct_error(
    pred_resid: torch.Tensor,
    last: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    sky_norm: ChannelNorm | None = None,
    floor: float = TRUTH_FLOOR,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sum of ``|pred - y| / |y|`` and tile count, in decoded sky units."""
    pred = last + pred_resid
    if sky_norm is not None:
        pred = sky_norm.decode_t(pred)
        y = sky_norm.decode_t(y)
    bright = (mask > 0) & (y.abs() >= floor)
    n = bright.sum()
    if int(n.item()) == 0:
        return pred.new_zeros(()), n
    frac = (pred - y).abs()[bright] / y.abs()[bright].clamp_min(1e-6)
    return frac.sum(), n


@dataclass
class TrainConfig:
    epochs: int = 2000
    lr: float = 1e-4
    min_lr: float = 1e-6
    weight_decay: float = 5e-2
    dropout: float = 0.4
    #: Extra Huber toward a zero residual. Default 0: event weights already
    #: pull the fit toward change, and a persistence attractor fights that.
    persist_coef: float = 0.0
    #: Train-only tile weight ``min(1 + α|r|, w_max)`` on the residual.
    event_alpha: float = 2.0
    event_w_max: float = 15.0
    #: Normalized |residual| cut for the "change" stratum in the epoch log.
    event_resid_thresh: float = 0.05
    #: Decoded last-frame transmission cut for the cloudy val ranking metric.
    cloud_trans_thresh: float = 0.9
    batch_size: int = 8
    num_workers: int = 4
    prefetch_factor: int = 2
    modality_drop: float = 0.3
    frame_drop: float = 0.2
    #: Epochs with no validation improvement before the LR is halved.
    patience: int = 10
    #: Stop after this many plateau-triggered halves (and one last plateau).
    max_lr_halves: int = 4
    inner_val_frac: float = 0.25
    #: Fraction of train *windows* used each run.
    #: Drawn inside every training day; not "keep this fraction of nights".
    train_frac: float = 0.25
    #: Fraction of val windows scored each epoch. 1.0 = full val every epoch.
    val_frac: float = 1.0
    seed: int = 0
    n_frames: int = 8
    sat_size: int = 48
    d_model: int = 48
    n_heads: int = 4
    n_layers: int = 2
    sky_width: int = 12
    min_nobs: int = DREAM_MIN_NOBS
    device: str | None = None
    verbose: bool = True
    history: list = field(default_factory=list)


def apply_eval_budget(cfg: TrainConfig) -> TrainConfig:
    """Return ``cfg`` unchanged. Kept so training entry points share one hook."""
    return cfg


def _progress(iterable, *, desc: str, total: int | None = None, leave: bool = True,
              disable: bool = False):
    from tqdm import tqdm

    return tqdm(
        iterable,
        desc=desc,
        total=total,
        leave=leave,
        dynamic_ncols=True,
        disable=disable,
    )



def save_checkpoint(
    path: Path,
    model: nn.Module,
    norm: BundleNorm,
    cfg: TrainConfig,
    info: dict,
    scores: dict,
) -> None:
    """Atomically overwrite ``path`` (and a ``.json`` sidecar) so a crash cannot
    leave a half-written checkpoint.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sky_mean = getattr(model, "sky_mean", None)
    sky_std = getattr(model, "sky_std", None)
    payload = {
        "state_dict": {
            k.replace("._orig_mod.", "."): v for k, v in model.state_dict().items()
        },
        "norm": asdict(norm),
        "nside": model.nside,
        "n_frames": model.n_frames,
        "d_model": int(getattr(model, "d_model", 0) or 0),
        "arch": getattr(model, "name", None),
        "sky_mean": None if sky_mean is None else float(sky_mean.detach()),
        "sky_std": None if sky_std is None else float(sky_std.detach()),
        "config": {k: v for k, v in asdict(cfg).items() if k != "history"},
        "info": info,
        "scores": scores,
    }
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    side = path.with_suffix(".json")
    side_tmp = side.with_name(side.name + ".tmp")
    side_tmp.write_text(json.dumps({"info": info, "scores": scores}, indent=2) + "\n")
    side_tmp.replace(side)


