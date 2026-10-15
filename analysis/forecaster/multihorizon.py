"""Multi-horizon labels on the T=24 cache, without touching paper / t24 files.

The recache writes ``y_h`` / ``y_h_mask`` at 90 s, 5 min, 20 min, and 30 min.
``y.npy`` / ``y_mask.npy`` stay the +20 min maps (hardlinked from t24, which
is hardlinked from the paper 8-frame cache). ``PairsDataset`` therefore
sees the same +20 min tensors as before when pointed at this directory
with ``--n-frames 24``.

Wrap with ``MultiHorizonPairs`` (every lead) or ``HorizonSelectPairs``
(one lead) *before* ``attach_horizon_lock``. Paper path
(``pairs/dream_20min_encoded``) is a different directory and is not read
or written here.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

HORIZONS_SEC: tuple[int, ...] = (90, 300, 1200, 1800)
HORIZON_LABELS: dict[int, str] = {
    90: "90s",
    300: "5min",
    1200: "20min",
    1800: "30min",
}
DEFAULT_HORIZON_SEC = 1200
Y_H_NAME = "y_h.npy"
Y_H_MASK_NAME = "y_h_mask.npy"
HIT_NAME = "horizon_hit.npy"
DT_NAME = "horizon_dt_sec.npy"


def lead_feat_hours(hours: float) -> np.float32:
    """Same ``meta[..., 8]`` convention as ``retarget_20min`` / ``retarget_2min``."""
    return np.float32(math.tanh(float(hours) / 6.0))


def log_lead_feat(horizon_sec: float, ref_sec: float = float(DEFAULT_HORIZON_SEC)) -> np.float32:
    """Lead embedding with dynamic range across 90 s–30 min. Not stored in the cache."""
    return np.float32(math.log(max(float(horizon_sec), 1.0) / float(ref_sec)))


def load_spec(cache_split: Path) -> dict:
    dest = Path(cache_split)
    meta = json.loads((dest / "meta.json").read_text())
    spec = meta.get("multihorizon")
    if not spec:
        raise FileNotFoundError(
            f"{dest}: no meta.json['multihorizon'] (not a multi-horizon cache)"
        )
    return spec


def is_multihorizon_cache(cache_split: Path) -> bool:
    path = Path(cache_split) / "meta.json"
    if not path.is_file():
        return False
    try:
        return "multihorizon" in json.loads(path.read_text())
    except json.JSONDecodeError:
        return False


def horizon_index(horizon_sec: int, horizons: tuple[int, ...] | list[int] = HORIZONS_SEC) -> int:
    hs = int(horizon_sec)
    try:
        return list(horizons).index(hs)
    except ValueError as exc:
        raise ValueError(f"horizon {hs}s not in {list(horizons)}") from exc


def _copy_attrs(dst, src) -> None:
    for name in ("n", "device", "norm", "unix", "dates", "_dest", "_shapes"):
        if hasattr(src, name):
            setattr(dst, name, getattr(src, name))


class HorizonSelectPairs(Dataset):
    """Slice ``y_h[:, k]`` into ``y`` / ``y_mask`` for one lead.

    ``horizon_sec=1200`` is a pass-through: ``y.npy`` is already the paper/t24
    +20 min label, so those tensors match the single-lead cache.
    """

    def __init__(
        self,
        base: Dataset,
        cache_split: Path,
        *,
        horizon_sec: int = DEFAULT_HORIZON_SEC,
    ) -> None:
        self.base = base
        self._inner = base
        _copy_attrs(self, base)
        self.n = int(getattr(base, "n", len(base)))
        self.cache_split = Path(cache_split)
        spec = load_spec(self.cache_split)
        horizons = tuple(int(h) for h in spec["horizons_sec"])
        self.horizons_sec = horizons
        self.horizon_sec = int(horizon_sec)
        self.k = horizon_index(self.horizon_sec, horizons)
        self.default_horizon_sec = int(spec.get("default_horizon_sec", DEFAULT_HORIZON_SEC))
        self._passthrough = self.horizon_sec == self.default_horizon_sec
        self._y_h: np.memmap | None = None
        self._y_h_mask: np.memmap | None = None
        self._hit: np.memmap | None = None
        if not self._passthrough:
            self.reopen_extra()

    def reopen_extra(self) -> None:
        inner = getattr(self.base, "reopen_extra", None)
        if inner is not None:
            inner()
        if self._passthrough:
            return
        spec = load_spec(self.cache_split)
        shapes = {k: tuple(v) for k, v in spec["shapes"].items()}
        dest = self.cache_split
        self._y_h = np.memmap(dest / Y_H_NAME, mode="r", dtype=np.float32, shape=shapes["y_h"])
        self._y_h_mask = np.memmap(
            dest / Y_H_MASK_NAME, mode="r", dtype=np.float32, shape=shapes["y_h_mask"]
        )
        self._hit = np.memmap(dest / HIT_NAME, mode="r", dtype=np.float32, shape=shapes["horizon_hit"])

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        batch = self.base[int(idx)]
        i = int(idx)
        k = self.k
        if not self._passthrough:
            if self._y_h is None:
                self.reopen_extra()
            assert self._y_h is not None and self._y_h_mask is not None and self._hit is not None
            batch["y"] = torch.from_numpy(np.array(self._y_h[i, k], dtype=np.float32, copy=True))
            batch["y_mask"] = torch.from_numpy(
                np.array(self._y_h_mask[i, k], dtype=np.float32, copy=True)
            )
            batch["horizon_hit"] = torch.tensor(float(self._hit[i, k]), dtype=torch.float32)
        batch["horizon_sec"] = torch.tensor(float(self.horizon_sec), dtype=torch.float32)
        batch["lead_sec"] = batch["horizon_sec"]
        return batch


class MultiHorizonPairs(Dataset):
    """Return every lead: ``y`` / ``y_mask`` are ``(K, npix)``."""

    def __init__(self, base: Dataset, cache_split: Path) -> None:
        self.base = base
        self._inner = base
        _copy_attrs(self, base)
        self.n = int(getattr(base, "n", len(base)))
        self.cache_split = Path(cache_split)
        spec = load_spec(self.cache_split)
        self.horizons_sec = tuple(int(h) for h in spec["horizons_sec"])
        self.k = len(self.horizons_sec)
        self._y_h: np.memmap | None = None
        self._y_h_mask: np.memmap | None = None
        self._hit: np.memmap | None = None
        self._dt: np.memmap | None = None
        self.reopen_extra()

    def reopen_extra(self) -> None:
        inner = getattr(self.base, "reopen_extra", None)
        if inner is not None:
            inner()
        spec = load_spec(self.cache_split)
        shapes = {k: tuple(v) for k, v in spec["shapes"].items()}
        dest = self.cache_split
        self._y_h = np.memmap(dest / Y_H_NAME, mode="r", dtype=np.float32, shape=shapes["y_h"])
        self._y_h_mask = np.memmap(
            dest / Y_H_MASK_NAME, mode="r", dtype=np.float32, shape=shapes["y_h_mask"]
        )
        self._hit = np.memmap(dest / HIT_NAME, mode="r", dtype=np.float32, shape=shapes["horizon_hit"])
        dt_path = dest / DT_NAME
        self._dt = None
        if dt_path.is_file():
            self._dt = np.memmap(
                dt_path, mode="r", dtype=np.float32, shape=shapes["horizon_dt_sec"]
            )

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        batch = self.base[int(idx)]
        if self._y_h is None:
            self.reopen_extra()
        assert self._y_h is not None and self._y_h_mask is not None and self._hit is not None
        i = int(idx)
        batch["y"] = torch.from_numpy(np.array(self._y_h[i], dtype=np.float32, copy=True))
        batch["y_mask"] = torch.from_numpy(np.array(self._y_h_mask[i], dtype=np.float32, copy=True))
        batch["horizon_hit"] = torch.from_numpy(np.array(self._hit[i], dtype=np.float32, copy=True))
        batch["horizon_sec"] = torch.tensor(self.horizons_sec, dtype=torch.float32)
        batch["lead_sec"] = batch["horizon_sec"]
        if self._dt is not None:
            batch["horizon_dt_sec"] = torch.from_numpy(
                np.array(self._dt[i], dtype=np.float32, copy=True)
            )
        return batch


def window_mean_t_horizons(
    cache_split: Path,
    keep: np.ndarray,
    *,
    mean: float,
    std: float,
    chunk: int = 512,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Per-window mean observed transmission at every cached lead. ``(n_keep, K)``.

    NaN if that lead has no valid tiles or ``horizon_hit==0``.
    """
    dest = Path(cache_split)
    spec = load_spec(dest)
    horizons = tuple(int(h) for h in spec["horizons_sec"])
    shapes = {k: tuple(v) for k, v in spec["shapes"].items()}
    y = np.memmap(dest / Y_H_NAME, mode="r", dtype=np.float32, shape=shapes["y_h"])
    ym = np.memmap(dest / Y_H_MASK_NAME, mode="r", dtype=np.float32, shape=shapes["y_h_mask"])
    hit = np.memmap(dest / HIT_NAME, mode="r", dtype=np.float32, shape=shapes["horizon_hit"])
    keep = np.asarray(keep, dtype=np.int64)
    k = len(horizons)
    out = np.full((keep.size, k), np.nan, dtype=np.float64)
    mu, sd = float(mean), float(std)
    step = max(1, int(chunk))
    for j in range(0, keep.size, step):
        ix = keep[j : j + step]
        yy = np.asarray(y[ix])
        valid = np.asarray(ym[ix]) > 0
        hh = np.asarray(hit[ix]) > 0
        t = yy * sd + mu
        n_valid = valid.reshape(valid.shape[0], k, -1).sum(axis=-1).astype(np.float64)
        acc = np.where(valid, t, 0.0).reshape(t.shape[0], k, -1).sum(axis=-1)
        ok = (n_valid > 0) & hh
        chunk_out = np.full((yy.shape[0], k), np.nan, dtype=np.float64)
        chunk_out[ok] = acc[ok] / n_valid[ok]
        out[j : j + ix.size] = chunk_out
    return out, horizons


def open_horizon_maps(cache_split: Path, horizon_sec: int) -> tuple[np.memmap, np.memmap, np.memmap]:
    """Memmap views of ``y_h[:, k]``, ``y_h_mask[:, k]``, ``horizon_hit[:, k]``."""
    dest = Path(cache_split)
    spec = load_spec(dest)
    horizons = tuple(int(h) for h in spec["horizons_sec"])
    k = horizon_index(int(horizon_sec), horizons)
    shapes = {name: tuple(v) for name, v in spec["shapes"].items()}
    y = np.memmap(dest / Y_H_NAME, mode="r", dtype=np.float32, shape=shapes["y_h"])
    ym = np.memmap(dest / Y_H_MASK_NAME, mode="r", dtype=np.float32, shape=shapes["y_h_mask"])
    hit = np.memmap(dest / HIT_NAME, mode="r", dtype=np.float32, shape=shapes["horizon_hit"])
    return y[:, k], ym[:, k], hit[:, k]


def wrap_horizon(
    data: Dataset,
    cache_split: Path,
    *,
    horizon_sec: int | None = None,
) -> Dataset:
    """Slice one lead. ``horizon_sec=None`` leaves ``data`` unchanged."""
    if not is_multihorizon_cache(cache_split):
        return data
    if horizon_sec is None:
        return data
    return HorizonSelectPairs(data, cache_split, horizon_sec=int(horizon_sec))


def wrap_multihorizon(data: Dataset, cache_split: Path) -> Dataset:
    """Stack every lead into ``y`` / ``y_mask`` as ``(K, npix)``."""
    dest = Path(cache_split)
    if not is_multihorizon_cache(dest):
        raise FileNotFoundError(
            f"{dest}: no meta.json['multihorizon'] — point at dream_multih_t24_encoded"
        )
    return MultiHorizonPairs(data, dest)
