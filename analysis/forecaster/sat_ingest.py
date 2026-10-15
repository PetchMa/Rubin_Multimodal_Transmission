"""Overlay the night GOES archive onto forecaster camera windows.

The encoded pair cache still stores ``sat`` as ``(T, 2, 48, 48)`` = C13 + ACHA.
Those arrays have no timestamps, so this module rebuilds ``(date, unix)`` from
the pair Parquets (same ``linspace`` subsample as ``encode_bundle``), then
replaces ``batch["sat"]`` with the 18-channel GOES cube
(original 14 plus C08/C09/C10 water vapour plus ``is_missing``). Empty
DMW frames get a single C13 optical-flow point when ``of.npz`` exists.

Alignment is causal on the C13 10-min spine: for each of the 8 camera-frame
times, take the latest ABI scan at or before that instant. LST is allowed
40 min of slop (hourly product). BTDs are computed at load, not stored.

    python -u -m analysis.forecaster.train --report-only
"""

from __future__ import annotations

import datetime as dt
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from analysis.our_model.model import (
    JD_UNIX_EPOCH,
    PairsDataset,
    _causal_index,
    _open_memmaps,
    _subsample_idx,
    list_bundles,
)
from analysis.our_model.sat_pretrain.dataset import (
    ACHA_FILL,
    ALIGN_SLOP_SEC,
    C07_FILL,
    C13_FILL,
    C15_FILL,
    CHANNEL_NAMES,
    LST_FILL,
    NATIVE_IR_HW,
    TPW_FILL,
    _align,
    _resize,
    encode_cube,
    ir_is_missing,
)
from analysis.satellite.c13_flow import OF_NAME
from analysis.satellite.fetch_pretrain import DEFAULT_OUT, MODEL_PRODUCTS
from analysis.satellite.goes import ACHA_2KM_PRODUCT

WV_PRODUCTS = ("c08", "c09", "c10")
CUBE_PRODUCTS = MODEL_PRODUCTS + WV_PRODUCTS
OF_HEIGHT_KM = 8.0
DEFAULT_HORIZON_SEC = 1200.0

SITE = "pachon"
N_SAT_CH = len(CHANNEL_NAMES)
TIMES_NAME = "times.npz"
TIMES_META = "times.json"
SAT17_NAME = "sat17.npy"
SAT17_META = "sat17.json"
# On-disk sat17.npy is the 17 thermal/WV planes. ``is_missing`` is a
# sidecar so we do not rewrite the 167 GB file when only the mask changes.
SAT17_FILE_NAMES = tuple(n for n in CHANNEL_NAMES if n != "is_missing")
N_SAT17_FILE_CH = len(SAT17_FILE_NAMES)
SAT_MISSING_NAME = "sat_missing.npy"
SAT_MISSING_META = "sat_missing.json"
WIND_OF_NAMES = {
    "wind_set": "wind_set_of.npy",
    "wind_mask": "wind_mask_of.npy",
    "wind_set_q": "wind_set_q_of.npy",
    "wind_mask_q": "wind_mask_q_of.npy",
}
# Baked path replaces ``sat`` and DMW winds. Do not open those maps.
_BAKED_BASE_KEYS = (
    "sky",
    "sky_mask",
    "meta",
    "meta_q",
    "y",
    "y_mask",
    "last",
    "last_mask",
)
# Camera pair ``date`` is the UTC calendar day of the frames. Archive
# folders are the evening the nautical night starts (~22:50 UTC → ~10:30
# next day). A 00:00 UTC window on date D lives in evening D−1.
NIGHT_SLOP_SEC = 15 * 60
# Live native-IR loads keep a few evenings in RAM (each ~40 MB).
_NIGHT_LRU = 8

# Channel index of the old 2-ch cache inside the loader cube (fallback only).
_CACHE_C13 = CHANNEL_NAMES.index("c13")
_CACHE_ACHA = CHANNEL_NAMES.index("acha")
_CACHE_MISS = CHANNEL_NAMES.index("is_missing")

_FILLS = {
    "c13": C13_FILL,
    "c07": C07_FILL,
    "c15": C15_FILL,
    "acm": 0.0,
    "acha": ACHA_FILL,
    "actp": 0.0,
    "lst": LST_FILL,
    "tpw": TPW_FILL,
    "c08": C13_FILL,
    "c09": C13_FILL,
    "c10": C13_FILL,
}


def night_dir(root: Path, date: str, site: str = SITE) -> Path:
    return Path(root) / site / date


def night_products(root: Path, date: str, site: str = SITE) -> dict[str, bool]:
    d = night_dir(root, date, site)
    have = {p: (d / f"{p}.npz").is_file() for p in CUBE_PRODUCTS}
    have["of"] = (d / OF_NAME).is_file()
    return have


def night_status(root: Path, date: str, site: str = SITE) -> str:
    have = night_products(root, date, site)
    if all(have.get(p) for p in MODEL_PRODUCTS):
        return "full"
    if have.get("c13"):
        return "partial"
    return "missing"


def _shift_date(date: str, days: int) -> str:
    return (dt.date.fromisoformat(str(date)) + dt.timedelta(days=int(days))).isoformat()


def peek_unix(root: Path, date: str, site: str = SITE) -> np.ndarray | None:
    path = night_dir(root, date, site) / "c13.npz"
    if not path.is_file():
        return None
    return np.asarray(np.load(path)["unix"], dtype=np.float64)


def resolve_evening(
    root: Path,
    pair_date: str,
    frame_unix: np.ndarray,
    *,
    site: str = SITE,
    slop: float = NIGHT_SLOP_SEC,
    unix_cache: dict[str, np.ndarray | None] | None = None,
) -> str | None:
    """Archive evening whose C13 spine covers the last camera frame."""
    mid = float(np.asarray(frame_unix, dtype=np.float64)[-1])
    for delta in (0, -1, 1):
        key = _shift_date(pair_date, delta)
        if unix_cache is not None and key in unix_cache:
            unix = unix_cache[key]
        else:
            unix = peek_unix(root, key, site)
            if unix_cache is not None:
                unix_cache[key] = unix
        if unix is None or unix.size == 0:
            continue
        if float(unix[0]) - slop <= mid <= float(unix[-1]) + slop:
            return key
    return None


def load_night(root: Path, date: str, site: str = SITE) -> dict[str, np.ndarray] | None:
    """Load whatever products exist for ``site/date``. C13 is the time spine."""
    d = night_dir(root, date, site)
    c13_path = d / "c13.npz"
    if not c13_path.is_file():
        return None
    z0 = np.load(c13_path)
    unix0 = np.asarray(z0["unix"], dtype=np.float64)
    out: dict[str, np.ndarray] = {
        "unix": unix0,
        "c13": np.asarray(z0["values"], dtype=np.float32),
        "_have": ["c13"],
    }
    for name in CUBE_PRODUCTS:
        if name == "c13":
            continue
        path = d / f"{name}.npz"
        if not path.is_file():
            continue
        z = np.load(path)
        unix = np.asarray(z["unix"], dtype=np.float64)
        val = np.asarray(z["values"], dtype=np.float32)
        slop = float(ALIGN_SLOP_SEC.get(name, 8 * 60))
        out[name] = _align(unix, unix0, val, slop)
        out["_have"].append(name)
    of_path = d / OF_NAME
    if of_path.is_file():
        z = np.load(of_path)
        out["of_unix"] = np.asarray(z["unix"], dtype=np.float64)
        out["of_u"] = np.asarray(z["u"], dtype=np.float32)
        out["of_v"] = np.asarray(z["v"], dtype=np.float32)
        out["of_height_km"] = (
            np.asarray(z["height_km"], dtype=np.float32)
            if "height_km" in z.files
            else np.full(out["of_unix"].shape, OF_HEIGHT_KM, dtype=np.float32)
        )
        out["_have"].append("of")
    return out


def _raw_frames(
    night: dict[str, np.ndarray], name: str, idx: np.ndarray
) -> np.ndarray:
    if name not in night:
        return np.full((len(idx), 2, 2), np.nan, dtype=np.float32)
    return np.asarray(night[name], dtype=np.float32)[idx]


def cube_at_times(
    night: dict[str, np.ndarray],
    frame_unix: np.ndarray,
    *,
    size: int | tuple[int, int] = 48,
) -> tuple[np.ndarray, np.ndarray]:
    """``(T, 18, H, W)`` cube + per-frame sat age in seconds (causal)."""
    unix0 = np.asarray(night["unix"], dtype=np.float64)
    frame_unix = np.asarray(frame_unix, dtype=np.float64)
    idx = _causal_index(unix0, frame_unix)
    age = frame_unix - unix0[idx]
    cube = encode_cube(
        _raw_frames(night, "c13", idx),
        _raw_frames(night, "c07", idx),
        _raw_frames(night, "c15", idx),
        _raw_frames(night, "acm", idx),
        _raw_frames(night, "acha", idx),
        _raw_frames(night, "actp", idx),
        _raw_frames(night, "lst", idx),
        _raw_frames(night, "tpw", idx),
        size=size,
        c08=_raw_frames(night, "c08", idx) if "c08" in night else None,
        c09=_raw_frames(night, "c09", idx) if "c09" in night else None,
        c10=_raw_frames(night, "c10", idx) if "c10" in night else None,
    )
    return cube, age.astype(np.float32)


def missing_at_times(
    night: dict[str, np.ndarray],
    frame_unix: np.ndarray,
    *,
    size: int | tuple[int, int] = 48,
) -> np.ndarray:
    """``(T, 1, H, W)`` ``is_missing`` aligned like ``cube_at_times``."""
    unix0 = np.asarray(night["unix"], dtype=np.float64)
    frame_unix = np.asarray(frame_unix, dtype=np.float64)
    idx = _causal_index(unix0, frame_unix)
    miss = ir_is_missing(
        _raw_frames(night, "c13", idx),
        _raw_frames(night, "c07", idx),
        _raw_frames(night, "c15", idx),
        size=size,
        c08=_raw_frames(night, "c08", idx) if "c08" in night else None,
        c09=_raw_frames(night, "c09", idx) if "c09" in night else None,
        c10=_raw_frames(night, "c10", idx) if "c10" in night else None,
    )
    return miss[:, None]


def _resize_cube(cube: np.ndarray, size: int | tuple[int, int]) -> np.ndarray:
    """``(T, C, H, W)`` → target spatial size. Identity when already matched."""
    if isinstance(size, int):
        hw = (int(size), int(size))
    else:
        hw = (int(size[0]), int(size[1]))
    if cube.shape[-2:] == hw:
        return cube
    planes = [_resize(cube[:, i], hw) for i in range(cube.shape[1])]
    return np.stack(planes, axis=1).astype(np.float32, copy=False)


def pad_cache_sat(sat_2ch: np.ndarray | torch.Tensor) -> np.ndarray:
    """Last-resort: park z-scored cache C13/ACHA in their cube slots."""
    sat = np.asarray(sat_2ch, dtype=np.float32)
    t, _, h, w = sat.shape
    out = np.zeros((t, N_SAT_CH, h, w), dtype=np.float32)
    out[:, _CACHE_C13] = sat[:, 0]
    out[:, _CACHE_ACHA] = sat[:, 1]
    # No native ABI validity — treat the whole fallback cube as filled.
    out[:, _CACHE_MISS] = 1.0
    return out


def _times_signature(files: list[Path], n_frames: int) -> dict:
    return {
        "n": len(files),
        "n_frames": int(n_frames),
        "first": files[0].name if files else "",
        "last": files[-1].name if files else "",
    }


def build_times_sidecar(
    pairs_dir: Path,
    split: str,
    dest: Path,
    *,
    n_frames: int = 8,
    force: bool = False,
) -> Path:
    """Write ``dates`` + ``unix (n, T)`` aligned to the encoded memmap rows."""
    import pyarrow.parquet as pq

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    files = list_bundles(pairs_dir, split)
    sig = _times_signature(files, n_frames)
    meta_path = dest / TIMES_META
    npz_path = dest / TIMES_NAME
    if (
        not force
        and meta_path.is_file()
        and npz_path.is_file()
        and json.loads(meta_path.read_text()) == sig
    ):
        print(f"  {split}: reusing times sidecar {npz_path}  n={sig['n']}", flush=True)
        return npz_path

    dates = np.empty(len(files), dtype="U10")
    unix = np.zeros((len(files), n_frames), dtype=np.float64)
    n_cam = np.zeros(len(files), dtype=np.int32)
    for i, path in enumerate(files):
        table = pq.read_table(path, columns=["date", "camera_jd"])
        date = table.column("date")[0].as_py()
        jd = np.asarray(table.column("camera_jd")[0].as_py(), dtype=np.float64)
        pick = _subsample_idx(jd.size, n_frames)
        dates[i] = str(date)
        unix[i] = (jd[pick] - JD_UNIX_EPOCH) * 86400.0
        n_cam[i] = int(jd.size)
        if (i + 1) % 20000 == 0 or i + 1 == len(files):
            print(f"  {split}: times sidecar {i + 1}/{len(files)}", flush=True)

    np.savez_compressed(npz_path, dates=dates, unix=unix, n_camera_frames=n_cam)
    meta_path.write_text(json.dumps(sig, indent=2) + "\n")
    print(f"  {split}: wrote times sidecar {npz_path}  n={len(files)}", flush=True)
    return npz_path


def load_times_sidecar(dest: Path) -> dict[str, np.ndarray]:
    z = np.load(Path(dest) / TIMES_NAME, allow_pickle=False)
    return {
        "dates": np.asarray(z["dates"]),
        "unix": np.asarray(z["unix"], dtype=np.float64),
        "n_camera_frames": np.asarray(z["n_camera_frames"], dtype=np.int32),
    }


def _of_point_at(night: dict[str, np.ndarray], query_unix: float) -> np.ndarray | None:
    """Latest finite C13-flow vector at or before ``query_unix``."""
    unix = night.get("of_unix")
    if unix is None or unix.size == 0:
        return None
    keep = np.isfinite(unix) & (unix <= float(query_unix))
    if not np.any(keep):
        return None
    i = int(np.nonzero(keep)[0][-1])
    u = float(night["of_u"][i])
    v = float(night["of_v"][i])
    if not (np.isfinite(u) and np.isfinite(v)):
        return None
    h = float(night["of_height_km"][i])
    if not np.isfinite(h):
        h = OF_HEIGHT_KM
    return np.array([u, v, h, 0.0, 0.0, 0.0], dtype=np.float32)


def _norm_wind_point(pt: np.ndarray, norm) -> np.ndarray:
    out = np.asarray(pt, dtype=np.float32).copy()
    winds = getattr(norm, "wind", None) if norm is not None else None
    if not winds:
        return out
    for i, nrm in enumerate(winds):
        out[i] = float(nrm.encode(np.asarray([out[i]], dtype=np.float32))[0])
    return out


def apply_of_wind_fallback(
    batch: dict[str, torch.Tensor],
    night: dict[str, np.ndarray],
    frame_unix: np.ndarray,
    *,
    horizon_sec: float,
    norm,
) -> None:
    """Fill empty DMW frames with one zenith optical-flow point (in-place)."""
    if "of_u" not in night:
        return
    wind = batch["wind_set"]
    mask = batch["wind_mask"]
    t = int(wind.shape[0])
    for i in range(t):
        if float(mask[i].sum()) > 0:
            continue
        pt = _of_point_at(night, float(frame_unix[i]))
        if pt is None:
            continue
        wind[i, 0] = torch.from_numpy(_norm_wind_point(pt, norm))
        mask[i, 0] = 1.0
    if float(batch["wind_mask_q"].sum()) == 0:
        # Same instant as the last history frame — not last+horizon.
        # C13 flow at T+lead uses future ABI scans.
        pt = _of_point_at(night, float(frame_unix[-1]))
        if pt is not None:
            batch["wind_set_q"][0] = torch.from_numpy(_norm_wind_point(pt, norm))
            batch["wind_mask_q"][0] = 1.0


class RichSatDataset(PairsDataset):
    """``PairsDataset`` whose ``sat`` is the 17-channel Pachón night cube."""

    def __init__(
        self,
        base: PairsDataset,
        *,
        dates: np.ndarray,
        unix: np.ndarray,
        goes_root: Path,
        site: str = SITE,
        sat_size: int | tuple[int, int] = 48,
    ) -> None:
        self.__dict__.update(base.__dict__)
        if int(unix.shape[0]) != int(self.n):
            raise ValueError(
                f"times sidecar n={unix.shape[0]} != memmap n={self.n}"
            )
        self.dates = np.asarray(dates)
        self.unix = np.asarray(unix, dtype=np.float64)
        self.goes_root = Path(goes_root)
        self.site = site
        self.sat_size = sat_size
        self.horizon_sec = DEFAULT_HORIZON_SEC
        if self._dest is not None:
            meta_path = Path(self._dest) / "meta.json"
            if meta_path.is_file():
                meta = json.loads(meta_path.read_text())
                self.horizon_sec = float(meta.get("horizon_sec", DEFAULT_HORIZON_SEC))
        self._nights: OrderedDict[str, dict[str, np.ndarray] | None] = OrderedDict()
        self._unix: dict[str, np.ndarray | None] = {}
        # None = all camera-subsample times. Present-mask only needs now+prev.
        self.sat_n_frames: int | None = None
        self.coverage = _coverage_windows(
            self.goes_root, self.dates, self.unix, self.site
        )

    def reopen_extra(self) -> None:
        self._nights = OrderedDict()
        self._unix = {}

    def _night(self, evening: str) -> dict[str, np.ndarray] | None:
        if evening in self._nights:
            self._nights.move_to_end(evening)
            return self._nights[evening]
        night = load_night(self.goes_root, evening, self.site)
        self._nights[evening] = night
        while len(self._nights) > _NIGHT_LRU:
            self._nights.popitem(last=False)
        return night

    def resolve(self, idx: int) -> str | None:
        return resolve_evening(
            self.goes_root,
            str(self.dates[int(idx)]),
            self.unix[int(idx)],
            site=self.site,
            unix_cache=self._unix,
        )

    def cube(self, idx: int) -> tuple[np.ndarray, str, np.ndarray | None, str | None]:
        evening = self.resolve(int(idx))
        if evening is None:
            arrays = self._ensure_arrays()
            cube = _resize_cube(pad_cache_sat(arrays["sat"][int(idx)]), self.sat_size)
            return cube, "missing", None, None
        night = self._night(evening)
        if night is None:
            arrays = self._ensure_arrays()
            cube = _resize_cube(pad_cache_sat(arrays["sat"][int(idx)]), self.sat_size)
            return cube, "missing", None, None
        cube, age = cube_at_times(night, self._sat_unix(idx), size=self.sat_size)
        have = set(night.get("_have", []))
        kind = "full" if have.issuperset(MODEL_PRODUCTS) else "partial"
        return cube, kind, age, evening

    def _sat_unix(self, idx: int) -> np.ndarray:
        u = np.asarray(self.unix[int(idx)], dtype=np.float64)
        n = self.sat_n_frames
        if n is None or u.size <= int(n):
            return u
        return u[-int(n) :]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        batch = PairsDataset.__getitem__(self, idx)
        cube, _, sat_age, evening = self.cube(int(idx))
        batch["sat"] = torch.from_numpy(np.array(cube, dtype=np.float32, copy=True))
        if sat_age is None:
            sat_age = np.full(cube.shape[0], 1.0e6, dtype=np.float32)
        batch["sat_age_sec"] = torch.from_numpy(
            np.asarray(sat_age, dtype=np.float32).copy()
        )
        frame_unix = np.asarray(self.unix[int(idx)], dtype=np.float64)
        batch["unix"] = torch.from_numpy(frame_unix.copy())
        if evening is not None:
            night = self._night(evening)
            if night is not None:
                apply_of_wind_fallback(
                    batch,
                    night,
                    frame_unix,
                    horizon_sec=self.horizon_sec,
                    norm=self.norm,
                )
        return batch


class BakedRichSatDataset(PairsDataset):
    """``PairsDataset`` that slices a pre-baked 17-ch cube + OF winds."""

    def __init__(
        self,
        base: PairsDataset,
        *,
        dates: np.ndarray,
        unix: np.ndarray,
        dest: Path,
        sat_size: int = 48,
    ) -> None:
        self.__dict__.update(base.__dict__)
        dest = Path(dest)
        if int(unix.shape[0]) != int(self.n):
            raise ValueError(f"times sidecar n={unix.shape[0]} != memmap n={self.n}")
        self.dates = np.asarray(dates)
        self.unix = np.asarray(unix, dtype=np.float64)
        self.sat_size = int(sat_size)
        self._bake_dest = dest
        self._sat17: np.memmap | None = None
        self._sat_missing: np.memmap | None = None
        self._sat17_ch: int = N_SAT17_FILE_CH
        self._wind_of: dict[str, np.memmap] | None = None
        self.coverage = {
            "baked": True,
            "n_windows": int(self.n),
            "sat17": str(dest / SAT17_NAME),
        }
        self.reopen_extra()

    def _ensure_arrays(self) -> dict[str, np.ndarray]:
        if self._arrays is None:
            if self._dest is None or self._shapes is None:
                raise RuntimeError("memmap dataset has no path to reopen")
            shapes = {k: self._shapes[k] for k in _BAKED_BASE_KEYS if k in self._shapes}
            self._arrays = _open_memmaps(self._dest, shapes, mode="r")
        return self._arrays

    def reopen_extra(self) -> None:
        dest = Path(self._bake_dest)
        meta = json.loads((dest / SAT17_META).read_text())
        nch = int(meta.get("n_channels", N_SAT17_FILE_CH))
        if nch not in (N_SAT17_FILE_CH, N_SAT_CH):
            raise ValueError(f"{dest / SAT17_META}: n_channels={nch}")
        self._sat17_ch = nch
        self._sat17 = np.memmap(
            dest / SAT17_NAME,
            mode="r",
            dtype=np.float32,
            shape=(self.n, int(self.unix.shape[1]), nch, self.sat_size, self.sat_size),
        )
        self._sat_missing = None
        miss_path = dest / SAT_MISSING_NAME
        if nch == N_SAT17_FILE_CH:
            if not miss_path.is_file():
                raise FileNotFoundError(
                    f"{miss_path}: 17-ch sat17 needs is_missing sidecar"
                )
            self._sat_missing = np.memmap(
                miss_path,
                mode="r",
                dtype=np.float32,
                shape=(self.n, int(self.unix.shape[1]), 1, self.sat_size, self.sat_size),
            )
        self._wind_of = {}
        if meta.get("of_winds") and self._shapes is not None:
            for key, name in WIND_OF_NAMES.items():
                path = dest / name
                if path.is_file() and key in self._shapes:
                    self._wind_of[key] = np.memmap(
                        path, mode="r", dtype=np.float32, shape=self._shapes[key]
                    )

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        arrays = self._ensure_arrays()
        i = int(idx)
        batch = {
            key: torch.from_numpy(np.array(arr[i], dtype=np.float32, copy=True))
            for key, arr in arrays.items()
        }
        if self._sat17 is None:
            self.reopen_extra()
        assert self._sat17 is not None
        sat = np.array(self._sat17[i], dtype=np.float32, copy=True)
        if sat.shape[1] == N_SAT17_FILE_CH:
            if self._sat_missing is None:
                self.reopen_extra()
            assert self._sat_missing is not None
            miss = np.array(self._sat_missing[i], dtype=np.float32, copy=True)
            sat = np.concatenate([sat, miss], axis=1)
        batch["sat"] = torch.from_numpy(sat)
        batch["unix"] = torch.from_numpy(np.asarray(self.unix[i], dtype=np.float64).copy())
        if self._wind_of:
            for key, arr in self._wind_of.items():
                batch[key] = torch.from_numpy(np.array(arr[i], dtype=np.float32, copy=True))
        return batch


def sat17_meta_path(cache_split: Path) -> Path:
    return Path(cache_split) / SAT17_META


def sat17_is_current(cache_split: Path, n: int, *, n_frames: int, sat_size: int) -> bool:
    dest = Path(cache_split)
    meta_path = dest / SAT17_META
    sat_path = dest / SAT17_NAME
    if not meta_path.is_file() or not sat_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return False
    if int(meta.get("n", -1)) != int(n):
        return False
    if int(meta.get("n_frames", -1)) != int(n_frames):
        return False
    if int(meta.get("sat_size", -1)) != int(sat_size):
        return False
    nch = int(meta.get("n_channels", -1))
    ch = list(meta.get("channels", []))
    if meta.get("acha_product") != ACHA_2KM_PRODUCT:
        return False
    if nch == N_SAT_CH and ch == list(CHANNEL_NAMES):
        expect = n * n_frames * N_SAT_CH * sat_size * sat_size * 4
        return sat_path.stat().st_size == expect
    if nch == N_SAT17_FILE_CH and ch == list(SAT17_FILE_NAMES):
        expect = n * n_frames * N_SAT17_FILE_CH * sat_size * sat_size * 4
        return sat_path.stat().st_size == expect
    return False


def sat_missing_is_current(
    cache_split: Path, n: int, *, n_frames: int, sat_size: int
) -> bool:
    dest = Path(cache_split)
    meta_path = dest / SAT_MISSING_META
    path = dest / SAT_MISSING_NAME
    if not meta_path.is_file() or not path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return False
    if int(meta.get("n", -1)) != int(n):
        return False
    if int(meta.get("n_frames", -1)) != int(n_frames):
        return False
    if int(meta.get("sat_size", -1)) != int(sat_size):
        return False
    if list(meta.get("channels", [])) != ["is_missing"]:
        return False
    expect = n * n_frames * 1 * sat_size * sat_size * 4
    return path.stat().st_size == expect


def _coverage_windows(
    root: Path,
    dates: np.ndarray,
    unix: np.ndarray,
    site: str,
) -> dict[str, int]:
    counts = {"full": 0, "partial": 0, "missing": 0}
    unix_cache: dict[str, np.ndarray | None] = {}
    evenings: dict[str, str] = {}
    for date, frame_unix in zip(dates, unix, strict=True):
        evening = resolve_evening(
            root, str(date), frame_unix, site=site, unix_cache=unix_cache
        )
        if evening is None:
            counts["missing"] += 1
            continue
        if evening not in evenings:
            evenings[evening] = night_status(root, evening, site)
        counts[evenings[evening]] += 1
    counts["n_windows"] = int(len(dates))
    counts["n_pair_dates"] = int(len(set(map(str, dates))))
    counts["n_evenings"] = len(evenings)
    counts["evenings_full"] = sum(1 for v in evenings.values() if v == "full")
    counts["evenings_partial"] = sum(1 for v in evenings.values() if v == "partial")
    return counts


def wrap_rich_sat(
    data: PairsDataset,
    pairs_dir: Path,
    split: str,
    cache_split: Path,
    *,
    goes_root: Path | None = None,
    n_frames: int = 8,
    sat_size: int | tuple[int, int] = 48,
) -> RichSatDataset | BakedRichSatDataset:
    goes_root = Path(goes_root) if goes_root is not None else DEFAULT_OUT
    cache_split = Path(cache_split)
    npz_path = cache_split / TIMES_NAME
    reuse_times = False
    if npz_path.is_file():
        existing = load_times_sidecar(cache_split)
        if int(existing["unix"].shape[0]) == int(data.n):
            print(
                f"  {split}: reusing times sidecar {npz_path}  n={data.n}  "
                "(aligned to memmap, not current parquet glob)",
                flush=True,
            )
            reuse_times = True
    if not reuse_times:
        files = list_bundles(pairs_dir, split)
        if len(files) != int(data.n):
            raise SystemExit(
                f"{split}: times sidecar n != memmap n={data.n} and "
                f"pair list n={len(files)}; refusing to rebuild times"
            )
        build_times_sidecar(
            pairs_dir, split, cache_split, n_frames=n_frames, force=True
        )
    times = load_times_sidecar(cache_split)
    bake_ok = isinstance(sat_size, int) and sat17_is_current(
        cache_split, int(data.n), n_frames=n_frames, sat_size=sat_size
    )
    if bake_ok:
        sat_meta = json.loads(sat17_meta_path(cache_split).read_text())
        if int(sat_meta.get("n_channels", N_SAT17_FILE_CH)) == N_SAT17_FILE_CH:
            if not sat_missing_is_current(
                cache_split, int(data.n), n_frames=n_frames, sat_size=sat_size
            ):
                from analysis.forecaster.bake_rich_sat import (
                    bake_missing_split,
                )
                bake_missing_split(
                    cache_split,
                    goes_root=goes_root,
                    site=SITE,
                )
        wrapped = BakedRichSatDataset(
            data,
            dates=times["dates"],
            unix=times["unix"],
            dest=cache_split,
            sat_size=sat_size,
        )
        print(
            f"  {split}: baked sat17  n={data.n}  {N_SAT_CH}ch  "
            f"{cache_split / SAT17_NAME}  "
            f"(skip sat.npy + DMW; OF winds={bool(wrapped._wind_of)})",
            flush=True,
        )
        return wrapped
    wrapped = RichSatDataset(
        data,
        dates=times["dates"],
        unix=times["unix"],
        goes_root=goes_root,
        sat_size=sat_size,
    )
    cov = wrapped.coverage
    print(
        f"  {split}: live sat  size={sat_size}  "
        f"{cov['full']}/{cov['n_windows']} full  "
        f"{cov['partial']} partial  {cov['missing']} missing  "
        f"({cov['evenings_full']}/{cov['n_evenings']} archive evenings)",
        flush=True,
    )
    return wrapped


def _channel_stats(cube: np.ndarray) -> list[dict]:
    rows = []
    for i, name in enumerate(CHANNEL_NAMES):
        sl = cube[:, i]
        rows.append(
            {
                "channel": name,
                "index": i,
                "mean": float(sl.mean()),
                "std": float(sl.std()),
                "min": float(sl.min()),
                "max": float(sl.max()),
            }
        )
    return rows


def report_ingest(
    datasets: dict[str, RichSatDataset],
    *,
    n_samples: int = 8,
    seed: int = 0,
) -> dict:
    """Sample real windows and describe what the model now sees as ``sat``."""
    rng = np.random.default_rng(int(seed))
    out: dict = {
        "site": SITE,
        "goes_root": str(next(iter(datasets.values())).goes_root),
        "sat_shape": [8, N_SAT_CH, int(NATIVE_IR_HW[0]), int(NATIVE_IR_HW[1])],
        "channels": list(CHANNEL_NAMES),
        "products_on_disk": list(CUBE_PRODUCTS),
        "alignment": {
            "spine": "c13 10-min ABI",
            "rule": "latest scan at or before each camera-frame unix",
            "evening_resolve": "pair UTC date D → archive evening D, D-1, or D+1 whose C13 spine covers t_end",
            "lst_slop_sec": ALIGN_SLOP_SEC.get("lst", 40 * 60),
            "other_slop_sec": 8 * 60,
            "night_slop_sec": NIGHT_SLOP_SEC,
            "camera_frames": "1 h history, linspace subsample to 8",
        },
        "normalization": {
            "c13_c07_c15": "(K - 278) / 15",
            "btd_fog": "(C13 - C07) / 5",
            "btd_split": "(C13 - C15) / 5",
            "acm": "clip [0, 1]",
            "acha": "(m - 62) / 1639",
            "phase": "clip[0,4] / 4",
            "liquid": "0.5 ≤ ACTP < 2.5",
            "ice": "ACTP ≥ 3.5",
            "lst": "(K - 272) / 15",
            "tpw": "(mm - 2) / 4",
            "dc13": "ΔC13 / 15",
            "dacm": "clip ΔACM [-1, 1]",
            "c08": "(K - 278) / 15  6.2 µm upper WV",
            "c09": "(K - 278) / 15  6.9 µm mid WV",
            "c10": "(K - 278) / 15  7.3 µm low/mid WV",
            "is_missing": "1 where C13/C07/C15/(WV) was NaN before 270 K fill",
        },
        "splits": {},
        "samples": {},
    }
    for split, ds in datasets.items():
        out["splits"][split] = dict(ds.coverage)
        n = min(int(n_samples), ds.n)
        pick = rng.choice(ds.n, size=n, replace=False)
        pick.sort()
        samples = []
        for i in pick:
            cube, kind, age, evening = ds.cube(int(i))
            date = str(ds.dates[int(i)])
            unix = ds.unix[int(i)]
            have = (
                night_products(ds.goes_root, evening, ds.site)
                if evening
                else {p: False for p in MODEL_PRODUCTS}
            )
            rec = {
                "index": int(i),
                "pair_date": date,
                "goes_evening": evening,
                "source": kind,
                "products_present": {k: bool(v) for k, v in have.items()},
                "camera_unix": [float(x) for x in unix],
                "camera_span_min": float((unix[-1] - unix[0]) / 60.0),
                "sat_shape": list(cube.shape),
                "sat_age_sec": None if age is None else [float(a) for a in age],
                "sat_age_sec_max": None if age is None else float(np.max(age)),
                "channels": _channel_stats(cube),
            }
            samples.append(rec)
        out["samples"][split] = samples
    return out
