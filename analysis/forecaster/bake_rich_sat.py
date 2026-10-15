"""Bake 17-ch sat cubes, is_missing sidecar, and OF-filled winds into memmaps.

Live ``RichSatDataset`` rebuilds each window from GOES ``.npz`` in
``__getitem__``. This writes the same tensors once, night-grouped, so
training is a memmap slice.

Train float32 size is ``N × 8 × 17 × 48 × 48 × 4`` ≈ 167 GB (not 17 GB).

    python -u -m analysis.forecaster.bake_rich_sat
    python -u -m analysis.forecaster.bake_rich_sat --splits validation
"""

from __future__ import annotations

from analysis.paths import MNT

import argparse
import json
import multiprocessing as mp
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from analysis.forecaster.sat_ingest import (
    DEFAULT_HORIZON_SEC,
    N_SAT17_FILE_CH,
    SAT17_FILE_NAMES,
    SAT17_META,
    SAT17_NAME,
    SAT_MISSING_META,
    SAT_MISSING_NAME,
    SITE,
    WIND_OF_NAMES,
    apply_of_wind_fallback,
    cube_at_times,
    load_night,
    load_times_sidecar,
    missing_at_times,
    pad_cache_sat,
    resolve_evening,
    sat17_is_current,
    sat_missing_is_current,
)
from analysis.our_model.model import resolve_device
from analysis.satellite.fetch_pretrain import DEFAULT_OUT

DEFAULT_CACHE = Path(str(MNT) + "/pairs/dream_20min_encoded")
WIND_KEYS = ("wind_set", "wind_mask", "wind_set_q", "wind_mask_q")


def _open_sat17(dest: Path, n: int, n_frames: int, sat_size: int, mode: str) -> np.memmap:
    return np.memmap(
        dest / SAT17_NAME,
        mode=mode,
        dtype=np.float32,
        shape=(n, n_frames, N_SAT17_FILE_CH, sat_size, sat_size),
    )


def _open_sat_missing(
    dest: Path, n: int, n_frames: int, sat_size: int, mode: str
) -> np.memmap:
    return np.memmap(
        dest / SAT_MISSING_NAME,
        mode=mode,
        dtype=np.float32,
        shape=(n, n_frames, 1, sat_size, sat_size),
    )


def _encode_evening(job: dict) -> dict:
    dest = Path(job["dest"])
    goes_root = Path(job["goes_root"])
    idxs = [int(i) for i in job["idxs"]]
    unix = np.asarray(np.load(dest / "times.npz")["unix"], dtype=np.float64)
    n = int(job["n"])
    n_frames = int(job["n_frames"])
    sat_size = int(job["sat_size"])
    horizon = float(job["horizon_sec"])
    evening = job["evening"]
    sat = _open_sat17(dest, n, n_frames, sat_size, "r+")
    wind_src = {
        k: np.memmap(dest / f"{k}.npy", mode="r", dtype=np.float32, shape=tuple(job["wind_shapes"][k]))
        for k in WIND_KEYS
    }
    wind_dst = {
        k: np.memmap(
            dest / WIND_OF_NAMES[k],
            mode="r+",
            dtype=np.float32,
            shape=tuple(job["wind_shapes"][k]),
        )
        for k in WIND_KEYS
    }
    from analysis.our_model.model import ChannelNorm, BundleNorm

    wind_norm = tuple(ChannelNorm(**w) for w in job["wind_norm"])
    norm = BundleNorm(
        sky=ChannelNorm(0.0, 1.0),
        c13=ChannelNorm(0.0, 1.0),
        acha=ChannelNorm(0.0, 1.0),
        wind=wind_norm,  # type: ignore[arg-type]
    )
    night = None if evening is None else load_night(goes_root, evening, job["site"])
    n_full = 0
    for i in idxs:
        frame_unix = unix[i]
        if night is None:
            cube = pad_cache_sat(np.asarray(np.memmap(
                dest / "sat.npy", mode="r", dtype=np.float32, shape=tuple(job["sat2_shape"])
            )[i]))
        else:
            cube, _ = cube_at_times(night, frame_unix, size=sat_size)
            n_full += 1
        sat[i] = cube[:, :N_SAT17_FILE_CH]
        batch = {k: torch.from_numpy(np.array(wind_src[k][i], dtype=np.float32, copy=True)) for k in WIND_KEYS}
        if night is not None:
            apply_of_wind_fallback(batch, night, frame_unix, horizon_sec=horizon, norm=norm)
        for k in WIND_KEYS:
            wind_dst[k][i] = batch[k].numpy()
    sat.flush()
    for arr in wind_dst.values():
        arr.flush()
    return {"evening": evening, "n": len(idxs), "n_full": n_full}


def group_evenings(
    dest: Path,
    goes_root: Path,
    dates: np.ndarray,
    unix: np.ndarray,
    site: str,
) -> dict[str | None, list[int]]:
    groups: dict[str | None, list[int]] = defaultdict(list)
    cache: dict[str, np.ndarray | None] = {}
    for i, (date, frame_unix) in enumerate(zip(dates, unix, strict=True)):
        evening = resolve_evening(
            goes_root, str(date), frame_unix, site=site, unix_cache=cache
        )
        groups[evening].append(i)
        if (i + 1) % 20000 == 0 or i + 1 == len(dates):
            print(f"  resolve evenings {i + 1}/{len(dates)}", flush=True)
    return dict(groups)


def bake_split(
    cache_split: Path,
    *,
    goes_root: Path = DEFAULT_OUT,
    site: str = SITE,
    workers: int = 16,
    force: bool = False,
) -> dict:
    dest = Path(cache_split)
    meta = json.loads((dest / "meta.json").read_text())
    n = int(meta["n"])
    n_frames = int(meta["n_frames"])
    sat_size = int(meta["sat_size"])
    horizon = float(meta.get("horizon_sec", DEFAULT_HORIZON_SEC))
    if sat17_is_current(dest, n, n_frames=n_frames, sat_size=sat_size) and not force:
        print(f"  {dest.name}: sat17 already current  n={n}", flush=True)
        miss = bake_missing_split(
            dest, goes_root=goes_root, site=site, workers=workers, force=False
        )
        return {"split": dest.name, "skipped": True, "n": n, "missing": miss}

    times = load_times_sidecar(dest)
    if int(times["unix"].shape[0]) != n:
        raise SystemExit(f"{dest}: times n={times['unix'].shape[0]} != meta n={n}")

    gb = n * n_frames * N_SAT17_FILE_CH * sat_size * sat_size * 4 / 1e9
    print(
        f"  {dest.name}: bake sat17  n={n}  {N_SAT17_FILE_CH}ch  {sat_size}  "
        f"{gb:.1f} GB  workers={workers}",
        flush=True,
    )
    sat = _open_sat17(dest, n, n_frames, sat_size, "w+")
    del sat
    wind_shapes = {k: tuple(meta["shapes"][k]) for k in WIND_KEYS}
    for key, name in WIND_OF_NAMES.items():
        np.memmap(dest / name, mode="w+", dtype=np.float32, shape=wind_shapes[key])

    groups = group_evenings(dest, Path(goes_root), times["dates"], times["unix"], site)
    print(f"  {dest.name}: {len(groups)} evenings", flush=True)

    jobs = []
    for evening, idxs in groups.items():
        jobs.append(
            {
                "dest": str(dest),
                "goes_root": str(goes_root),
                "site": site,
                "evening": evening,
                "idxs": idxs,
                "n": n,
                "n_frames": n_frames,
                "sat_size": sat_size,
                "horizon_sec": horizon,
                "wind_shapes": {k: list(v) for k, v in wind_shapes.items()},
                "sat2_shape": list(meta["shapes"]["sat"]),
                "wind_norm": meta["norm"]["wind"],
            }
        )

    n_ok = n_win = n_full = 0
    ctx = mp.get_context("spawn")
    workers = max(1, min(int(workers), len(jobs) or 1))
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        futs = [pool.submit(_encode_evening, job) for job in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            n_ok += 1
            n_win += int(rec["n"])
            n_full += int(rec["n_full"])
            if i % 10 == 0 or i == len(jobs):
                print(
                    f"  {dest.name}: {i}/{len(jobs)} evenings  "
                    f"windows {n_win}/{n}  full {n_full}",
                    flush=True,
                )

    info = {
        "n": n,
        "n_frames": n_frames,
        "n_channels": N_SAT17_FILE_CH,
        "channels": list(SAT17_FILE_NAMES),
        "sat_size": sat_size,
        "horizon_sec": horizon,
        "of_winds": True,
        "dtype": "float32",
        "n_evenings": len(groups),
        "n_full_windows": n_full,
        "site": site,
        "goes_root": str(goes_root),
        "acha_product": "ABI-L2-ACHA2KMF",
    }
    (dest / SAT17_META).write_text(json.dumps(info, indent=2) + "\n")
    print(f"  {dest.name}: wrote {dest / SAT17_NAME}  {info}", flush=True)
    miss = bake_missing_split(
        dest, goes_root=goes_root, site=site, workers=workers, force=True
    )
    info["missing"] = miss
    return info


def _encode_missing_evening(job: dict) -> dict:
    dest = Path(job["dest"])
    goes_root = Path(job["goes_root"])
    idxs = [int(i) for i in job["idxs"]]
    unix = np.asarray(np.load(dest / "times.npz")["unix"], dtype=np.float64)
    n = int(job["n"])
    n_frames = int(job["n_frames"])
    sat_size = int(job["sat_size"])
    evening = job["evening"]
    miss = _open_sat_missing(dest, n, n_frames, sat_size, "r+")
    night = None if evening is None else load_night(goes_root, evening, job["site"])
    n_full = 0
    ones = np.ones((n_frames, 1, sat_size, sat_size), dtype=np.float32)
    for i in idxs:
        frame_unix = unix[i]
        if night is None:
            miss[i] = ones
        else:
            miss[i] = missing_at_times(night, frame_unix, size=sat_size)
            n_full += 1
    miss.flush()
    return {"evening": evening, "n": len(idxs), "n_full": n_full}


def bake_missing_split(
    cache_split: Path,
    *,
    goes_root: Path = DEFAULT_OUT,
    site: str = SITE,
    workers: int = 16,
    force: bool = False,
) -> dict:
    dest = Path(cache_split)
    meta = json.loads((dest / "meta.json").read_text())
    n = int(meta["n"])
    n_frames = int(meta["n_frames"])
    sat_size = int(meta["sat_size"])
    if sat_missing_is_current(dest, n, n_frames=n_frames, sat_size=sat_size) and not force:
        print(f"  {dest.name}: sat_missing already current  n={n}", flush=True)
        return {"split": dest.name, "skipped": True, "n": n}

    times = load_times_sidecar(dest)
    if int(times["unix"].shape[0]) != n:
        raise SystemExit(f"{dest}: times n={times['unix'].shape[0]} != meta n={n}")

    gb = n * n_frames * 1 * sat_size * sat_size * 4 / 1e9
    print(
        f"  {dest.name}: bake sat_missing  n={n}  1ch  {sat_size}  "
        f"{gb:.1f} GB  workers={workers}",
        flush=True,
    )
    miss = _open_sat_missing(dest, n, n_frames, sat_size, "w+")
    del miss

    groups = group_evenings(dest, Path(goes_root), times["dates"], times["unix"], site)
    jobs = [
        {
            "dest": str(dest),
            "goes_root": str(goes_root),
            "site": site,
            "evening": evening,
            "idxs": idxs,
            "n": n,
            "n_frames": n_frames,
            "sat_size": sat_size,
        }
        for evening, idxs in groups.items()
    ]
    n_ok = n_win = n_full = 0
    ctx = mp.get_context("spawn")
    workers = max(1, min(int(workers), len(jobs) or 1))
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        futs = [pool.submit(_encode_missing_evening, job) for job in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            n_ok += 1
            n_win += int(rec["n"])
            n_full += int(rec["n_full"])
            if i % 10 == 0 or i == len(jobs):
                print(
                    f"  {dest.name}: missing {i}/{len(jobs)} evenings  "
                    f"windows {n_win}/{n}  full {n_full}",
                    flush=True,
                )

    info = {
        "n": n,
        "n_frames": n_frames,
        "n_channels": 1,
        "channels": ["is_missing"],
        "sat_size": sat_size,
        "dtype": "float32",
        "n_evenings": len(groups),
        "n_full_windows": n_full,
        "site": site,
        "goes_root": str(goes_root),
    }
    (dest / SAT_MISSING_META).write_text(json.dumps(info, indent=2) + "\n")
    print(f"  {dest.name}: wrote {dest / SAT_MISSING_NAME}  {info}", flush=True)
    return info


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--splits", nargs="+", default=["train", "validation"])
    p.add_argument("--goes-root", type=Path, default=DEFAULT_OUT)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--missing-only",
        action="store_true",
        help="Bake the is_missing sidecar; do not rewrite sat17.npy",
    )
    args = p.parse_args(argv)
    resolve_device("cpu")
    for split in args.splits:
        dest = Path(args.cache) / split
        if not (dest / "meta.json").is_file():
            print(f"  skip {split}: no memmap at {dest}", flush=True)
            continue
        if args.missing_only:
            bake_missing_split(
                dest,
                goes_root=Path(args.goes_root),
                workers=int(args.workers),
                force=bool(args.force),
            )
            continue
        bake_split(
            dest,
            goes_root=Path(args.goes_root),
            workers=int(args.workers),
            force=bool(args.force),
        )


if __name__ == "__main__":
    main()
