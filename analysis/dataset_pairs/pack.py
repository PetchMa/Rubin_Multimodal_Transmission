"""Pack all-sky windows into self-contained Parquet bundles.

One file = one training bundle (a single-row Parquet):

* 1 h of all-sky transmission (HEALPix nested) ending at ``t_end``
* 1 h of causal GOES-East cutouts (C13, ACHA, DMW) available at ``t_end``
* metadata (times, geometry, split, file names)
* the measured all-sky map at ``t_end + 20 min``

Nights without GOES are skipped, so this can run while ``fetch_dream`` is
still filling ``data/satellite/dream/``. Existing bundle files are left
alone unless ``--force``, or unless ``sat_acha`` is still the legacy 10 km
grid (those are rewritten automatically).

Usage
-----
    python -m analysis.dataset_pairs.pack --all-windows
    python -m analysis.dataset_pairs.pack --dates 2025-06-27 --windows-per-night 8
    python -m analysis.dataset_pairs.pack --status
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from analysis.satellite.archive import (
    SCAN_LATENCY_SEC,
    WindArchive,
    available_dream_nights,
    dmw_path,
    jd_to_unix,
    load_grid,
)
from analysis.eval_harness.data import HISTORY_SEC, HORIZON_SEC, ROOT
from analysis.eval_harness.dream_data import (
    DREAM_MIN_NOBS,
    DREAM_WINDOWS_PER_NIGHT,
    apply_validation_carve,
    load_night,
    load_split_index,
    nights_for,
    pick_evenly,
    write_split_json,
)
from analysis.eval_harness.samples import build_window_samples
from analysis.satellite.goes import SITE_LAT, SITE_LON, ACHA_2KM_MIN_AXIS
from analysis.satellite.winds import pressure_to_height_km
from analysis.paths import MNT

PROTOCOL = "north_star_1h_20m_v1"
DEFAULT_OUT = MNT / "pairs" / "dream_20min"
REPO_LINK = ROOT / "data/pairs/dream_20min"
MANIFEST = "manifest.json"
#: Extra GOES scans kept before the camera window so ``t_end - 1 h`` is covered.
SAT_PAD_SEC = 1200.0
LIST_COLS = ("camera_jd", "sat_unix", "sat_acha_unix")


def pairs_dir(out_dir: Path | None = None) -> Path:
    dest = Path(out_dir) if out_dir is not None else DEFAULT_OUT
    dest.mkdir(parents=True, exist_ok=True)
    if not REPO_LINK.exists():
        REPO_LINK.parent.mkdir(parents=True, exist_ok=True)
        try:
            REPO_LINK.symlink_to(dest.resolve(), target_is_directory=True)
        except FileExistsError:
            pass
    return dest


def bundle_path(dest: Path, split: str, date: str, sample_id: str) -> Path:
    """``train|test/<date>/<sample_id>.parquet`` — one file per bundle."""
    return dest / split / date / f"{sample_id}.parquet"


def _utc(unix: float) -> str:
    return dt.datetime.fromtimestamp(float(unix), tz=dt.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _blob(a: np.ndarray, dtype) -> bytes:
    return np.ascontiguousarray(a, dtype=dtype).tobytes()


def _array(blob: bytes, shape: tuple[int, ...], dtype) -> np.ndarray:
    return np.frombuffer(blob, dtype=dtype).reshape(shape).copy()


def _split_of(date: str, fold: str = "even_train") -> str:
    if date in nights_for("train", fold):
        return "train"
    if date in nights_for("validation", fold):
        return "validation"
    if date in nights_for("test", fold):
        return "test"
    return "unassigned"


def _sat_indices(unix: np.ndarray, t_end_unix: float, history_sec: float) -> np.ndarray:
    available = unix + SCAN_LATENCY_SEC
    keep = (available <= t_end_unix) & (
        unix >= t_end_unix - history_sec - SAT_PAD_SEC
    )
    return np.flatnonzero(keep)


def _dmw_slice(date: str, t0: float, t1: float) -> dict[str, np.ndarray]:
    path = dmw_path(date)
    empty = {
        k: np.zeros(0, dtype=np.float64)
        for k in ("time", "u", "v", "height_km", "lat", "lon", "pressure")
    }
    if not path.is_file():
        return empty
    d = np.load(path)
    t = np.asarray(d["time"], dtype=np.float64)
    keep = (t >= t0) & (t <= t1)
    if not np.any(keep):
        return empty
    p = np.asarray(d["pressure"], dtype=np.float64)[keep]
    return {
        "time": t[keep],
        "u": np.asarray(d["u"], dtype=np.float64)[keep],
        "v": np.asarray(d["v"], dtype=np.float64)[keep],
        "height_km": pressure_to_height_km(p),
        "lat": np.asarray(d["lat"], dtype=np.float64)[keep],
        "lon": np.asarray(d["lon"], dtype=np.float64)[keep],
        "pressure": p,
    }


def build_rows(
    date: str,
    *,
    windows_per_night: int | None = DREAM_WINDOWS_PER_NIGHT,
    history_sec: float = HISTORY_SEC,
    horizon_sec: float = HORIZON_SEC,
    min_nobs: int = DREAM_MIN_NOBS,
    fold: str = "even_train",
) -> list[dict]:
    """Build in-memory bundle dicts for one DREAM night with GOES."""
    night = load_night(date)
    samples = build_window_samples(
        night["jd"],
        history_sec=history_sec,
        horizon_sec=horizon_sec,
    )
    samples = pick_evenly(samples, windows_per_night)
    if not samples:
        return []

    c13 = load_grid("c13", date)
    acha = load_grid("acha", date)
    wind = WindArchive.load(date, min_height_km=4.0)
    split = _split_of(date, fold)
    nside = int(night["nside"])
    npix = int(night["npix"])
    rows: list[dict] = []

    for s in samples:
        idx = np.asarray(s.window_indices, dtype=int)
        t_end_unix = float(jd_to_unix(s.t_end_jd))
        t_tgt_unix = float(jd_to_unix(s.t_target_jd))
        si = _sat_indices(c13.unix, t_end_unix, history_sec)
        ai = _sat_indices(acha.unix, t_end_unix, history_sec)
        if si.size == 0 or ai.size == 0:
            continue
        t0 = t_end_unix - history_sec - SAT_PAD_SEC
        dmw = _dmw_slice(date, t0, t_end_unix)
        u, v, h = wind.at(s.t_end_jd) if wind.unix.size else (0.0, 0.0, 8.0)
        cam_T = night["T"][idx]
        cam_nobs = night["nobs"][idx]
        tgt_T = night["T"][s.target_idx]
        tgt_nobs = night["nobs"][s.target_idx]
        valid_end = int(
            np.sum(np.isfinite(cam_T[-1]) & (cam_nobs[-1] >= min_nobs))
        )
        valid_tgt = int(np.sum(np.isfinite(tgt_T) & (tgt_nobs >= min_nobs)))
        last_scan = float(c13.unix[si[-1]])
        names = night["names"]
        rows.append(
            {
                "sample_id": f"{date}_{int(t_end_unix)}",
                "protocol": PROTOCOL,
                "date": date,
                "split": split,
                "fold": fold,
                "product": night.get("product", "zps"),
                "nside": nside,
                "npix": npix,
                "scheme": str(night.get("scheme", "NEST")),
                "min_nobs": int(min_nobs),
                "history_sec": float(history_sec),
                "horizon_sec": float(horizon_sec),
                "site_lat": float(SITE_LAT),
                "site_lon": float(SITE_LON),
                "t_end_jd": float(s.t_end_jd),
                "t_target_jd": float(s.t_target_jd),
                "dt_ahead_days": float(s.dt_ahead_days),
                "t_end_unix": t_end_unix,
                "t_target_unix": t_tgt_unix,
                "t_end_utc": _utc(t_end_unix),
                "t_target_utc": _utc(t_tgt_unix),
                "n_camera_frames": int(idx.size),
                "camera_jd": np.asarray(night["jd"][idx], dtype=np.float64).tolist(),
                "camera_T": _blob(cam_T, np.float32),
                "camera_nobs": _blob(cam_nobs, np.int16),
                "camera_name_start": str(names[int(idx[0])]),
                "camera_name_end": str(names[int(idx[-1])]),
                "n_valid_end": valid_end,
                "target_T": _blob(tgt_T, np.float32),
                "target_nobs": _blob(tgt_nobs, np.int16),
                "target_name": str(names[int(s.target_idx)]),
                "n_valid_target": valid_tgt,
                "satellite": "goes19",
                "sat_n_scans": int(si.size),
                "sat_ny": int(c13.values.shape[1]),
                "sat_nx": int(c13.values.shape[2]),
                "sat_unix": np.asarray(c13.unix[si], dtype=np.float64).tolist(),
                "sat_c13": _blob(c13.values[si], np.float32),
                "sat_lat": _blob(c13.lat, np.float32),
                "sat_lon": _blob(c13.lon, np.float32),
                "sat_acha_n": int(ai.size),
                "sat_acha_ny": int(acha.values.shape[1]),
                "sat_acha_nx": int(acha.values.shape[2]),
                "sat_acha_unix": np.asarray(acha.unix[ai], dtype=np.float64).tolist(),
                "sat_acha": _blob(acha.values[ai], np.float32),
                "sat_acha_lat": _blob(acha.lat, np.float32),
                "sat_acha_lon": _blob(acha.lon, np.float32),
                "sat_scan_age_sec": float(t_end_unix - last_scan - SCAN_LATENCY_SEC),
                "sat_u": float(u),
                "sat_v": float(v),
                "sat_height_km": float(h),
                "dmw_n": int(dmw["u"].size),
                "dmw_time": _blob(dmw["time"], np.float64),
                "dmw_u": _blob(dmw["u"], np.float32),
                "dmw_v": _blob(dmw["v"], np.float32),
                "dmw_height_km": _blob(dmw["height_km"], np.float32),
                "dmw_lat": _blob(dmw["lat"], np.float32),
                "dmw_lon": _blob(dmw["lon"], np.float32),
            }
        )
    return rows


def _table(rows: list[dict]) -> pa.Table:
    if not rows:
        raise ValueError("no rows to write")
    arrays = {}
    for key in rows[0]:
        vals = [r[key] for r in rows]
        if key in LIST_COLS:
            arrays[key] = pa.array(vals, type=pa.list_(pa.float64()))
        elif isinstance(vals[0], (bytes, bytearray)):
            arrays[key] = pa.array(vals, type=pa.large_binary())
        else:
            arrays[key] = pa.array(vals)
    return pa.table(arrays)


def _bundle_acha_is_2km(path: Path) -> bool:
    try:
        table = pq.read_table(path, columns=["sat_acha_ny"])
        return int(table.column("sat_acha_ny")[0].as_py()) >= ACHA_2KM_MIN_AXIS
    except Exception:  # noqa: BLE001
        return False


def write_bundle(row: dict, dest: Path, *, force: bool = False) -> Path | None:
    """Write one self-contained single-row Parquet. Returns path or None if skipped."""
    path = bundle_path(dest, row["split"], row["date"], row["sample_id"])
    if path.is_file() and not force and _bundle_acha_is_2km(path):
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    table = _table([row])
    table = table.replace_schema_metadata(
        {
            "protocol": PROTOCOL,
            "sample_id": row["sample_id"],
            "date": row["date"],
            "split": row["split"],
        }
    )
    pq.write_table(table, path, compression="zstd", compression_level=3)
    return path


def pack_night(
    date: str,
    *,
    out_dir: Path | None = None,
    windows_per_night: int | None = DREAM_WINDOWS_PER_NIGHT,
    force: bool = False,
    **kwargs,
) -> list[Path]:
    dest = pairs_dir(out_dir)
    leftover = dest / f"{date}.parquet"
    if leftover.is_file():
        leftover.unlink()
    for split in ("train", "validation", "test"):
        legacy = dest / split / f"{date}.parquet"
        if legacy.is_file():
            legacy.unlink()
        # Old flat layout: train/<sample_id>.parquet without a date folder.
        flat = dest / split
        if flat.is_dir():
            for p in flat.glob(f"{date}_*.parquet"):
                p.unlink()
    rows = build_rows(date, windows_per_night=windows_per_night, **kwargs)
    if not rows:
        print(f"  {date}: no packable windows")
        return []
    written: list[Path] = []
    skipped = 0
    for row in rows:
        path = write_bundle(row, dest, force=force)
        if path is None:
            skipped += 1
        else:
            written.append(path)
    size_mb = sum(p.stat().st_size for p in written) / 1e6
    print(
        f"  {date}: {len(written)} bundles written  "
        f"{skipped} skipped  {size_mb:.2f} MB"
    )
    return written


def load_table(path: Path) -> pa.Table:
    return pq.read_table(path)


def decode_sample(row: dict) -> dict:
    """Turn one Parquet row (as dict) into numpy arrays for training."""
    n_frames = int(row["n_camera_frames"])
    npix = int(row["npix"])
    n_scans = int(row["sat_n_scans"])
    ny, nx = int(row["sat_ny"]), int(row["sat_nx"])
    n_acha = int(row["sat_acha_n"])
    any_, anx = int(row["sat_acha_ny"]), int(row["sat_acha_nx"])
    n_dmw = int(row["dmw_n"])
    camera_T = _array(row["camera_T"], (n_frames, npix), np.float32)
    return {
        "sample_id": row["sample_id"],
        "date": row["date"],
        "split": row["split"],
        "nside": int(row["nside"]),
        "npix": npix,
        "scheme": row["scheme"],
        "history_sec": float(row["history_sec"]),
        "horizon_sec": float(row["horizon_sec"]),
        "t_end_jd": float(row["t_end_jd"]),
        "t_target_jd": float(row["t_target_jd"]),
        "dt_ahead_days": float(row["dt_ahead_days"]),
        "t_end_utc": row["t_end_utc"],
        "t_target_utc": row["t_target_utc"],
        "field_window": camera_T,
        "jd_window": np.asarray(row["camera_jd"], dtype=np.float64),
        "nobs_window": _array(row["camera_nobs"], (n_frames, npix), np.int16),
        "target": _array(row["target_T"], (npix,), np.float32),
        "target_nobs": _array(row["target_nobs"], (npix,), np.int16),
        "sat_c13": _array(row["sat_c13"], (n_scans, ny, nx), np.float32),
        "sat_unix": np.asarray(row["sat_unix"], dtype=np.float64),
        "sat_lat": _array(row["sat_lat"], (ny, nx), np.float32),
        "sat_lon": _array(row["sat_lon"], (ny, nx), np.float32),
        "sat_acha": _array(row["sat_acha"], (n_acha, any_, anx), np.float32),
        "sat_acha_unix": np.asarray(row["sat_acha_unix"], dtype=np.float64),
        "sat_acha_lat": _array(row["sat_acha_lat"], (any_, anx), np.float32),
        "sat_acha_lon": _array(row["sat_acha_lon"], (any_, anx), np.float32),
        "sat_u": float(row["sat_u"]),
        "sat_v": float(row["sat_v"]),
        "sat_height_km": float(row["sat_height_km"]),
        "dmw": {
            "time": _array(row["dmw_time"], (n_dmw,), np.float64),
            "u": _array(row["dmw_u"], (n_dmw,), np.float32),
            "v": _array(row["dmw_v"], (n_dmw,), np.float32),
            "height_km": _array(row["dmw_height_km"], (n_dmw,), np.float32),
            "lat": _array(row["dmw_lat"], (n_dmw,), np.float32),
            "lon": _array(row["dmw_lon"], (n_dmw,), np.float32),
        },
        "meta": {
            k: row[k]
            for k in (
                "protocol",
                "product",
                "fold",
                "min_nobs",
                "site_lat",
                "site_lon",
                "satellite",
                "camera_name_start",
                "camera_name_end",
                "target_name",
                "n_valid_end",
                "n_valid_target",
                "sat_scan_age_sec",
            )
        },
    }


def iter_samples(path: Path):
    """Yield decoded bundles from one file or a directory tree of them."""
    path = Path(path)
    files = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
    for f in files:
        table = load_table(f)
        if table.num_rows != 1:
            raise ValueError(f"{f} has {table.num_rows} rows; expected one bundle")
        row = {name: table.column(name)[0].as_py() for name in table.column_names}
        yield decode_sample(row)


def _bundle_files(dest: Path) -> list[Path]:
    return sorted(p for p in dest.rglob("*.parquet") if p.name != MANIFEST)


def _write_manifest(dest: Path) -> dict:
    files = _bundle_files(dest)
    nights: dict[str, dict] = {}
    n_train = n_val = n_test = 0
    bytes_total = 0
    for f in files:
        folder_split = f.parent.parent.name
        date = f.parent.name
        split = (
            folder_split
            if folder_split in {"train", "validation", "test"}
            else "unassigned"
        )
        rec = nights.setdefault(date, {"date": date, "split": split, "n": 0, "mb": 0.0})
        rec["n"] += 1
        rec["mb"] = round(rec["mb"] + f.stat().st_size / 1e6, 3)
        n_train += int(split == "train")
        n_val += int(split == "validation")
        n_test += int(split == "test")
        bytes_total += f.stat().st_size
    payload = {
        "protocol": PROTOCOL,
        "layout": "one parquet file per bundle under {split}/{date}/",
        "n_nights": len(nights),
        "n_bundles": len(files),
        "n_train": n_train,
        "n_validation": n_val,
        "n_test": n_test,
        "size_mb": round(bytes_total / 1e6, 2),
        "history_sec": HISTORY_SEC,
        "horizon_sec": HORIZON_SEC,
        "nights": sorted(nights.values(), key=lambda r: r["date"]),
    }
    (dest / MANIFEST).write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def carve_validation_pairs(out_dir: Path | None = None) -> dict:
    """Move 10% of paired test nights into ``validation/`` (no recopy).

    Updates the frozen even/odd JSON first, then renames
    ``test/<date>/`` → ``validation/<date>/`` for those nights.
    """
    idx = apply_validation_carve(load_split_index())
    write_split_json(idx)
    dest = pairs_dir(out_dir)
    val_dir = dest / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)
    moved = []
    for date in nights_for("validation", index=idx):
        src = dest / "test" / date
        dst = val_dir / date
        if dst.exists():
            moved.append(date)
            continue
        if not src.is_dir():
            print(f"  missing pair folder {src}")
            continue
        src.rename(dst)
        moved.append(date)
    man = _write_manifest(dest)
    man["validation_nights"] = list(nights_for("validation", index=idx))
    man["moved"] = moved
    return man


def status(out_dir: Path | None = None) -> dict:
    dest = pairs_dir(out_dir)
    files = _bundle_files(dest)
    dates = sorted({p.parent.name for p in files}) if files else []
    paired = available_dream_nights()
    return {
        "out": str(dest),
        "n_bundles": len(files),
        "n_nights_packed": len(dates),
        "n_goes_ready": len(paired),
        "packed_dates": dates,
        "unpacked": [d for d in paired if d not in dates],
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dates", nargs="+", help="YYYY-MM-DD subset (default: all GOES-ready).")
    p.add_argument(
        "--windows-per-night",
        type=int,
        default=None,
        help="Evenly spaced windows per night. Default: all overlapping windows.",
    )
    p.add_argument(
        "--all-windows",
        action="store_true",
        help="Pack every valid 1 h → +20 min window (same as omitting --windows-per-night).",
    )
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--status", action="store_true")
    p.add_argument(
        "--carve-validation",
        action="store_true",
        help="Move 10% of test nights into validation/ (night-wise, no recopy).",
    )
    args = p.parse_args(argv)

    dest = pairs_dir(args.out)
    if args.carve_validation:
        man = carve_validation_pairs(dest)
        print(
            f"validation: {len(man.get('validation_nights', []))} nights  "
            f"{man.get('n_validation', 0)} bundles\n"
            f"  nights: {', '.join(man.get('validation_nights', []))}\n"
            f"  remaining test: {man.get('n_test', 0)} bundles  "
            f"train: {man.get('n_train', 0)} bundles"
        )
        return
    if args.status:
        st = status(dest)
        print(
            f"pairs {st['n_bundles']} bundles / {st['n_nights_packed']} nights "
            f"/ {st['n_goes_ready']} GOES-ready\n  {st['out']}"
        )
        print("  train / validation / test from even_train fold")
        if st["unpacked"]:
            print(f"  waiting on pack: {len(st['unpacked'])} nights")
        return

    per_night = None if args.all_windows or args.windows_per_night is None else args.windows_per_night
    dates = list(args.dates) if args.dates else available_dream_nights()
    print(
        f"Packing {len(dates)} nights → {dest}\n"
        f"  protocol={PROTOCOL}  1 file/bundle  "
        f"windows/night={'all' if per_night is None else per_night}  "
        f"split=even_train"
    )
    n_ok = n_fail = 0
    for i, date in enumerate(dates, 1):
        print(f"[{i}/{len(dates)}] {date}", flush=True)
        try:
            written = pack_night(
                date,
                out_dir=dest,
                windows_per_night=per_night,
                force=args.force,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL {date}: {type(exc).__name__}: {exc}")
            n_fail += 1
            continue
        n_ok += len(written)
    man = _write_manifest(dest)
    print(
        f"done: wrote {n_ok} bundles  failed nights {n_fail}  "
        f"total {man['n_bundles']}  {man['size_mb']:.1f} MB"
    )


if __name__ == "__main__":
    main()
