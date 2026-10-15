"""Even / odd night split for the multi-night DREAM HDF5 archive.

The FITS night ``data/fits/20240219_hp4`` is one night split in time. The
DREAM dump under ``data/dream/dream_daily`` is one folder per UTC night;
adjacent nights can share weather, so a chronological half-night split is
the wrong unit. Folders missing ``zps`` or ``sys`` are skipped.

Rule (frozen in ``dream_even_odd_split.json``)
---------------------------------------------
Sort nights by UTC date. **Even index** (0, 2, 4, …) → ``train``.
**Odd index** (1, 3, 5, …) → ``test``. Consecutive archive nights therefore
land on opposite sides of the split. Each test night is bordered by train
nights when the archive has no gap. This is not a block split, and no buffer
nights are dropped. Neighbouring nights never fall in the *same* split.
Calendar-day even/odd is *not* used: a missing night would put two
same-parity dates next to each other.

Default product is ``cloud_sys``. Do not mix ``zps`` and ``sys``.

The night list in ``dream_even_odd_split.json`` is frozen. New nights are
not auto-assigned (an insert would flip every later index). Rebuild the
JSON only when you intend to redefine the split.

This module does **not** replace ``load_split`` / ``EvaluationHarness``
(those still read the NSIDE=16 FITS night). Load DREAM nights one at a time;
concatenating the split would be many GB.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator

import numpy as np

from .data import ROOT, extinction_to_transmission

DREAM_ROOT = ROOT / "data/dream/dream_daily"
SPLIT_JSON = Path(__file__).resolve().parent / "dream_even_odd_split.json"
SYMLINK_ROOT = ROOT / "data/dream/splits/even_odd"

DREAM_NSIDE = 32
DREAM_SCHEME = "NEST"
DREAM_NPIX = 12 * DREAM_NSIDE * DREAM_NSIDE  # 12288
DEFAULT_PRODUCT = "sys"
FOLDS = ("even_train", "odd_train")

# NSIDE=32 tiles are 4× smaller than the old NSIDE=16 products. MIN_NOBS=20
# there is the same star surface density as MIN_NOBS=5 here.
DREAM_MIN_NOBS = 5
#: Evenly spaced nights used to fit / tune (train) and score (test).
DREAM_TRAIN_NIGHTS = 12
#: A handful of held-out nights, not the full odd-night test fold.
DREAM_EVAL_NIGHTS = 4
DREAM_WINDOWS_PER_NIGHT = 8
#: Cap on (window, target) pairs drawn from a stacked train cube.
DREAM_MAX_TRAIN_WINDOWS = 240
#: Fraction of the odd-night (test) fold held out as validation.
VAL_FROM_TEST_FRAC = 0.10


def _night_complete(root: Path, date: str) -> bool:
    """True if both ``cloud_zps`` and ``cloud_sys`` cubes are on disk."""
    return all(
        (root / date / f"DREAM_B_cloud_{product}_{date}.hdf5").is_file()
        for product in ("zps", "sys")
    )


def discover_nights(dream_root: Path | None = None) -> list[str]:
    """UTC dates present under ``dream_daily/``, sorted.

    Incomplete folders (missing ``zps`` or ``sys``) are omitted so a
    truncated last night cannot enter the even/odd index.
    """
    root = dream_root or DREAM_ROOT
    if not root.is_dir():
        raise FileNotFoundError(
            f"DREAM daily archive not found at {root}. "
            "Expected data/dream → the dream_daily dump."
        )
    nights = sorted(
        p.name
        for p in root.iterdir()
        if p.is_dir() and _is_date(p.name) and _night_complete(root, p.name)
    )
    if not nights:
        raise FileNotFoundError(f"No YYYY-MM-DD night folders in {root}")
    return nights


def pick_evenly(items, n: int | None):
    """Keep ``n`` items spaced evenly through ``items`` (or all, if fewer)."""
    items = list(items)
    if n is None or n >= len(items):
        return items
    if n <= 0:
        return []
    idx = np.round(np.linspace(0, len(items) - 1, n)).astype(int)
    out: list = []
    seen: set[int] = set()
    for i in idx:
        i = int(i)
        if i in seen:
            continue
        seen.add(i)
        out.append(items[i])
    return out


def _is_date(name: str) -> bool:
    parts = name.split("-")
    return (
        len(parts) == 3
        and len(parts[0]) == 4
        and all(p.isdigit() for p in parts)
    )


def night_hdf5(date: str, product: str = DEFAULT_PRODUCT, *,
               dream_root: Path | None = None) -> Path:
    """Path to one night's ``cloud_{zps|sys}`` cube."""
    if product not in {"zps", "sys"}:
        raise ValueError("product must be 'zps' or 'sys'")
    root = dream_root or DREAM_ROOT
    path = root / date / f"DREAM_B_cloud_{product}_{date}.hdf5"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def build_split(nights: list[str] | None = None, *,
                dream_root: Path | None = None) -> dict:
    """Even-index nights → train, odd-index nights → test."""
    import h5py

    root = dream_root or DREAM_ROOT
    nights = list(nights if nights is not None else discover_nights(root))
    records = []
    train, test = [], []
    for i, date in enumerate(nights):
        split = "train" if i % 2 == 0 else "test"
        rec = {
            "date": date,
            "index": i,
            "parity": "even" if i % 2 == 0 else "odd",
            "split": split,
            "zps": str(night_hdf5(date, "zps", dream_root=root).relative_to(ROOT)),
            "sys": str(night_hdf5(date, "sys", dream_root=root).relative_to(ROOT)),
        }
        with h5py.File(night_hdf5(date, "zps", dream_root=root), "r") as h:
            rec["n_frames_zps"] = int(h.attrs.get("n_frames", h["jd"].shape[0]))
        with h5py.File(night_hdf5(date, "sys", dream_root=root), "r") as h:
            rec["n_frames_sys"] = int(h.attrs.get("n_frames", h["jd"].shape[0]))
        records.append(rec)
        (train if split == "train" else test).append(date)

    test, validation = _carve_validation(test, VAL_FROM_TEST_FRAC)
    val_set = set(validation)
    for rec in records:
        if rec["date"] in val_set:
            rec["split"] = "validation"

    return {
        "name": "dream_even_odd",
        "description": (
            "Night-wise train/test on the DREAM daily archive. "
            "Sorted UTC dates; even index → train, odd index → test, "
            "then 10% of test nights → validation."
        ),
        "rule": (
            "Nights sorted by UTC date. Even 0-based index → train; "
            "odd index → test. Then 10% of those test nights (evenly spaced) "
            "are carved to validation. Complementary fold odd_train swaps "
            "train with the remaining test nights; validation stays held out."
        ),
        "product_default": DEFAULT_PRODUCT,
        "root": "data/dream/dream_daily",
        "nside": DREAM_NSIDE,
        "scheme": DREAM_SCHEME,
        "npix": DREAM_NPIX,
        "n_nights": len(nights),
        "first": nights[0],
        "last": nights[-1],
        "folds": {
            "even_train": {
                "train": train,
                "validation": validation,
                "test": test,
            },
            "odd_train": {
                "train": test,
                "validation": validation,
                "test": train,
            },
        },
        "nights": records,
    }


def write_split_json(payload: dict | None = None, *,
                     path: Path | None = None,
                     dream_root: Path | None = None) -> Path:
    """Write (or refresh) the frozen even/odd split JSON."""
    path = path or SPLIT_JSON
    payload = payload if payload is not None else build_split(dream_root=dream_root)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def load_split_index(path: Path | None = None) -> dict:
    """Read the frozen even/odd split. Builds it if the JSON is missing."""
    path = path or SPLIT_JSON
    if not path.is_file():
        write_split_json(path=path)
    return json.loads(path.read_text())


def _carve_validation(test_nights: list[str], frac: float) -> tuple[list[str], list[str]]:
    """Hold out ``frac`` of test nights, spaced through the year."""
    if not test_nights or frac <= 0:
        return list(test_nights), []
    n_val = max(1, int(round(len(test_nights) * frac)))
    n_val = min(n_val, len(test_nights) - 1) if len(test_nights) > 1 else n_val
    validation = pick_evenly(test_nights, n_val)
    val_set = set(validation)
    test = [d for d in test_nights if d not in val_set]
    return test, validation


def apply_validation_carve(index: dict, *, frac: float = VAL_FROM_TEST_FRAC) -> dict:
    """Carve validation out of an existing even/odd index (idempotent)."""
    even = index["folds"]["even_train"]
    if even.get("validation"):
        return index
    test, validation = _carve_validation(list(even["test"]), frac)
    val_set = set(validation)
    even["test"] = test
    even["validation"] = validation
    odd = index["folds"]["odd_train"]
    odd["train"] = test
    odd["validation"] = validation
    odd["test"] = list(even["train"])
    for rec in index.get("nights", []):
        if rec.get("date") in val_set:
            rec["split"] = "validation"
    index["description"] = (
        "Night-wise train/test on the DREAM daily archive. "
        "Sorted UTC dates; even index → train, odd index → test, "
        f"then {frac:.0%} of test nights → validation."
    )
    index["rule"] = (
        "Nights sorted by UTC date. Even 0-based index → train; "
        "odd index → test. Then 10% of those test nights (evenly spaced) "
        "are carved to validation. Complementary fold odd_train swaps "
        "train with the remaining test nights; validation stays held out."
    )
    return index


def nights_for(split: str, fold: str = "even_train", *,
               index: dict | None = None) -> list[str]:
    """Dates assigned to ``train``, ``validation``, or ``test`` in one fold."""
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be 'train', 'validation', or 'test'")
    if fold not in FOLDS:
        raise ValueError(f"fold must be one of {FOLDS}")
    idx = index if index is not None else load_split_index()
    return list(idx["folds"][fold].get(split, []))


def write_symlinks(fold: str = "even_train", *,
                   dest: Path | None = None,
                   dream_root: Path | None = None) -> Path:
    """Point ``dest/{train,test}/<date>`` at each night folder."""
    dest = dest or SYMLINK_ROOT
    root = dream_root or DREAM_ROOT
    for split in ("train", "validation", "test"):
        split_dir = dest / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for existing in split_dir.iterdir():
            if existing.is_symlink() or existing.is_dir():
                existing.unlink()
        for date in nights_for(split, fold):
            link = split_dir / date
            target = (root / date).resolve()
            link.symlink_to(target, target_is_directory=True)
    return dest


def load_night(
    date: str,
    *,
    product: str = DEFAULT_PRODUCT,
    dream_root: Path | None = None,
    prefer_cleaned: bool = False,
) -> dict:
    """Load one night's cube in the same dict shape as ``load_split``.

    Extra keys: ``date``, ``product``, ``mask_visible``, ``sigma``,
    ``lstmin``, ``lstmax``, ``time``.
    """
    import h5py

    path = night_hdf5(date, product, dream_root=dream_root)
    with h5py.File(path, "r") as h:
        nside = int(h.attrs["NSIDE"])
        npix = int(h.attrs["NPIX"])
        scheme = str(h.attrs.get("GRID", "healpix"))
        if nside != DREAM_NSIDE or npix != DREAM_NPIX:
            raise ValueError(
                f"{path.name}: geometry NSIDE={nside} NPIX={npix} "
                f"!= expected {DREAM_NSIDE}/{DREAM_NPIX}"
            )
        clouds_key = (
            "clouds_cleaned"
            if prefer_cleaned and "clouds_cleaned" in h
            else "clouds"
        )
        nobs_key = (
            "nobs_cleaned"
            if prefer_cleaned and "nobs_cleaned" in h
            else "nobs"
        )
        A = np.asarray(h[clouds_key], dtype=np.float64)
        nobs = np.asarray(h[nobs_key], dtype=np.int32)
        jd = np.asarray(h["jd"], dtype=np.float64)
        sigma = np.asarray(h["sigma"], dtype=np.float64) if "sigma" in h else None
        mask = (
            np.asarray(h["mask_visible"], dtype=bool)
            if "mask_visible" in h
            else None
        )
        lstmin = np.asarray(h["lstmin"]) if "lstmin" in h else None
        lstmax = np.asarray(h["lstmax"]) if "lstmax" in h else None
        time = (
            np.array([t.decode() if isinstance(t, bytes) else str(t)
                      for t in h["time"][:]])
            if "time" in h
            else None
        )
        names = (
            [n.decode() if isinstance(n, bytes) else str(n)
             for n in h["filename"][:]]
            if "filename" in h
            else [f"{date}_{i:05d}" for i in range(len(jd))]
        )

    A[(nobs <= 0) | ~np.isfinite(A)] = np.nan
    order = np.argsort(jd, kind="mergesort")
    if np.any(order != np.arange(len(jd))):
        jd = jd[order]
        A = A[order]
        nobs = nobs[order]
        if sigma is not None:
            sigma = sigma[order]
        if mask is not None:
            mask = mask[order]
        if lstmin is not None:
            lstmin = lstmin[order]
        if lstmax is not None:
            lstmax = lstmax[order]
        if time is not None:
            time = time[order]
        names = [names[i] for i in order.tolist()]
    keep = np.ones(len(jd), dtype=bool)
    if len(jd) > 1:
        keep[1:] = jd[1:] > jd[:-1]
    if not np.all(keep):
        jd = jd[keep]
        A = A[keep]
        nobs = nobs[keep]
        if sigma is not None:
            sigma = sigma[keep]
        if mask is not None:
            mask = mask[keep]
        if lstmin is not None:
            lstmin = lstmin[keep]
        if lstmax is not None:
            lstmax = lstmax[keep]
        if time is not None:
            time = time[keep]
        names = [names[i] for i, k in enumerate(keep.tolist()) if k]
    if len(jd) < 2 or not np.all(np.diff(jd) > 0):
        raise ValueError(f"{date}: Julian Dates are not strictly increasing")

    return {
        "T": extinction_to_transmission(A),
        "A": A,
        "nobs": nobs,
        "jd": jd,
        "names": names,
        "split": date,
        "date": date,
        "product": product,
        "nside": nside,
        "scheme": DREAM_SCHEME,
        "npix": int(npix),
        "grid": scheme,
        "mask_visible": mask,
        "sigma": sigma,
        "lstmin": lstmin,
        "lstmax": lstmax,
        "time": time,
        "path": str(path),
    }


def stack_nights(
    dates: list[str],
    *,
    product: str = DEFAULT_PRODUCT,
    dream_root: Path | None = None,
    max_train_windows: int | None = DREAM_MAX_TRAIN_WINDOWS,
) -> dict:
    """Concatenate nights into one ``load_split``-shaped cube for fitting.

    Nights are days apart, so 1 h windows never span a concatenation gap.
    Failed nights are skipped. ``max_train_windows`` is stored on the dict so
    fitters that honour it (analog already caps itself) can subsample.
    """
    cubes = []
    used = []
    for date in dates:
        try:
            cubes.append(load_night(date, product=product, dream_root=dream_root))
            used.append(date)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {date}: {type(exc).__name__}: {exc}")
    if not cubes:
        raise RuntimeError("no DREAM nights could be loaded")
    A = np.concatenate([c["A"] for c in cubes], axis=0)
    return {
        "T": np.concatenate([c["T"] for c in cubes], axis=0),
        "A": A,
        "nobs": np.concatenate([c["nobs"] for c in cubes], axis=0),
        "jd": np.concatenate([c["jd"] for c in cubes], axis=0),
        "names": [n for c in cubes for n in c["names"]],
        "split": "train",
        "dates": used,
        "product": product,
        "nside": int(cubes[0]["nside"]),
        "scheme": DREAM_SCHEME,
        "npix": int(cubes[0]["npix"]),
        "max_train_windows": max_train_windows,
    }


def load_fit_cube(
    split: str = "train",
    *,
    fold: str = "even_train",
    n_nights: int = DREAM_TRAIN_NIGHTS,
    product: str = DEFAULT_PRODUCT,
) -> dict:
    """Evenly spaced nights from a split, stacked for analog / parametric / neural fit."""
    dates = pick_evenly(nights_for(split, fold), n_nights)
    return stack_nights(dates, product=product)


def iter_split(
    split: str,
    *,
    fold: str = "even_train",
    product: str = DEFAULT_PRODUCT,
    dream_root: Path | None = None,
) -> Iterator[dict]:
    """Yield ``load_night`` dicts for every night in a split (one at a time)."""
    for date in nights_for(split, fold):
        yield load_night(date, product=product, dream_root=dream_root)


def summarize_split(index: dict | None = None) -> str:
    idx = index if index is not None else load_split_index()
    even = idx["folds"]["even_train"]
    lines = [
        f"{idx['name']}: {idx['n_nights']} nights "
        f"{idx['first']} → {idx['last']}",
        f"  geometry: NSIDE={idx['nside']} {idx['scheme']} NPIX={idx['npix']}",
        f"  product:  {idx['product_default']}",
        f"  fold even_train: train={len(even['train'])}  "
        f"validation={len(even.get('validation', []))}  "
        f"test={len(even['test'])}",
        f"  rule: {idx['rule']}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Build or inspect the DREAM even/odd night split."
    )
    p.add_argument(
        "--write-json",
        action="store_true",
        help="Refresh analysis/eval_harness/dream_even_odd_split.json.",
    )
    p.add_argument(
        "--symlinks",
        action="store_true",
        help="Write data/dream/splits/even_odd/{train,test}/<date> links.",
    )
    p.add_argument(
        "--fold",
        default="even_train",
        choices=list(FOLDS),
        help="Which side of the even/odd cut is train (default: even_train).",
    )
    args = p.parse_args(argv)

    payload = build_split()
    print(summarize_split(payload))
    if args.write_json:
        path = write_split_json(payload)
        print(f"  wrote {path.relative_to(ROOT)}")
    else:
        # Keep the on-disk JSON in sync when the archive matches the freeze.
        if not SPLIT_JSON.is_file():
            write_split_json(payload)
            print(f"  wrote {SPLIT_JSON.relative_to(ROOT)}")
    if args.symlinks:
        dest = write_symlinks(args.fold)
        print(f"  symlinks {dest.relative_to(ROOT)}  fold={args.fold}")


if __name__ == "__main__":
    main()
