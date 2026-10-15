"""Carve a night-wise validation split from the even-night train cache.

Evenly spaced train nights (about 20%) become validation. The remaining
even nights stay in train. Windows that are more than 99% cloudy are
dropped first. Odd nights are the holdout and are not used here.

    python -m analysis.forecaster.carve_train_val
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analysis.baselines.neural.filter_overcast import load_keep_indices
from analysis.eval_harness.dream_data import pick_evenly
from analysis.paths import MNT

CACHE_TRAIN = MNT / "pairs" / "dream_20min_encoded" / "train"
OVERCAST_KEEP = MNT / "baselines_neural" / "keep_cloudy_lt_99"
DEFAULT_OUT = MNT / "baselines_neural" / "keep_train_nights_val20"
EXHAUSTED_VAL = MNT / "baselines_neural" / "keep_cloudy_lt_99"
VAL_FRAC = 0.20


def _dates_for_cache(cache: Path, n: int) -> np.ndarray:
    times = np.load(cache / "times.npz")
    dates = np.asarray(times["dates"])
    if dates.shape[0] != n:
        raise SystemExit(f"{cache}: times n={dates.shape[0]} != cache n={n}")
    return np.array([str(d) for d in dates], dtype=object)


def carve(
    cache: Path = CACHE_TRAIN,
    overcast_keep: Path = OVERCAST_KEEP,
    out_dir: Path = DEFAULT_OUT,
    *,
    val_frac: float = VAL_FRAC,
) -> dict:
    meta = json.loads((Path(cache) / "meta.json").read_text())
    n = int(meta["n"])
    keep = load_keep_indices(Path(overcast_keep) / "train.npy", n, "train")
    dates = _dates_for_cache(Path(cache), n)
    nights = sorted(set(str(d) for d in dates[keep]))
    n_val_nights = max(1, int(round(len(nights) * float(val_frac))))
    val_nights = [str(d) for d in pick_evenly(nights, n_val_nights)]
    val_set = set(val_nights)
    train_nights = [d for d in nights if d not in val_set]
    night_of = dates[keep]
    is_val = np.array([str(d) in val_set for d in night_of], dtype=bool)
    train_idx = keep[~is_val].astype(np.int64)
    val_idx = keep[is_val].astype(np.int64)
    if train_idx.size == 0 or val_idx.size == 0:
        raise SystemExit("inner-val carve produced an empty split")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "train.npy", train_idx)
    np.save(out_dir / "validation.npy", val_idx)

    def _night_counts(idx: np.ndarray) -> list[dict]:
        by: dict[str, int] = {}
        for d in dates[idx]:
            by[str(d)] = by.get(str(d), 0) + 1
        return [{"date": d, "n_keep_windows": by[d]} for d in sorted(by)]

    info = {
        "purpose": (
            "Night-wise val carved from even-night train. "
            "Do not use keep_cloudy_lt_99/validation for new training."
        ),
        "val_from_train_nights": True,
        "cache": str(Path(cache)),
        "overcast_keep": str(Path(overcast_keep)),
        "exhausted_val": str(Path(EXHAUSTED_VAL) / "validation.npy"),
        "exhausted_val_note": (
            "Odd-night index. Held out of this train/validation split."
        ),
        "val_frac": float(val_frac),
        "n_cache": n,
        "n_overcast_keep": int(keep.size),
        "n_train_nights": len(train_nights),
        "n_val_nights": len(val_nights),
        "n_train_windows": int(train_idx.size),
        "n_val_windows": int(val_idx.size),
        "train_nights": train_nights,
        "val_nights": val_nights,
        "train_nights_detail": _night_counts(train_idx),
        "val_nights_detail": _night_counts(val_idx),
        "indices_index": "train memmap row",
        "rule": (
            f"Sort packed train nights that survive cloudy_frac<=0.99. "
            f"Keep {val_frac:.0%} evenly spaced via pick_evenly as val; "
            f"remainder train. Windows stay with their night."
        ),
    }
    (out_dir / "split.json").write_text(json.dumps(info, indent=2) + "\n")
    # Record where the holdout index lives when writing the default keep dir.
    if Path(out_dir).resolve() == Path(DEFAULT_OUT).resolve():
        exhausted = {
            "status": "holdout",
            "split": "validation",
            "nights": 15,
            "keep_dir": str(Path(EXHAUSTED_VAL)),
            "replacement": str(out_dir),
            "note": (
                "Odd nights are the holdout. This keep dir is train and validation only."
            ),
        }
        (Path(EXHAUSTED_VAL) / "EXHAUSTED.json").write_text(
            json.dumps(exhausted, indent=2) + "\n"
        )
    print(
        f"  inner val  nights {len(val_nights)}/{len(nights)}  "
        f"windows train {train_idx.size}  val {val_idx.size}  → {out_dir}",
        flush=True,
    )
    print(f"  val nights: {', '.join(val_nights)}", flush=True)
    return info


def subset_train_nights(
    src: Path = DEFAULT_OUT,
    out_dir: Path | None = None,
    *,
    frac: float,
) -> dict:
    """Keep ``frac`` of parent train nights via ``pick_evenly``. Val unchanged.

    This drops whole nights. For a 10% *window* draw that keeps every
    training night, use ``--train-frac 0.10`` (``model.train_frac_indices``).
    """
    src = Path(src)
    parent = json.loads((src / "split.json").read_text())
    train_nights = [str(d) for d in parent["train_nights"]]
    val_nights = [str(d) for d in parent["val_nights"]]
    frac = float(frac)
    if frac <= 0.0 or frac > 1.0:
        raise SystemExit(f"train_night_frac must be in (0, 1], got {frac}")
    n_keep = max(1, int(round(len(train_nights) * frac)))
    kept_nights = [str(d) for d in pick_evenly(train_nights, n_keep)]
    kept_set = set(kept_nights)
    cache = Path(parent["cache"])
    dates = _dates_for_cache(cache, int(parent["n_cache"]))
    train_full = np.load(src / "train.npy").astype(np.int64)
    val_idx = np.load(src / "validation.npy").astype(np.int64)
    night_of = dates[train_full]
    mask = np.array([str(d) in kept_set for d in night_of], dtype=bool)
    train_idx = train_full[mask]
    if train_idx.size == 0:
        raise SystemExit("train-night subset produced an empty train split")

    if out_dir is None:
        tag = f"{frac:.2f}".replace("0.", "")
        out_dir = src.parent / f"{src.name}_frac{tag}"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "train.npy", train_idx)
    np.save(out_dir / "validation.npy", val_idx)

    def _night_counts(idx: np.ndarray) -> list[dict]:
        by: dict[str, int] = {}
        for d in dates[idx]:
            by[str(d)] = by.get(str(d), 0) + 1
        return [{"date": d, "n_keep_windows": by[d]} for d in sorted(by)]

    info = {
        "purpose": (
            f"Night-wise train subset ({frac:.0%} of parent train nights) "
            "for a data learning curve. Same 28-night inner val."
        ),
        "val_from_train_nights": True,
        "parent_keep": str(src),
        "train_night_frac": frac,
        "cache": parent["cache"],
        "n_cache": parent["n_cache"],
        "n_train_nights": len(kept_nights),
        "n_train_nights_parent": len(train_nights),
        "n_val_nights": len(val_nights),
        "n_train_windows": int(train_idx.size),
        "n_val_windows": int(val_idx.size),
        "train_nights": kept_nights,
        "held_out_train_nights": [d for d in train_nights if d not in kept_set],
        "val_nights": val_nights,
        "train_nights_detail": _night_counts(train_idx),
        "val_nights_detail": _night_counts(val_idx),
        "indices_index": "train memmap row",
        "rule": (
            f"pick_evenly {n_keep}/{len(train_nights)} parent train nights "
            f"({frac:.0%}). Validation nights and windows copied unchanged."
        ),
    }
    (out_dir / "split.json").write_text(json.dumps(info, indent=2) + "\n")
    print(
        f"  train-night frac {frac:.0%}  nights {len(kept_nights)}/"
        f"{len(train_nights)}  windows {train_idx.size}  val {val_idx.size}  "
        f"→ {out_dir}",
        flush=True,
    )
    return info


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path, default=CACHE_TRAIN)
    p.add_argument("--overcast-keep", type=Path, default=OVERCAST_KEEP)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--val-frac", type=float, default=VAL_FRAC)
    p.add_argument(
        "--src-keep",
        type=Path,
        default=DEFAULT_OUT,
        help="Parent keep dir when using --train-night-frac.",
    )
    p.add_argument(
        "--train-night-frac",
        type=float,
        default=None,
        help="Subset parent train nights (val unchanged). For a data curve.",
    )
    args = p.parse_args(argv)
    if args.train_night_frac is not None:
        info = subset_train_nights(
            args.src_keep,
            args.out_dir,
            frac=args.train_night_frac,
        )
        keys = (
            "train_night_frac",
            "n_train_nights",
            "n_train_nights_parent",
            "n_val_nights",
            "n_train_windows",
            "n_val_windows",
        )
    else:
        info = carve(
            args.cache_dir,
            args.overcast_keep,
            args.out_dir or DEFAULT_OUT,
            val_frac=args.val_frac,
        )
        keys = (
            "n_train_nights",
            "n_val_nights",
            "n_train_windows",
            "n_val_windows",
            "val_nights",
        )
    print(json.dumps({k: info[k] for k in keys}, indent=2))


if __name__ == "__main__":
    main()
