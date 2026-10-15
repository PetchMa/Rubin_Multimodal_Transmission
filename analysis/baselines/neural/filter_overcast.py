"""Drop windows that are socked-in or moonlit.

Writes keep-index arrays for neural-baseline train/val/test. Does not rewrite
memmaps. By default drops windows with moon illumination ≥ 0.25 and moon
altitude > 0°. Pass ``--keep-moon`` to include those frames. Optional
``--filter-patchy`` also requires extinction variance ``> 0.05 mag²``.

    python -m analysis.baselines.neural.filter_overcast --splits train validation
    python -m analysis.baselines.neural.filter_overcast --splits test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analysis.our_model.mask_train import CLOUD_T_THRESH
from analysis.paths import MNT
from analysis.satellite.sites import (
    MOON_ALT_MIN_DEG,
    MOON_ILLUM_MAX,
    moon_altitude_illumination,
    moon_is_bright,
)

DEFAULT_CACHE = MNT / "pairs" / "dream_20min_encoded"
DEFAULT_OUT = MNT / "baselines_neural" / "keep_cloudy_lt_99"
CHUNK = 512
PATCH_VAR_MIN = 0.05
_T_FLOOR = 1e-4


def load_keep_indices(path: Path, n: int, split: str) -> np.ndarray:
    """Load keep-index ``.npy`` and check it is in ``[0, n)``."""
    idx = np.asarray(np.load(path), dtype=np.int64)
    if idx.size == 0:
        raise ValueError(f"{split} keep-indices {path} is empty")
    if int(idx.min()) < 0 or int(idx.max()) >= n:
        raise ValueError(f"{split} keep-indices {path} out of range for n={n}")
    return idx


def window_cloudy_frac(
    y: np.ndarray,
    y_mask: np.ndarray,
    *,
    mean: float,
    std: float,
    thresh: float = CLOUD_T_THRESH,
) -> np.ndarray:
    """Per-window cloudy / valid. NaN when the window has no valid tiles."""
    valid = y_mask > 0
    t = y * std + mean
    cloudy = (t < thresh) & valid
    n_valid = valid.reshape(valid.shape[0], -1).sum(axis=1).astype(np.float64)
    n_cloud = cloudy.reshape(cloudy.shape[0], -1).sum(axis=1).astype(np.float64)
    frac = np.full(y.shape[0], np.nan, dtype=np.float64)
    ok = n_valid > 0
    frac[ok] = n_cloud[ok] / n_valid[ok]
    return frac


def window_patchiness(
    y: np.ndarray,
    y_mask: np.ndarray,
    *,
    mean: float,
    std: float,
    thresh: float = CLOUD_T_THRESH,
    sigma: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-window cloudy fraction and extinction variance (mag²).

    ``sigma > 0`` applies the in-memory HEALPix-face Gaussian used for
    label generation so photon noise does not inflate the variance cut.
    """
    valid = np.asarray(y_mask) > 0
    t = np.asarray(y, dtype=np.float64) * float(std) + float(mean)
    if float(sigma) > 0.0:
        import torch

        from analysis.healpix_torch import gaussian_smooth_healpix

        t = (
            gaussian_smooth_healpix(
                torch.as_tensor(t, dtype=torch.float32),
                torch.as_tensor(valid),
                sigma=float(sigma),
            )
            .numpy()
            .astype(np.float64)
        )
    n_valid = valid.reshape(valid.shape[0], -1).sum(axis=1).astype(np.float64)
    n_cloud = ((t < float(thresh)) & valid).reshape(valid.shape[0], -1).sum(axis=1)
    frac = np.full(y.shape[0], np.nan, dtype=np.float64)
    ok = n_valid > 0
    frac[ok] = n_cloud[ok] / n_valid[ok]
    ext = -2.5 * np.log10(np.clip(t, _T_FLOOR, None))
    ext = np.where(valid, ext, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        var = np.nanvar(ext.reshape(ext.shape[0], -1), axis=1)
    return frac, var


def window_patchy_ok(
    frac: np.ndarray,
    var: np.ndarray,
    *,
    min_var: float = PATCH_VAR_MIN,
) -> np.ndarray:
    """True when the window has valid tiles and structured extinction."""
    return np.isfinite(frac) & np.isfinite(var) & (var > float(min_var))


def load_or_compute_patchiness(
    cache: Path,
    n: int,
    *,
    mean: float,
    std: float,
    thresh: float,
    sigma: float,
    which: str = "last",
) -> tuple[np.ndarray, np.ndarray]:
    """Cloudy fraction + extinction variance for every window in ``cache``.

    Writes ``patchiness_{which}.npz`` next to the memmap. Camera arrays
    are not rewritten.
    """
    dest = Path(cache)
    sidecar = dest / f"patchiness_{which}.npz"
    if sidecar.is_file():
        z = np.load(sidecar)
        if (
            int(z["n"]) == int(n)
            and abs(float(z["thresh"]) - float(thresh)) < 1e-12
            and abs(float(z["sigma"]) - float(sigma)) < 1e-12
            and abs(float(z["mean"]) - float(mean)) < 1e-6
            and abs(float(z["std"]) - float(std)) < 1e-6
        ):
            return (
                np.asarray(z["cloudy_frac"], dtype=np.float64),
                np.asarray(z["var_mag2"], dtype=np.float64),
            )
    meta = json.loads((dest / "meta.json").read_text())
    shape = tuple(meta["shapes"][which])
    field = np.memmap(dest / f"{which}.npy", dtype=np.float32, mode="r", shape=shape)
    mask = np.memmap(
        dest / f"{which}_mask.npy", dtype=np.float32, mode="r", shape=tuple(meta["shapes"][f"{which}_mask"])
    )
    if int(shape[0]) != int(n):
        raise ValueError(f"{dest}/{which}.npy n={shape[0]} != {n}")
    print(
        f"  computing patchiness  {which}  n={n}  σ={sigma:g}  → {sidecar}",
        flush=True,
    )
    frac = np.empty(n, dtype=np.float64)
    var = np.empty(n, dtype=np.float64)
    for i in range(0, n, CHUNK):
        sl = slice(i, min(i + CHUNK, n))
        frac[sl], var[sl] = window_patchiness(
            np.asarray(field[sl]),
            np.asarray(mask[sl]),
            mean=mean,
            std=std,
            thresh=thresh,
            sigma=sigma,
        )
        if (i + CHUNK) % 8192 < CHUNK or sl.stop == n:
            print(f"    patchiness {sl.stop}/{n}", flush=True)
    try:
        np.savez_compressed(
            sidecar,
            cloudy_frac=frac,
            var_mag2=var,
            n=np.int64(n),
            thresh=np.float64(thresh),
            sigma=np.float64(sigma),
            mean=np.float64(mean),
            std=np.float64(std),
        )
    except OSError as exc:
        print(f"  patchiness sidecar not written ({exc})", flush=True)
    return frac, var


def load_or_compute_moon(cache: Path) -> tuple[np.ndarray, np.ndarray]:
    """Last-frame moon alt / illumination for a cache split.

    Writes ``moon.npz`` next to ``times.npz`` on first call. The camera
    memmaps are not touched.
    """
    dest = Path(cache)
    times = np.load(dest / "times.npz")
    unix = np.asarray(times["unix"], dtype=np.float64)
    unix_end = unix[:, -1] if unix.ndim == 2 else unix
    sidecar = dest / "moon.npz"
    if sidecar.is_file():
        z = np.load(sidecar)
        if (
            "unix" in z
            and z["unix"].shape == unix_end.shape
            and np.array_equal(z["unix"], unix_end)
        ):
            return (
                np.asarray(z["alt_deg"], dtype=np.float64),
                np.asarray(z["illumination"], dtype=np.float64),
            )
    print(f"  computing moon ephemeris  n={unix_end.size}  → {sidecar}", flush=True)
    alt, illum = moon_altitude_illumination(unix_end)
    try:
        np.savez_compressed(
            sidecar,
            unix=unix_end,
            alt_deg=alt,
            illumination=illum,
        )
    except OSError as exc:
        print(f"  moon sidecar not written ({exc})", flush=True)
    return alt, illum


def window_moon_ok(
    cache: Path,
    n: int,
    *,
    max_illum: float = MOON_ILLUM_MAX,
    min_alt_deg: float = MOON_ALT_MIN_DEG,
) -> np.ndarray:
    """True for windows that are dark enough (not a bright moon above the horizon)."""
    alt, illum = load_or_compute_moon(cache)
    if alt.shape[0] != int(n):
        raise ValueError(f"moon sidecar n={alt.shape[0]} != cache n={n}")
    return ~moon_is_bright(
        alt, illum, max_illum=max_illum, min_alt_deg=min_alt_deg
    )


def persist_iou_on_keep(
    y: np.ndarray,
    y_mask: np.ndarray,
    last: np.ndarray,
    last_mask: np.ndarray,
    keep: np.ndarray,
    *,
    mean: float,
    std: float,
    thresh: float = CLOUD_T_THRESH,
) -> dict[str, float]:
    yy, ym = y[keep], y_mask[keep] > 0
    t_y = yy * std + mean
    t_l = last[keep] * std + mean
    conf = ym
    gold = (t_y < thresh) & conf
    pers = (t_l < thresh) & (last_mask[keep] > 0) & conf
    tp = int((pers & gold).sum())
    fp = int((pers & ~gold & conf).sum())
    fn = int((~pers & gold).sum())
    n = int(conf.sum())
    union = tp + fp + fn
    cloudy = tp + fn
    return {
        "n_windows": int(keep.size),
        "n_tiles": n,
        "cloudy_frac": (cloudy / n) if n else float("nan"),
        "persist_iou": (tp / union) if union else float("nan"),
    }


def filter_split(
    cache: Path,
    split: str,
    out_dir: Path,
    *,
    max_frac: float,
    thresh: float,
    filter_moon: bool = True,
    filter_patchy: bool = False,
    label_sigma: float = 0.0,
) -> dict:
    dest = Path(cache) / split
    meta_path = dest / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"no finished cache for {split}: {meta_path}")
    meta = json.loads(meta_path.read_text())
    mean = float(meta["norm"]["sky"]["mean"])
    std = float(meta["norm"]["sky"]["std"])
    n = int(meta["n"])
    y = np.memmap(dest / "y.npy", dtype=np.float32, mode="r", shape=tuple(meta["shapes"]["y"]))
    ym = np.memmap(
        dest / "y_mask.npy", dtype=np.float32, mode="r", shape=tuple(meta["shapes"]["y_mask"])
    )
    n_patchy = 0
    n_overcast = 0
    if filter_patchy:
        frac, var = load_or_compute_patchiness(
            dest,
            n,
            mean=mean,
            std=std,
            thresh=thresh,
            sigma=label_sigma,
            which="y",
        )
        patchy = window_patchy_ok(frac, var)
        drop = ~patchy
        n_patchy = int(drop.sum())
        empty = int((~np.isfinite(frac)).sum())
    else:
        fracs = np.empty(n, dtype=np.float64)
        for i in range(0, n, CHUNK):
            sl = slice(i, min(i + CHUNK, n))
            fracs[sl] = window_cloudy_frac(
                np.asarray(y[sl]), np.asarray(ym[sl]), mean=mean, std=std, thresh=thresh
            )
        drop = ~np.isfinite(fracs) | (fracs > max_frac)
        n_overcast = int(drop.sum())
        empty = int(np.isnan(fracs).sum())
    n_moon = 0
    if filter_moon:
        if not (dest / "times.npz").is_file():
            print(f"  {split}: no times.npz, skipping moon filter", flush=True)
            filter_moon = False
        else:
            moon_ok = window_moon_ok(dest, n)
            n_moon = int((~moon_ok & ~drop).sum())
            drop = drop | ~moon_ok
    keep = np.flatnonzero(~drop).astype(np.int64)
    out_dir.mkdir(parents=True, exist_ok=True)
    idx_path = out_dir / f"{split}.npy"
    np.save(idx_path, keep)

    info = {
        "split": split,
        "cache": str(dest),
        "n_full": n,
        "n_keep": int(keep.size),
        "n_drop": int(drop.sum()),
        "n_drop_overcast": n_overcast,
        "n_drop_patchy": n_patchy,
        "n_drop_moon": n_moon,
        "max_cloudy_frac": max_frac if not filter_patchy else None,
        "patch_var_min": PATCH_VAR_MIN if filter_patchy else None,
        "filter_patchy": bool(filter_patchy),
        "cloud_t_thresh": thresh,
        "filter_moon": bool(filter_moon),
        "moon_illum_max": MOON_ILLUM_MAX if filter_moon else None,
        "moon_alt_min_deg": MOON_ALT_MIN_DEG if filter_moon else None,
        "empty_windows": empty,
        "indices": str(idx_path),
    }
    if n <= 20_000:
        last = np.memmap(
            dest / "last.npy", dtype=np.float32, mode="r", shape=tuple(meta["shapes"]["last"])
        )
        lm = np.memmap(
            dest / "last_mask.npy",
            dtype=np.float32,
            mode="r",
            shape=tuple(meta["shapes"]["last_mask"]),
        )
        info["persist_full"] = persist_iou_on_keep(
            y, ym, last, lm, np.arange(n), mean=mean, std=std, thresh=thresh
        )
        info["persist_keep"] = persist_iou_on_keep(
            y, ym, last, lm, keep, mean=mean, std=std, thresh=thresh
        )
    (out_dir / f"{split}.json").write_text(json.dumps(info, indent=2) + "\n")
    extra = ""
    if "persist_keep" in info:
        extra = (
            f"  persist {info['persist_full']['persist_iou']:.4f} → "
            f"{info['persist_keep']['persist_iou']:.4f}"
        )
    moon_note = f"  moon {n_moon}" if filter_moon else ""
    scene_note = f"  patchy {n_patchy}" if filter_patchy else f"  overcast {n_overcast}"
    print(
        f"  {split}: keep {info['n_keep']}/{n}  dropped {info['n_drop']}  "
        f"{scene_note}{moon_note}  empty {info['empty_windows']}"
        f"{extra}  → {idx_path}",
        flush=True,
    )
    return info


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation"],
        choices=("train", "validation", "test"),
    )
    p.add_argument("--max-cloudy-frac", type=float, default=0.99)
    p.add_argument("--cloud-t", type=float, default=CLOUD_T_THRESH)
    p.add_argument(
        "--keep-moon",
        action="store_true",
        help="Keep bright-moon windows (illum≥0.25 and alt>0°). Default drops them.",
    )
    p.add_argument(
        "--filter-patchy",
        action="store_true",
        help=f"Also drop windows with extinction variance ≤ {PATCH_VAR_MIN} mag².",
    )
    p.add_argument(
        "--label-smooth-sigma",
        type=float,
        default=0.0,
        help="In-memory Gaussian σ (face pixels) before the patchiness cuts.",
    )
    args = p.parse_args(argv)
    print(
        f"filter windows  T < {args.cloud_t}  "
        + (
            f"patchy var>{PATCH_VAR_MIN} mag²  "
            if args.filter_patchy
            else f"overcast cloudy_frac > {args.max_cloudy_frac}  "
        )
        + f"moon={'keep' if args.keep_moon else f'drop illum≥{MOON_ILLUM_MAX} alt>{MOON_ALT_MIN_DEG:g}°'}  "
        f"cache {args.cache_dir}",
        flush=True,
    )
    for split in args.splits:
        filter_split(
            args.cache_dir,
            split,
            args.out_dir,
            max_frac=args.max_cloudy_frac,
            thresh=args.cloud_t,
            filter_moon=not args.keep_moon,
            filter_patchy=args.filter_patchy,
            label_sigma=args.label_smooth_sigma,
        )


if __name__ == "__main__":
    main()
