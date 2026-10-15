"""Shared DREAM/GOES cache loading for the multimodal forecaster.

Training itself is ``train_multihorizon.py`` (every cached lead, two-stage
curriculum).

    python -u -m analysis.forecaster.train_multihorizon
    python -u -m analysis.forecaster.run_curriculum --out ...
    python -u -m analysis.forecaster.train --report-only
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Subset

from analysis.baselines.neural.filter_overcast import load_keep_indices
from analysis.forecaster.carve_train_val import (
    DEFAULT_OUT as DEFAULT_INNER_KEEP_DIR,
    carve as carve_inner_val,
)
from analysis.forecaster.geom import attach_wind_decode_stats
from analysis.forecaster.port_prep import attach_horizon_lock
from analysis.forecaster.sat_ingest import (
    CHANNEL_NAMES,
    report_ingest,
    wrap_rich_sat,
)
from analysis.our_model.model import (
    BundleNorm,
    PairsDataset,
    TrainConfig,
    _cache_matches,
    _cache_signature,
    _norm_from_dict,
    _worker_init,
    default_cache_dir,
    encode_split_cache,
    list_bundles,
    resolve_device,
    resolve_pairs,
)
from analysis.our_model.sat_pretrain.dataset import NATIVE_IR_HW
from analysis.paths import MNT

HERE = Path(__file__).resolve().parent
SOURCE_PATH = HERE / "SOURCE.json"
DEFAULT_HORIZON_SEC = 1200
DEFAULT_OUT = MNT / "baselines_neural" / "paper_multih_t24"
DEFAULT_CACHE = MNT / "pairs" / "dream_20min_encoded_t24"
BAND_DELTA = 0.0
DEFAULT_N_FRAMES = 24
OBJECTIVE = "val_yield_mse"


def _enable_fast_cuda() -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def _apply_keep(ds, keep_dir: Path | None, split: str):
    """Restrict to keep-index rows. Inner-val indices are into the train memmap."""
    if keep_dir is None:
        return ds
    path = Path(keep_dir) / f"{split}.npy"
    if not path.is_file():
        raise SystemExit(f"missing keep-indices {path} (use --no-keep to disable)")
    n = int(getattr(ds, "n", len(ds)))
    idx = load_keep_indices(path, n, split)
    print(
        f"  {split} keep-set {path}  {idx.size}/{n}  dropped {n - int(idx.size)}",
        flush=True,
    )
    return Subset(ds, idx.tolist())


def _set_sky_norm(model: nn.Module, sky_norm) -> None:
    if not hasattr(model, "sky_mean"):
        return
    with torch.no_grad():
        model.sky_mean.fill_(float(sky_norm.mean))
        model.sky_std.fill_(float(sky_norm.std))


def _json_num(x):
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)


def _snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        k.replace("._orig_mod.", "."): v.detach().clone()
        for k, v in model.state_dict().items()
    }


def _load_split_cache(
    files: list[Path],
    dest: Path,
    split: str,
    cfg: TrainConfig,
    norm,
) -> Path:
    """Reuse memmaps if they exist. Never recode on a pair-list drift."""
    dest = Path(dest)
    meta_path = dest / "meta.json"
    sky = dest / "sky.npy"
    if meta_path.is_file() and sky.is_file():
        old = json.loads(meta_path.read_text())
        for key, expected in (
            ("n_frames", cfg.n_frames),
            ("sat_size", cfg.sat_size),
            ("min_nobs", cfg.min_nobs),
        ):
            if old.get(key) != expected:
                raise SystemExit(
                    f"{split} cache {dest} has {key}={old.get(key)!r}, "
                    f"but this run requests {expected!r}. Use a separate "
                    "--cache-dir for this run."
                )
        sig = _cache_signature(
            files,
            n_frames=cfg.n_frames,
            sat_size=cfg.sat_size,
            min_nobs=cfg.min_nobs,
        )
        if _cache_matches(dest, sig) is None:
            print(
                f"  {split}: pair list drifted; reusing cache {dest}  "
                f"n={old.get('n')}  (pairs n={sig['n']})",
                flush=True,
            )
        else:
            print(f"  {split}: reusing cache {dest}  n={old.get('n')}", flush=True)
        return dest
    return encode_split_cache(
        files,
        dest,
        split=split,
        n_frames=cfg.n_frames,
        sat_size=cfg.sat_size,
        min_nobs=cfg.min_nobs,
        norm=norm,
        verbose=cfg.verbose,
    )


def load_data(
    cfg: TrainConfig,
    pairs_dir: Path,
    cache_dir: Path | None,
    *,
    score_test: bool = False,
    val_from_train: bool = False,
):
    cache_root = Path(cache_dir) if cache_dir is not None else default_cache_dir(pairs_dir)
    train_files = list_bundles(pairs_dir, "train")
    val_files = list_bundles(pairs_dir, "validation")
    test_files = list_bundles(pairs_dir, "test") if score_test else []
    if not train_files:
        raise SystemExit(f"missing train bundles in {pairs_dir}")
    if not val_from_train and not val_files:
        raise SystemExit(f"missing train or validation bundles in {pairs_dir}")
    train_meta = cache_root / "train" / "meta.json"
    if train_meta.is_file():
        cached = json.loads(train_meta.read_text())
        print("  reusing train-cache normalization", flush=True)
        norm = _norm_from_dict(cached["norm"])
    else:
        norm = BundleNorm.fit_files(
            train_files,
            n_frames=cfg.n_frames,
            sat_size=cfg.sat_size,
            min_nobs=cfg.min_nobs,
            verbose=cfg.verbose,
        )
    train_cache = _load_split_cache(
        train_files, cache_root / "train", "train", cfg, norm
    )
    val_cache = train_cache
    if not val_from_train:
        val_cache = _load_split_cache(
            val_files, cache_root / "validation", "validation", cfg, norm
        )
    test_cache = None
    if test_files:
        test_cache = _load_split_cache(
            test_files, cache_root / "test", "test", cfg, norm
        )
    device = resolve_device(cfg.device)
    data = PairsDataset.from_memmap(train_cache, device)
    val_data = (
        data if val_from_train else PairsDataset.from_memmap(val_cache, device)
    )
    test_data = PairsDataset.from_memmap(test_cache, device) if test_cache else None
    nside = json.loads((train_cache / "meta.json").read_text())["nside"]
    return data, val_data, test_data, norm, nside, device, cache_root


def attach_rich_sat(
    data,
    val_data,
    test_data,
    pairs_dir: Path,
    cache_root: Path,
    cfg: TrainConfig,
    *,
    val_from_train: bool = False,
):
    """Replace cached 2-ch C13/ACHA with the 18-ch Pachón night cube."""
    print(
        f"  rich sat ingest  {len(CHANNEL_NAMES)} ch  "
        + " ".join(CHANNEL_NAMES),
        flush=True,
    )
    sat_size = NATIVE_IR_HW
    data = wrap_rich_sat(
        data, pairs_dir, "train", cache_root / "train",
        n_frames=cfg.n_frames, sat_size=sat_size,
    )
    if val_from_train:
        val_data = data
    else:
        val_data = wrap_rich_sat(
            val_data, pairs_dir, "validation", cache_root / "validation",
            n_frames=cfg.n_frames, sat_size=sat_size,
        )
    if test_data is not None:
        test_data = wrap_rich_sat(
            test_data, pairs_dir, "test", cache_root / "test",
            n_frames=cfg.n_frames, sat_size=sat_size,
        )
    return data, val_data, test_data


def attach_port_physics(
    data,
    val_data,
    test_data,
    cache_root: Path,
    *,
    val_from_train: bool = True,
):
    """Horizon-lock camera maps. Times sidecar comes from ``wrap_rich_sat``."""
    data = attach_horizon_lock(data, cache_root / "train")
    if val_from_train:
        val_data = data
    else:
        val_data = attach_horizon_lock(val_data, cache_root / "validation")
    if test_data is not None:
        test_data = attach_horizon_lock(test_data, cache_root / "test")
    return data, val_data, test_data


def _fork_loader(dataset, cfg: TrainConfig, device: torch.device, sampler) -> DataLoader:
    """Same worker/fork setup as ``make_loader``, with a caller-supplied sampler."""
    kwargs: dict = {
        "batch_size": cfg.batch_size,
        "shuffle": False,
        "num_workers": int(cfg.num_workers),
        "pin_memory": device.type == "cuda",
        "drop_last": False,
        "sampler": sampler,
    }
    if cfg.num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = int(cfg.prefetch_factor)
        kwargs["worker_init_fn"] = _worker_init
        kwargs["multiprocessing_context"] = mp.get_context("fork")
    return DataLoader(dataset, **kwargs)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--report-only",
        action="store_true",
        help="Build the rich-sat sidecar, print ingest coverage, and exit.",
    )
    p.add_argument("--pairs", type=Path, default=None)
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--n-frames", type=int, default=DEFAULT_N_FRAMES)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)
    if not args.report_only:
        raise SystemExit(
            "The proposed model is multihorizon. Train with\n"
            "  python -m analysis.forecaster.train_multihorizon\n"
            "  python -m analysis.forecaster.run_curriculum --out DIR\n"
            "This module only loads caches. Pass --report-only for sat ingest coverage."
        )
    out = Path(args.out) if args.out is not None else DEFAULT_OUT
    out.mkdir(parents=True, exist_ok=True)
    cfg = TrainConfig(
        n_frames=args.n_frames,
        seed=args.seed,
        device=args.device,
        verbose=True,
        d_model=64,
        n_heads=4,
        n_layers=2,
        sky_width=16,
    )
    pairs_dir = (
        resolve_pairs(args.pairs)
        if args.pairs is not None
        else MNT / "data" / "pairs" / "dream_20min"
    )
    cache_dir = Path(args.cache_dir) if args.cache_dir is not None else DEFAULT_CACHE
    data, val_data, test_data, norm, nside, device, cache_root = load_data(
        cfg,
        pairs_dir,
        cache_dir,
        score_test=False,
        val_from_train=True,
    )
    cfg.wind_age_mean = float(norm.wind[-1].mean)
    cfg.wind_age_std = float(norm.wind[-1].std)
    attach_wind_decode_stats(cfg, norm)
    data, val_data, test_data = attach_rich_sat(
        data,
        val_data,
        test_data,
        pairs_dir,
        cache_root,
        cfg,
        val_from_train=True,
    )
    keep_dir = DEFAULT_INNER_KEEP_DIR
    if not (keep_dir / "validation.npy").is_file():
        print(f"  carving inner val → {keep_dir}", flush=True)
        carve_inner_val(out_dir=keep_dir)
    payload = report_ingest(
        {
            "train": data,
            "validation": val_data,
            **({"test": test_data} if test_data is not None else {}),
        },
        n_samples=8,
        seed=cfg.seed,
    )
    report_path = out / "sat_ingest_report.json"
    _write_json(report_path, payload)
    print(f"wrote ingest report → {report_path}", flush=True)
    print(
        json.dumps(
            {k: payload[k] for k in ("channels", "splits", "alignment", "normalization")},
            indent=2,
        ),
        flush=True,
    )
    _ = (nside, device)


if __name__ == "__main__":
    main()
