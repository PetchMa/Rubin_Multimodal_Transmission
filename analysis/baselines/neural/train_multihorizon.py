"""Train camera-only SimVP / ConvLSTM on every cached lead.

Same cache, keep split, encode-once / decode-per-lead, and mean airmass
yield MSE as the multimodal forecaster. No GOES or winds. History is encoded
once; each lead (90 s, 5 min, 20 min, 30 min) gets a centered lead embedding
that is identity at +20 min. Plateau on mean val yield MSE.

    python -u -m analysis.baselines.neural.train_multihorizon --arch simvp
    python -u -m analysis.baselines.neural.train_multihorizon --arch convlstm \\
        --init-ckpt .../paper_convlstm_t24/seed_42/convlstm/best.pt
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from analysis.baselines.neural.lead import CenteredLead
from analysis.baselines.neural.wrapper import (
    build_nowcast,
    load_nowcast_backbone,
    load_nowcast_matching,
    slim_camera_pairs,
)
from analysis.forecaster.carve_train_val import (
    DEFAULT_OUT as DEFAULT_INNER_KEEP_DIR,
    carve as carve_inner_val,
)
from analysis.forecaster.curriculum import (
    add_curriculum_args,
    apply_curriculum_stage,
    format_line as format_curriculum_line,
    make_record as make_curriculum_record,
    write_record as write_curriculum_record,
)
from analysis.forecaster.geom import LEAD_SEC
from analysis.forecaster.multihorizon import HORIZONS_SEC, wrap_multihorizon
from analysis.forecaster.port_prep import LABEL_SMOOTH_SIGMA, attach_horizon_lock
from analysis.forecaster.train import SOURCE_PATH, _write_json, load_data
from analysis.forecaster.train_multihorizon import (
    BAND_DELTA,
    DEFAULT_CACHE,
    DEFAULT_PAIRS,
    FALSE_CLEAR_T,
    LOSS_NAME,
    OBJECTIVE,
    train_multihorizon,
)
from analysis.our_model.model import TrainConfig, resolve_pairs, seed_everything
from analysis.paths import MNT
from analysis.torch_util import resolve_device

DEFAULT_OUT = MNT / "baselines_neural"


def _assert_lead_identity(model, device) -> None:
    nside = int(model.nside)
    n_frames = int(model.n_frames)
    npix = 12 * nside * nside
    x = torch.rand(2, n_frames, 2, npix, device=device)
    with torch.no_grad():
        ctx = model.net.encode(x)
        d20 = model.net.decode(ctx, LEAD_SEC)
        d_none = model.net.decode(ctx, None)
        if not torch.allclose(d20, d_none, atol=1e-6):
            raise RuntimeError("decode(None) must match decode(+20 min)")
        for h in HORIZONS_SEC:
            dh = model.net.decode(ctx, float(h))
            if not torch.allclose(dh, d20, atol=1e-6):
                raise RuntimeError(f"lead MLP not identity at init for {h}s")
        encoded = (x[:, :, 0] - 0.9) / 0.1
        mask = x[:, :, 1]
        hats = model.forward_horizons(
            encoded, mask, leads=[float(h) for h in HORIZONS_SEC]
        )
    if len(hats) != len(HORIZONS_SEC):
        raise RuntimeError(f"forward_horizons n={len(hats)}")
    for hat in hats:
        if tuple(hat.shape) != (2, npix):
            raise RuntimeError(f"hat shape {tuple(hat.shape)}")
        if not torch.allclose(hat, hats[0], atol=1e-6):
            raise RuntimeError("forward_horizons not identity across leads at init")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", choices=("convlstm", "simvp"), required=True)
    p.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument(
        "--init-ckpt",
        type=Path,
        default=None,
        help="Warm-start from a T=24 +20 min checkpoint. lead MLP stays identity.",
    )
    p.add_argument(
        "--init-full",
        action="store_true",
        help="With --init-ckpt, copy every matching tensor including the lead MLP.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Continue from <out>/last.pt (weights, epoch, plateau LR). "
            "Keeps best*.pt. Overrides --init-ckpt when last.pt exists."
        ),
    )
    p.add_argument(
        "--fc-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Add λ·mean(relu(T̂−thresh)) on gold-cloud tiles at every "
            "present lead (same horizon weights as yield MSE). "
            "0 (default) is off."
        ),
    )
    p.add_argument(
        "--fc-loss-thresh",
        type=float,
        default=FALSE_CLEAR_T,
        help="False-clear hinge threshold (default 0.63).",
    )
    p.add_argument(
        "--fc-yield-slack",
        type=float,
        default=1.03,
        help=(
            "Save best_20min_fc.pt when +20 min val yield ≤ this × the "
            "init checkpoint's +20 min val yield and η_FC is best among those."
        ),
    )
    add_curriculum_args(p)
    p.add_argument("--keep-dir", type=Path, default=DEFAULT_INNER_KEEP_DIR)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--n-frames", type=int, default=24)
    p.add_argument("--train-frac", type=float, default=1.0)
    p.add_argument(
        "--val-frac",
        type=float,
        default=1.0,
        help="Fraction of inner val scored each epoch (default 1.0 = full).",
    )
    p.add_argument(
        "--full-val-every",
        type=int,
        default=5,
        help="Score the full inner val every N epochs when --val-frac < 1.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=5e-2)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--max-lr-halves", type=int, default=4)
    p.add_argument("--device", default=None)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--amp", action="store_true")
    p.add_argument(
        "--h20-weight",
        type=float,
        default=1.0,
        help="Extra train weight on the +20 min lead (default 1 = equal).",
    )
    p.add_argument(
        "--schedule",
        choices=("plateau", "cosine"),
        default="plateau",
    )
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument(
        "--band-mix",
        type=float,
        default=0.0,
        help=(
            "Train sampling: this fraction of each epoch from the night-capped "
            "mean-T∈[band-t-min, band-t-max] pool; the rest from the usual "
            "all-night 40/night cap. 0 (default) is the paper multihorizon keep."
        ),
    )
    p.add_argument("--band-t-min", type=float, default=0.1)
    p.add_argument("--band-t-max", type=float, default=0.9)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the net, check lead identity at init, and exit.",
    )
    argv_list = list(sys.argv[1:] if argv is None else argv)
    args = p.parse_args(argv)
    args = apply_curriculum_stage(p, args, argv=argv_list, trainer="multihorizon")
    if args.band_mix < 0 or args.band_mix > 1:
        p.error("--band-mix must be in [0, 1]")
    if args.band_mix > 0 and not (args.band_t_min < args.band_t_max):
        p.error("--band-t-min must be < --band-t-max")
    if args.fc_loss_weight < 0:
        p.error("--fc-loss-weight must be ≥ 0")
    if args.init_full and args.init_ckpt is None and not args.resume:
        p.error("--init-full requires --init-ckpt")

    arch = str(args.arch)
    out = (
        Path(args.out)
        if args.out is not None
        else DEFAULT_OUT / f"paper_{arch}_multih_t24" / f"seed_{args.seed}" / arch
    )
    cache_dir = Path(args.cache_dir)
    keep_dir = Path(args.keep_dir)
    if keep_dir.name != "keep_train_nights_val20":
        raise SystemExit(f"keep_dir must be keep_train_nights_val20, got {keep_dir}")

    cfg = TrainConfig(
        epochs=1 if args.dry_run else args.epochs,
        batch_size=args.batch_size,
        num_workers=0 if args.dry_run else args.num_workers,
        n_frames=args.n_frames,
        train_frac=min(args.train_frac, 0.002) if args.dry_run else args.train_frac,
        seed=args.seed,
        lr=args.lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        max_lr_halves=args.max_lr_halves,
        device=args.device,
        verbose=True,
    )
    seed_everything(cfg.seed)
    model = build_nowcast(
        arch,
        nside=32,
        n_frames=int(cfg.n_frames),
        sky_mean=0.9,
        sky_std=0.1,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lead = model.net.lead
    if not isinstance(lead, CenteredLead):
        raise SystemExit(f"{arch} missing CenteredLead")
    print(
        f"neural multihorizon {arch}  n_frames={cfg.n_frames}  "
        f"params={n_params:,}  leads={list(HORIZONS_SEC)}  → {out}",
        flush=True,
    )
    if args.dry_run:
        device = resolve_device(args.device or "cpu")
        model = model.to(device)
        _assert_lead_identity(model, device)
        print(
            f"  dry-run ok  device={device}  n_frames={cfg.n_frames}  "
            f"lead identity at init  params={n_params:,}",
            flush=True,
        )
        return

    if not (keep_dir / "validation.npy").is_file():
        print(f"  carving inner val → {keep_dir}", flush=True)
        carve_inner_val(out_dir=keep_dir)
    out.mkdir(parents=True, exist_ok=True)
    curriculum = make_curriculum_record(args, trainer="multihorizon")
    write_curriculum_record(out, curriculum)
    print(format_curriculum_line(curriculum), flush=True)

    pairs_dir = resolve_pairs(args.pairs)
    init_note = str(args.init_ckpt) if args.init_ckpt is not None else "scratch"
    print(
        f"multihorizon {arch}  leads={list(HORIZONS_SEC)}  "
        f"loss={LOSS_NAME}  init={init_note}  → {out}",
        flush=True,
    )
    data, _val_unused, _test, norm, nside, device, cache_root = load_data(
        cfg,
        pairs_dir,
        cache_dir,
        score_test=False,
        val_from_train=True,
    )
    slim_camera_pairs(data)
    print("  camera-only memmaps  (no sat / winds)", flush=True)
    mh = wrap_multihorizon(data, cache_root / "train")
    mh = attach_horizon_lock(mh, cache_root / "train")

    sky = data.norm.sky if norm is None else norm.sky
    model = build_nowcast(
        arch,
        nside=int(nside),
        n_frames=int(cfg.n_frames),
        sky_mean=float(sky.mean),
        sky_std=float(sky.std),
    )
    last_pt = out / "last.pt"
    skip_init = bool(args.resume) and last_pt.is_file()
    if skip_init:
        print(f"  resume {last_pt}  skip --init-ckpt", flush=True)
    elif args.init_ckpt is not None:
        payload = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
        loader = load_nowcast_matching if args.init_full else load_nowcast_backbone
        missing, unexpected = loader(model, payload["state_dict"])
        print(
            f"  init {args.init_ckpt}  full={bool(args.init_full)}  "
            f"missing={len(missing)}  unexpected={len(unexpected)}",
            flush=True,
        )
    else:
        print("  init from scratch (random weights)", flush=True)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    source = json.loads(SOURCE_PATH.read_text()) if SOURCE_PATH.is_file() else {}
    n_train_keep = int(np.load(keep_dir / "train.npy").size)
    n_val_keep = int(np.load(keep_dir / "validation.npy").size)
    _write_json(
        out / "run.json",
        {
            "source": source,
            "started": dt.datetime.now(dt.timezone.utc).isoformat(),
            "arch": arch,
            "pairs": str(pairs_dir),
            "init_ckpt": None if args.init_ckpt is None else str(args.init_ckpt),
            "init_from_scratch": args.init_ckpt is None,
            "horizons_sec": list(HORIZONS_SEC),
            "cache": str(cache_root),
            "nside": nside,
            "n_train_full": data.n,
            "n_train_keep": n_train_keep,
            "n_val": n_val_keep,
            "keep_dir": str(keep_dir),
            "n_params": n_params,
            "curriculum": curriculum,
            "device": str(device),
            "config": {
                "epochs": cfg.epochs,
                "batch_size": cfg.batch_size,
                "lr": cfg.lr,
                "min_lr": cfg.min_lr,
                "weight_decay": cfg.weight_decay,
                "patience": cfg.patience,
                "max_lr_halves": cfg.max_lr_halves,
                "seed": cfg.seed,
                "n_frames": cfg.n_frames,
                "loss": LOSS_NAME,
                "objective": OBJECTIVE,
                "h20_weight": float(args.h20_weight),
                "schedule": str(args.schedule),
                "warmup_epochs": int(args.warmup_epochs),
                "ema": float(args.ema),
                "grad_accum": int(args.grad_accum),
                "val_frac": float(args.val_frac),
                "full_val_every": int(args.full_val_every),
                "band_mix": float(args.band_mix),
                "band_t_min": float(args.band_t_min),
                "band_t_max": float(args.band_t_max),
                "fc_loss_weight": float(args.fc_loss_weight),
                "fc_loss_thresh": float(args.fc_loss_thresh),
                "fc_yield_slack": float(args.fc_yield_slack),
                "init_full": bool(args.init_full),
                "curriculum_stage": curriculum["stage"],
                "curriculum_name": curriculum["name"],
                "parent_checkpoint": curriculum["parent_checkpoint"],
                "camera_only": True,
            },
        },
    )
    print(
        f"training  nside={nside}  train={n_train_keep}  val={n_val_keep}  "
        f"params={n_params:,}  device={device}",
        flush=True,
    )
    info = train_multihorizon(
        model,
        mh,
        cfg,
        out=out,
        norm=norm,
        cache_root=cache_root,
        keep_dir=keep_dir,
        val_frac=float(args.val_frac),
        full_val_every=int(args.full_val_every),
        compile_model=bool(args.compile),
        amp=bool(args.amp),
        band_delta=BAND_DELTA,
        label_sigma=LABEL_SMOOTH_SIGMA,
        run=f"paper_{arch}_multih_t24",
        task="cloud_mask_forecast_multihorizon",
        schedule=str(args.schedule),
        warmup_epochs=int(args.warmup_epochs),
        ema_decay=float(args.ema),
        grad_accum=int(args.grad_accum),
        h20_weight=float(args.h20_weight),
        band_mix=float(args.band_mix),
        band_t_min=float(args.band_t_min),
        band_t_max=float(args.band_t_max),
        fc_loss_weight=float(args.fc_loss_weight),
        fc_loss_thresh=float(args.fc_loss_thresh),
        fc_yield_slack=float(args.fc_yield_slack),
        curriculum=curriculum,
        resume=bool(args.resume),
    )
    info["arch"] = arch
    print(json.dumps(info, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
