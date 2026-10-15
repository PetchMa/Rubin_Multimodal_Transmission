"""Train the multimodal forecaster on every cached lead.

History is encoded once. Each lead (90 s, 5 min, 20 min, 30 min) has its
own wind-advection decode and a centered lead embedding that is the
identity at +20 min. The loss is airmass-weighted yield MSE, averaged
across leads. ``--fc-loss-weight`` adds the false-clear hinge.
``--curriculum-stage calibrate|risk`` is the two-stage schedule.

    python -u -m analysis.forecaster.train_multihorizon --seed 42
    python -u -m analysis.forecaster.run_curriculum \\
        --out /path/to/seed_42
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset

from analysis.forecaster import program as cand
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
from analysis.forecaster.program import parse_ablate
from analysis.forecaster.geom import attach_wind_decode_stats, local_pix_altaz, nside_from_npix
from analysis.forecaster.multihorizon import (
    HORIZON_LABELS,
    HORIZONS_SEC,
    wrap_multihorizon,
)
from analysis.forecaster.port_prep import (
    LABEL_SMOOTH_SIGMA,
    MAX_TRAIN_PER_NIGHT,
    apply_band_mix_train,
    apply_port_keeps,
    attach_horizon_lock,
)
from analysis.forecaster.schedule_util import ModelEma, cosine_lr
from analysis.forecaster.train import (
    BAND_DELTA,
    SOURCE_PATH,
    _enable_fast_cuda,
    _fork_loader,
    _set_sky_norm,
    _snapshot,
    _write_json,
    attach_rich_sat,
    load_data,
)
from analysis.our_model.mask_train import (
    FALSE_CLEAR_T,
    SCHEDULER_MIN_ALT_DEG,
    decode_camera_t,
    masked_airmass_yield_mse,
    masked_false_clear_hinge,
)
from analysis.our_model.model import (
    TrainConfig,
    apply_batch_preprocess,
    apply_eval_budget,
    make_loader,
    resolve_device,
    resolve_pairs,
    save_checkpoint,
    seed_everything,
    train_frac_indices,
    _progress,
    _to_device,
)
from analysis.paths import MNT

DEFAULT_CACHE = MNT / "pairs" / "dream_multih_t24_encoded"
DEFAULT_OUT = MNT / "baselines_neural" / "paper_multih_t24"
DEFAULT_PAIRS = MNT / "data" / "pairs" / "dream_20min"
H20_SEC = 1200
H30_SEC = 1800
OBJECTIVE = "mean_val_yield_mse"
LOSS_NAME = "yield_mse"
EXTRA_KEYS = (
    "sat",
    "meta",
    "meta_q",
    "wind_set",
    "wind_mask",
    "wind_set_q",
    "wind_mask_q",
)
SCALE_EPS = 1e-4
SCALE_BATCHES = 64


def _lab(horizon_sec: int) -> str:
    return HORIZON_LABELS.get(int(horizon_sec), f"{int(horizon_sec)}s")


def _sin_alt_map(npix: int, *, device, dtype) -> torch.Tensor:
    alt, _ = local_pix_altaz(nside_from_npix(int(npix)))
    return torch.sin(
        torch.deg2rad(torch.as_tensor(alt, device=device, dtype=dtype))
    ).clamp(min=0.0)


def _forward_horizons(model, b):
    extra = [b[k] for k in EXTRA_KEYS if k in b]
    leads = b["lead_sec"]
    dt = b.get("horizon_dt_sec")
    if leads.ndim == 1:
        lead_list = [leads]
    else:
        lead_list = []
        for k in range(int(leads.shape[-1])):
            col = leads[:, k]
            if dt is not None:
                dtk = dt[:, k].to(dtype=col.dtype)
                col = torch.where(dtk > 1.0, dtk, col)
            lead_list.append(col)
    hats = model.forward_horizons(
        b["sky"],
        b["sky_mask"],
        *extra,
        leads=lead_list,
        unix=b.get("unix"),
        sat_age_sec=b.get("sat_age_sec"),
    )
    return torch.stack(hats, dim=1)


def _decode_batch(b, sky_norm, label_sigma: float):
    y, ym = b["y"], b["y_mask"]
    if y.ndim == 3:
        bsz, n_h, npix = int(y.shape[0]), int(y.shape[1]), int(y.shape[2])
        t_y, y_valid = decode_camera_t(
            y.reshape(bsz * n_h, npix),
            ym.reshape(bsz * n_h, npix),
            sky_norm,
            sigma=label_sigma,
        )
        t_y = t_y.reshape(bsz, n_h, npix)
        y_valid = y_valid.reshape(bsz, n_h, npix)
    else:
        t_y, y_valid = decode_camera_t(y, ym, sky_norm, sigma=label_sigma)
        t_y = t_y.unsqueeze(1)
        y_valid = y_valid.unsqueeze(1)
    t_last, last_valid = decode_camera_t(
        b["last"], b["last_mask"], sky_norm, sigma=label_sigma
    )
    valid = y_valid * last_valid.unsqueeze(1)
    hit = b.get("horizon_hit")
    if hit is not None:
        valid = valid * hit.reshape(hit.shape[0], -1).unsqueeze(-1)
    return t_y, t_last, valid


def _airmass_mae(t_hat, t_true, valid, sin_alt) -> torch.Tensor:
    w = sin_alt.to(dtype=t_hat.dtype).clamp(min=0.0)
    min_sin = math.sin(math.radians(float(SCHEDULER_MIN_ALT_DEG)))
    w = torch.where(w >= min_sin, w, torch.zeros_like(w))
    ae = (t_hat.clamp(0.0, 1.0) - t_true.clamp(0.0, 1.0)).abs()
    e = ae.reshape(ae.shape[0], -1)
    m = valid.reshape(valid.shape[0], -1).to(dtype=e.dtype)
    ww = w.reshape(w.shape[0], -1) * m
    den = ww.sum(dim=-1)
    ok = den > 0
    if int(ok.sum().item()) == 0:
        return e.new_zeros(())
    per = (e * ww).sum(dim=-1) / den.clamp_min(1e-6)
    return per[ok].mean()


def _yield_mse_per_horizon(hat, t_y, valid, t_last, sin_alt) -> tuple[list, list]:
    preds, pers = [], []
    n_h = int(t_y.shape[1])
    bsz = int(t_y.shape[0])
    for k in range(n_h):
        vk = valid[:, k]
        if float(vk.reshape(bsz, -1).sum()) <= 0:
            preds.append(None)
            pers.append(None)
            continue
        preds.append(masked_airmass_yield_mse(hat[:, k], t_y[:, k], vk, sin_alt))
        pers.append(masked_airmass_yield_mse(t_last, t_y[:, k], vk, sin_alt))
    return preds, pers


def _fc_hinge_per_horizon(hat, t_y, valid, sin_alt, thresh: float) -> list:
    terms = []
    n_h = int(t_y.shape[1])
    bsz = int(t_y.shape[0])
    for k in range(n_h):
        vk = valid[:, k]
        if float(vk.reshape(bsz, -1).sum()) <= 0:
            terms.append(None)
            continue
        terms.append(
            masked_false_clear_hinge(
                hat[:, k], t_y[:, k], vk, sin_alt, thresh=thresh
            )
        )
    return terms


def _mean_weighted(terms: list, weights: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    present = [(k, t) for k, t in enumerate(terms) if t is not None]
    if not present:
        return ref.new_zeros(())
    stacked = torch.stack([t for _, t in present])
    w = weights[[k for k, _ in present]].to(device=stacked.device, dtype=stacked.dtype)
    return (stacked * w).sum() / w.clamp_min(1e-8).sum()


def _trim_truncated_jsonl(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    kept: list[str] = []
    for line in path.read_text().splitlines():
        raw = line.strip()
        if not raw:
            continue
        try:
            json.loads(raw)
        except json.JSONDecodeError:
            continue
        kept.append(raw + "\n")
    path.write_text("".join(kept))


def _clone_state_dict(state: dict) -> dict[str, torch.Tensor]:
    return {
        k.replace("._orig_mod.", "."): v.detach().clone() for k, v in state.items()
    }


def _load_model_state(model: nn.Module, state: dict) -> tuple[list[str], list[str]]:
    cleaned = {
        k.replace("._orig_mod.", "."): v for k, v in state.items()
    }
    return model.load_state_dict(cleaned, strict=False)


def _ckpt_info(*, task: str, run: str, kind: str, cfg: TrainConfig, extra: dict) -> dict:
    payload = {
        "task": task,
        "run": run,
        "checkpoint_kind": kind,
        "multihorizon": True,
        "n_frames": int(cfg.n_frames),
        "config": {
            "multihorizon": True,
            "n_frames": int(cfg.n_frames),
            "loss": LOSS_NAME,
            "objective": OBJECTIVE,
        },
    }
    payload.update(extra)
    return payload


@torch.no_grad()
def estimate_persist_scales(
    loader,
    sky_norm,
    *,
    device,
    label_sigma: float,
    n_h: int,
    max_batches: int = SCALE_BATCHES,
) -> torch.Tensor:
    """Fixed train-set persistence yield MSE per horizon. Not per-batch."""
    acc = torch.zeros(n_h, dtype=torch.float64)
    seen = torch.zeros(n_h, dtype=torch.float64)
    sin_alt = None
    n_batches = 0
    for raw in loader:
        b = _to_device(raw, device)
        t_y, t_last, valid = _decode_batch(b, sky_norm, label_sigma)
        if sin_alt is None:
            sin_alt = _sin_alt_map(
                int(t_last.shape[-1]), device=t_last.device, dtype=t_last.dtype
            )
            if sin_alt.ndim == 1:
                sin_alt = sin_alt.unsqueeze(0)
        bs = int(t_y.shape[0])
        _, pers = _yield_mse_per_horizon(
            t_last.unsqueeze(1).expand_as(t_y), t_y, valid, t_last, sin_alt
        )
        for k, p in enumerate(pers):
            if p is None:
                continue
            v = float(p.detach())
            if math.isfinite(v) and v > 0:
                acc[k] += v * bs
                seen[k] += bs
        n_batches += 1
        if n_batches >= int(max_batches):
            break
    scales = torch.ones(n_h, dtype=torch.float32)
    for k in range(n_h):
        if seen[k] > 0:
            scales[k] = float(acc[k] / seen[k])
    return scales.clamp_min(SCALE_EPS)


@torch.no_grad()
def eval_horizons(
    model,
    loader,
    sky_norm,
    device,
    *,
    horizons_sec: tuple[int, ...],
    scales: torch.Tensor,
    label_sigma: float,
    amp_dtype,
    use_amp: bool,
    verbose: bool = False,
) -> dict:
    """Yield MSE / MAYE / η_FC / photometric MAYE at every lead."""
    model.eval()
    n_h = len(horizons_sec)
    min_sin = math.sin(math.radians(float(SCHEDULER_MIN_ALT_DEG)))
    fc_cut = float(FALSE_CLEAR_T)
    acc = {
        k: {
            "yield": 0.0,
            "persist": 0.0,
            "maye": 0.0,
            "photo": 0.0,
            "n": 0,
            "fc_fn": 0,
            "fc_den": 0,
        }
        for k in range(n_h)
    }
    for raw in _progress(
        loader, desc="  val", total=len(loader), leave=False, disable=not verbose
    ):
        b = apply_batch_preprocess(model, _to_device(raw, device))
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            hat = _forward_horizons(model, b)
        t_y, t_last, valid = _decode_batch(b, sky_norm, label_sigma)
        sin_alt = getattr(model, "last_sin_alt", None)
        if sin_alt is None:
            sin_alt = _sin_alt_map(
                int(t_last.shape[-1]), device=t_last.device, dtype=t_last.dtype
            )
            if sin_alt.ndim == 1:
                sin_alt = sin_alt.unsqueeze(0).expand(t_last.shape[0], -1)
        bs = int(t_y.shape[0])
        preds, pers = _yield_mse_per_horizon(hat, t_y, valid, t_last, sin_alt)
        alt_ok = sin_alt >= min_sin
        last_ok = b["last_mask"] > 0
        if last_ok.ndim == 1:
            last_ok = last_ok.unsqueeze(0)
        for k in range(n_h):
            vk = valid[:, k]
            if preds[k] is None:
                continue
            acc[k]["yield"] += float(preds[k]) * bs
            acc[k]["persist"] += float(pers[k]) * bs
            acc[k]["maye"] += float(_airmass_mae(hat[:, k], t_y[:, k], vk, sin_alt)) * bs
            photo = (
                vk
                * last_ok.to(dtype=vk.dtype)
                * (t_last > fc_cut).to(dtype=vk.dtype)
                * (t_y[:, k] > fc_cut).to(dtype=vk.dtype)
            )
            acc[k]["photo"] += float(
                _airmass_mae(hat[:, k], t_y[:, k], photo, sin_alt)
            ) * bs
            acc[k]["n"] += bs
            conf = (vk > 0) & last_ok & alt_ok
            gold_fc = (t_y[:, k] <= fc_cut) & conf
            pred_fc = (hat[:, k] <= fc_cut) & conf
            acc[k]["fc_fn"] += int((~pred_fc & gold_fc).sum().item())
            acc[k]["fc_den"] += int(gold_fc.sum().item())
    out: dict[str, dict] = {}
    norms = []
    yields = []
    for k, h in enumerate(horizons_sec):
        n = max(acc[k]["n"], 1)
        ymse = acc[k]["yield"] / n
        pms = acc[k]["persist"] / n
        scale = float(scales[k].clamp_min(SCALE_EPS))
        fc_den = acc[k]["fc_den"]
        rec = {
            "horizon_sec": int(h),
            "label": _lab(h),
            "yield_mse": ymse,
            "persist_yield_mse": pms,
            "persist_norm_mse": ymse / scale,
            "maye": acc[k]["maye"] / n,
            "photometric_maye": acc[k]["photo"] / n,
            "false_clear_063": (
                acc[k]["fc_fn"] / fc_den if fc_den else float("nan")
            ),
            "n_windows": acc[k]["n"],
            "scale": scale,
        }
        out[_lab(h)] = rec
        norms.append(rec["persist_norm_mse"])
        yields.append(ymse)
    finite_n = [x for x in norms if math.isfinite(x)]
    finite_y = [x for x in yields if math.isfinite(x)]
    return {
        "horizons": out,
        "joint_yield_mse": float(sum(finite_y) / max(len(finite_y), 1)),
        "joint_persist_norm_mse": float(sum(finite_n) / max(len(finite_n), 1)),
    }


def train_multihorizon(
    model: nn.Module,
    data,
    cfg: TrainConfig,
    *,
    out: Path,
    norm,
    cache_root: Path,
    keep_dir: Path | None,
    val_frac: float,
    full_val_every: int,
    compile_model: bool,
    amp: bool,
    band_delta: float,
    label_sigma: float,
    run: str,
    task: str,
    horizons_sec: tuple[int, ...] = HORIZONS_SEC,
    schedule: str = "plateau",
    warmup_epochs: int = 5,
    ema_decay: float = 0.999,
    grad_accum: int = 1,
    h20_weight: float = 1.0,
    band_mix: float = 0.0,
    band_t_min: float = 0.1,
    band_t_max: float = 0.9,
    fc_loss_weight: float = 0.0,
    fc_loss_thresh: float = FALSE_CLEAR_T,
    fc_yield_slack: float = 1.03,
    curriculum: dict | None = None,
    resume: bool = False,
) -> dict:
    cfg = apply_eval_budget(cfg)
    device = resolve_device(cfg.device)
    _enable_fast_cuda()
    model = model.to(device)
    if compile_model:
        model = torch.compile(model)
        print("  torch.compile(model)", flush=True)
    use_amp = bool(amp) and device.type == "cuda"
    amp_dtype = (
        torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    )
    scaler = (
        torch.amp.GradScaler("cuda")
        if use_amp and amp_dtype == torch.float16
        else None
    )
    night_cap = 0 if cfg.train_frac < 1.0 else MAX_TRAIN_PER_NIGHT
    mix_frac = float(band_mix)
    mix_info: dict | None = None
    mix_sampler = None
    if mix_frac > 0.0:
        if cfg.train_frac < 1.0:
            raise SystemExit("--band-mix needs --train-frac 1.0")
        train_set, val_full, moon_keep, _ = apply_port_keeps(
            data,
            keep_dir,
            Path(cache_root) / "train",
            val_from_train=True,
            max_train_per_night=0,
        )
        sky_for_mix = data.norm.sky if norm is None else norm.sky
        train_set, mix_sampler, mix_info = apply_band_mix_train(
            data,
            moon_keep,
            cache_split=Path(cache_root) / "train",
            sky_mean=float(sky_for_mix.mean),
            sky_std=float(sky_for_mix.std),
            max_per_night=MAX_TRAIN_PER_NIGHT,
            band_frac=mix_frac,
            t_min=float(band_t_min),
            t_max=float(band_t_max),
            seed=int(cfg.seed),
        )
        _write_json(out / "band_mix.json", mix_info)
    else:
        train_set, val_full, _, _ = apply_port_keeps(
            data,
            keep_dir,
            Path(cache_root) / "train",
            val_from_train=True,
            max_train_per_night=night_cap,
        )
    n_train = len(train_set)
    n_val_full = len(val_full)
    if cfg.train_frac < 1.0:
        keep, frac_info = train_frac_indices(train_set, cfg.train_frac, cfg.seed)
        train_set = Subset(train_set, keep)
        n_train = len(keep)
        print(
            f"  train subset  {n_train}/{frac_info['n']}  "
            f"({cfg.train_frac:.0%} of windows, seed={cfg.seed})",
            flush=True,
        )
    val_epoch = val_full
    n_val = n_val_full
    if float(val_frac) < 1.0:
        vkeep, vinfo = train_frac_indices(val_full, float(val_frac), cfg.seed + 1)
        val_epoch = Subset(val_full, vkeep)
        n_val = len(vkeep)
        print(
            f"  val subset    {n_val}/{vinfo['n']}  "
            f"({float(val_frac):.0%} of windows each epoch; "
            f"full val every {int(full_val_every)} epochs)",
            flush=True,
        )

    if mix_sampler is not None:
        n_train = int(len(mix_sampler))
    sky_norm = data.norm.sky if norm is None else norm.sky
    _set_sky_norm(model, sky_norm)
    if mix_sampler is not None:
        train_loader = _fork_loader(train_set, cfg, device, mix_sampler)
        print(
            f"  loader band-mix  n_epoch={len(mix_sampler)}  "
            f"unique={len(train_set)}  seed={cfg.seed}",
            flush=True,
        )
    else:
        train_loader = make_loader(
            train_set, cfg, shuffle=True, device=device, block_shuffle=True
        )
    scale_loader = make_loader(
        train_set, cfg, shuffle=False, device=device, block_shuffle=False
    )
    scales = estimate_persist_scales(
        scale_loader,
        sky_norm,
        device=device,
        label_sigma=label_sigma,
        n_h=len(horizons_sec),
    )
    scales = scales.to(device)
    print(
        "  persist yield-MSE scales (fixed)  "
        + "  ".join(
            f"{_lab(h)}={float(scales[k]):.5f}" for k, h in enumerate(horizons_sec)
        ),
        flush=True,
    )
    _write_json(
        out / "persist_scales.json",
        {_lab(h): float(scales[k]) for k, h in enumerate(horizons_sec)},
    )
    h_w = torch.ones(len(horizons_sec), dtype=torch.float32)
    try:
        h_w[list(horizons_sec).index(H20_SEC)] = float(h20_weight)
    except ValueError:
        pass
    print(
        f"  multihorizon train  loss={LOSS_NAME}  objective={OBJECTIVE}  "
        f"leads={list(horizons_sec)}  h20_weight={float(h20_weight):g}  "
        f"label_sigma={label_sigma:g}  n_train={n_train}  n_val={n_val}/{n_val_full}",
        flush=True,
    )
    fc_w = float(fc_loss_weight)
    fc_cut = float(fc_loss_thresh)
    fc_slack = float(fc_yield_slack)
    if fc_w > 0:
        print(
            f"  fc hinge  λ={fc_w:g}  T≤{fc_cut:g}  all leads  "
            f"relu(T̂−cut) on gold-cloud ∩ last ∩ alt≥30°  "
            f"mean-weighted like yield MSE  "
            f"yield cap {fc_slack:g}× init +20 min val",
            flush=True,
        )
    val_loader = make_loader(val_epoch, cfg, shuffle=False, device=device)
    full_loader = (
        val_loader
        if n_val == n_val_full
        else make_loader(val_full, cfg, shuffle=False, device=device)
    )
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    use_cosine = str(schedule).strip().lower() == "cosine"
    accum = max(1, int(grad_accum))
    ema = ModelEma(model, ema_decay) if use_cosine else None
    if use_cosine:
        print(
            f"  cosine schedule  epochs={cfg.epochs}  warmup={warmup_epochs}  "
            f"ema={ema_decay}  accum={accum}  (no early stop)",
            flush=True,
        )

    epoch_log = out / "epochs.jsonl"
    metrics_path = out / "metrics.json"
    joint_ckpt = out / "best_joint.pt"
    h20_ckpt = out / "best_20min.pt"
    h30_ckpt = out / "best_30min.pt"
    last_ckpt = out / "last.pt"
    alias_ckpt = out / "best.pt"

    def run_epoch() -> float:
        model.train()
        total, n_seen = 0.0, 0
        opt.zero_grad(set_to_none=True)
        n_step = 0
        for raw in _progress(
            train_loader, desc="  train", total=len(train_loader), leave=False
        ):
            b = apply_batch_preprocess(model, _to_device(raw, device))
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=use_amp
            ):
                hat = _forward_horizons(model, b)
                t_y, t_last, valid = _decode_batch(b, sky_norm, label_sigma)
                sin_alt = getattr(model, "last_sin_alt", None)
                if sin_alt is None:
                    sin_alt = _sin_alt_map(
                        int(t_last.shape[-1]), device=hat.device, dtype=hat.dtype
                    )
                    if sin_alt.ndim == 1:
                        sin_alt = sin_alt.unsqueeze(0).expand(hat.shape[0], -1)
                preds, _pers = _yield_mse_per_horizon(hat, t_y, valid, t_last, sin_alt)
                batch_loss = _mean_weighted(preds, h_w, hat) / float(accum)
                if fc_w > 0:
                    fc_terms = _fc_hinge_per_horizon(
                        hat, t_y, valid, sin_alt, thresh=fc_cut
                    )
                    batch_loss = batch_loss + (
                        fc_w * _mean_weighted(fc_terms, h_w, hat) / float(accum)
                    )
            if scaler is not None:
                scaler.scale(batch_loss).backward()
            else:
                batch_loss.backward()
            n_step += 1
            if n_step % accum == 0:
                if scaler is not None:
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                if ema is not None:
                    ema.update(model)
                opt.zero_grad(set_to_none=True)
            bs = int(b["y"].shape[0])
            total += float(batch_loss.detach()) * bs * accum
            n_seen += bs
        if n_step % accum != 0:
            if scaler is not None:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            if ema is not None:
                ema.update(model)
            opt.zero_grad(set_to_none=True)
        return total / max(n_seen, 1)

    def _save(path: Path, kind: str, extra: dict, scores: dict) -> None:
        save_checkpoint(
            path,
            model,
            data.norm if norm is None else norm,
            cfg,
            _ckpt_info(task=task, run=run, kind=kind, cfg=cfg, extra=extra),
            scores,
        )

    t0 = time.perf_counter()
    best_joint = float("inf")
    best_h20 = float("inf")
    best_h30 = float("inf")
    best_epoch = -1
    best_va: dict = {}
    best_state = None
    rows: list[dict] = []
    stale = 0
    n_halves = 0
    stop_reason = "max_epochs"
    epoch = -1
    start_epoch = 0
    fc_ckpt = out / "best_20min_fc.pt"
    init_h20 = None
    init_fc = None
    best_fc = float("inf")
    best_fc_epoch = -1
    resumed = False
    if resume and last_ckpt.is_file():
        _trim_truncated_jsonl(epoch_log)
        last = torch.load(last_ckpt, map_location=device, weights_only=False)
        missing, unexpected = _load_model_state(model, last["state_dict"])
        last_info = last.get("info") or {}
        start_epoch = int(last_info.get("epoch", -1)) + 1
        stale = int(last_info.get("stale", 0))
        n_halves = int(last_info.get("n_lr_halves", 0))
        lr_resume = last_info.get("lr")
        if lr_resume is not None:
            for group in opt.param_groups:
                group["lr"] = float(lr_resume)
        best_joint = float(last_info.get("best_joint_yield_mse", best_joint))
        best_epoch = int(last_info.get("best_epoch", best_epoch))
        if metrics_path.is_file():
            prev = json.loads(metrics_path.read_text())
            best_h20 = float(prev.get("best_20min_yield_mse", best_h20))
            best_h30 = float(prev.get("best_30min_yield_mse", best_h30))
            if prev.get("best_20min_fc") is not None:
                best_fc = float(prev["best_20min_fc"])
            best_fc_epoch = int(prev.get("best_20min_fc_epoch") or best_fc_epoch)
        if alias_ckpt.is_file():
            best_payload = torch.load(
                alias_ckpt, map_location="cpu", weights_only=False
            )
            best_state = _clone_state_dict(best_payload["state_dict"])
            best_info = best_payload.get("info") or {}
            best_joint = float(best_info.get("best_joint_yield_mse", best_joint))
            best_epoch = int(best_info.get("best_epoch", best_epoch))
            best_scores = best_payload.get("scores") or {}
            if best_scores.get("validation"):
                best_va = dict(best_scores["validation"])
        if epoch_log.is_file():
            for line in epoch_log.read_text().splitlines():
                if not line.strip():
                    continue
                rows.append(json.loads(line))
            rows = [r for r in rows if int(r.get("epoch", -1)) < start_epoch]
        resumed = True
        print(
            f"  resume {last_ckpt}  next_epoch={start_epoch}  "
            f"stale={stale}  halves={n_halves}/{cfg.max_lr_halves}  "
            f"lr={float(opt.param_groups[0]['lr']):.2e}  "
            f"best_joint={best_joint:.5f}@{best_epoch}  "
            f"missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )
    elif resume:
        print(f"  --resume set but {last_ckpt} missing; training from init", flush=True)
        epoch_log.write_text("")
    else:
        epoch_log.write_text("")

    if fc_w > 0 and resumed and (out / "fc_init.json").is_file():
        fc_meta = json.loads((out / "fc_init.json").read_text())
        init_h20 = float(fc_meta["init_val_20min_yield_mse"])
        init_fc = float(fc_meta["init_val_20min_false_clear_063"])
        if fc_meta.get("best_fc") is not None:
            best_fc = float(fc_meta["best_fc"])
        best_fc_epoch = int(fc_meta.get("best_fc_epoch") or best_fc_epoch)
        print(
            f"  resume fc cap  +20 min yieldMSE {init_h20:.5f}  "
            f"η_FC {init_fc:.3f}  cap {fc_slack * init_h20:.5f}  "
            f"best_fc={best_fc if math.isfinite(best_fc) else None}",
            flush=True,
        )
    elif fc_w > 0:
        va0 = eval_horizons(
            model,
            val_loader,
            sky_norm,
            device,
            horizons_sec=horizons_sec,
            scales=scales,
            label_sigma=label_sigma,
            amp_dtype=amp_dtype,
            use_amp=use_amp,
            verbose=True,
        )
        h20_0 = va0["horizons"][_lab(H20_SEC)]
        init_h20 = float(h20_0["yield_mse"])
        init_fc = float(h20_0["false_clear_063"])
        best_fc = init_fc if math.isfinite(init_fc) else float("inf")
        _save(
            fc_ckpt,
            "best_20min_fc",
            {
                **{
                    "loss": LOSS_NAME,
                    "objective": OBJECTIVE,
                    "label_sigma": label_sigma,
                    "h20_weight": float(h20_weight),
                },
                "best_epoch": -1,
                "best_20min_yield_mse": init_h20,
                "best_20min_false_clear_063": init_fc,
                "fc_loss_weight": fc_w,
                "fc_loss_thresh": fc_cut,
                "fc_yield_slack": fc_slack,
                "fc_init_20min_yield_mse": init_h20,
                "curriculum": curriculum,
            },
            {"validation": va0},
        )
        _write_json(
            out / "fc_init.json",
            {
                "init_val_20min_yield_mse": init_h20,
                "init_val_20min_false_clear_063": init_fc,
                "fc_loss_weight": fc_w,
                "fc_loss_thresh": fc_cut,
                "fc_yield_slack": fc_slack,
                "fc_yield_cap": fc_slack * float(init_h20),
                "best_fc": best_fc,
                "best_fc_epoch": best_fc_epoch,
                "curriculum": curriculum,
            },
        )
        print(
            f"  fc init val  +20 min yieldMSE {init_h20:.5f}  "
            f"η_FC {init_fc:.3f}  cap {fc_slack * init_h20:.5f}  "
            f"seeded {fc_ckpt.name}",
            flush=True,
        )

    if start_epoch >= int(cfg.epochs):
        epoch = max(start_epoch - 1, -1)
        print(
            f"  resume already finished {start_epoch}/{cfg.epochs} epochs; "
            "writing summary",
            flush=True,
        )

    for epoch in range(start_epoch, int(cfg.epochs)):
        if mix_sampler is not None:
            mix_sampler.set_epoch(epoch)
        epoch_t0 = time.perf_counter()
        if use_cosine:
            lr_now = cosine_lr(
                epoch,
                base=float(cfg.lr),
                min_lr=float(cfg.min_lr),
                warmup=int(warmup_epochs),
                total=int(cfg.epochs),
            )
            for group in opt.param_groups:
                group["lr"] = lr_now
        tr_loss = run_epoch()
        va = eval_horizons(
            model,
            val_loader,
            sky_norm,
            device,
            horizons_sec=horizons_sec,
            scales=scales,
            label_sigma=label_sigma,
            amp_dtype=amp_dtype,
            use_amp=use_amp,
            verbose=True,
        )
        ran_full = False
        va_full = None
        if n_val != n_val_full and int(full_val_every) > 0:
            if (epoch + 1) % int(full_val_every) == 0:
                va_full = eval_horizons(
                    model,
                    full_loader,
                    sky_norm,
                    device,
                    horizons_sec=horizons_sec,
                    scales=scales,
                    label_sigma=label_sigma,
                    amp_dtype=amp_dtype,
                    use_amp=use_amp,
                    verbose=True,
                )
                ran_full = True
        lr_now = float(opt.param_groups[0]["lr"])
        joint = float(va["joint_yield_mse"])
        persist_joint = float(va["joint_persist_norm_mse"])
        h20 = float(va["horizons"][_lab(H20_SEC)]["yield_mse"])
        h30 = float(va["horizons"][_lab(H30_SEC)]["yield_mse"])
        improved = math.isfinite(joint) and joint < best_joint - 1e-8
        rec = {
            "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
            "epoch": epoch,
            "train_loss": tr_loss,
            "loss": LOSS_NAME,
            "objective": OBJECTIVE,
            "joint_yield_mse": joint,
            "joint_persist_norm_mse": persist_joint,
            "horizons": va["horizons"],
            "val_full": None if va_full is None else va_full,
            "lr": lr_now,
            "n_train_windows": n_train,
            "n_val_windows": n_val,
            "n_val_full_windows": n_val_full,
            "stale": stale,
            "n_lr_halves": n_halves,
            "epoch_sec": time.perf_counter() - epoch_t0,
            "band_delta": band_delta,
        }
        rows.append(rec)
        epoch_log.write_text("".join(json.dumps(r) + "\n" for r in rows))

        extra_common = {
            "loss": LOSS_NAME,
            "objective": OBJECTIVE,
            "label_sigma": label_sigma,
            "h20_weight": float(h20_weight),
            "persist_scales": {
                _lab(h): float(scales[k]) for k, h in enumerate(horizons_sec)
            },
            "fc_loss_weight": fc_w,
            "fc_loss_thresh": fc_cut,
            "fc_yield_slack": fc_slack,
            "curriculum": curriculum,
        }
        if getattr(model, "name", None):
            extra_common["arch"] = str(model.name)
        scores = {"validation": va, "validation_full": va_full}
        if not use_cosine:
            if improved:
                best_joint = joint
                best_epoch = epoch
                best_va = dict(va)
                best_state = _snapshot(model)
                stale = 0
                extra = {
                    **extra_common,
                    "best_epoch": epoch,
                    "best_joint_yield_mse": joint,
                    "best_20min_yield_mse": h20,
                }
                _save(joint_ckpt, "best_joint", extra, scores)
                _save(alias_ckpt, "best_joint", extra, scores)
                print(
                    f"  saved {joint_ckpt.name}  joint {joint:.5f}  "
                    f"20min yieldMSE {h20:.5f}  30min yieldMSE {h30:.5f}",
                    flush=True,
                )
            else:
                stale += 1
            if math.isfinite(h20) and h20 < best_h20 - 1e-8:
                best_h20 = h20
                _save(
                    h20_ckpt,
                    "best_20min",
                    {**extra_common, "best_epoch": epoch, "best_20min_yield_mse": h20},
                    scores,
                )
                print(f"  saved {h20_ckpt.name}  20min yieldMSE {h20:.5f}", flush=True)
            if math.isfinite(h30) and h30 < best_h30 - 1e-8:
                best_h30 = h30
                _save(
                    h30_ckpt,
                    "best_30min",
                    {**extra_common, "best_epoch": epoch, "best_30min_yield_mse": h30},
                    scores,
                )
                print(f"  saved {h30_ckpt.name}  30min yieldMSE {h30:.5f}", flush=True)
            if fc_w > 0 and init_h20 is not None:
                fc063 = float(va["horizons"][_lab(H20_SEC)]["false_clear_063"])
                cap = fc_slack * float(init_h20)
                rec["fc_yield_cap"] = cap
                rec["fc_init_20min_yield_mse"] = init_h20
                if (
                    math.isfinite(h20)
                    and math.isfinite(fc063)
                    and h20 <= cap + 1e-12
                    and fc063 < best_fc - 1e-8
                ):
                    best_fc = fc063
                    best_fc_epoch = epoch
                    _save(
                        fc_ckpt,
                        "best_20min_fc",
                        {
                            **extra_common,
                            "best_epoch": epoch,
                            "best_20min_yield_mse": h20,
                            "best_20min_false_clear_063": fc063,
                            "fc_loss_weight": fc_w,
                            "fc_loss_thresh": fc_cut,
                            "fc_yield_slack": fc_slack,
                            "fc_init_20min_yield_mse": init_h20,
                            "fc_yield_cap": cap,
                        },
                        scores,
                    )
                    print(
                        f"  saved {fc_ckpt.name}  η_FC {fc063:.3f}  "
                        f"20min yieldMSE {h20:.5f} ≤ cap {cap:.5f}",
                        flush=True,
                    )
                    _write_json(
                        out / "fc_init.json",
                        {
                            "init_val_20min_yield_mse": init_h20,
                            "init_val_20min_false_clear_063": init_fc,
                            "fc_loss_weight": fc_w,
                            "fc_loss_thresh": fc_cut,
                            "fc_yield_slack": fc_slack,
                            "fc_yield_cap": cap,
                            "best_fc": best_fc,
                            "best_fc_epoch": best_fc_epoch,
                            "curriculum": curriculum,
                        },
                    )
                rec["best_fc"] = best_fc if math.isfinite(best_fc) else None
        else:
            best_joint = joint
            best_epoch = epoch
            best_va = dict(va)
            best_h20 = h20
            best_h30 = h30

        _save(
            last_ckpt,
            "last",
            {
                **extra_common,
                "epoch": epoch,
                "joint_yield_mse": joint,
                "joint_persist_norm_mse": persist_joint,
                "best_epoch": best_epoch,
                "best_joint_yield_mse": best_joint,
                "stale": stale,
                "n_lr_halves": n_halves,
                "lr": lr_now,
            },
            scores,
        )
        metrics = {
            "run": run,
            "task": task,
            "horizons_sec": list(horizons_sec),
            "loss": LOSS_NAME,
            "objective": OBJECTIVE,
            "latest": rec,
            "best_joint": {
                "epoch": best_epoch,
                "joint_yield_mse": best_joint,
                "horizons": best_va.get("horizons"),
            },
            "best_20min_yield_mse": best_h20,
            "best_30min_yield_mse": best_h30,
            "best_20min_fc": best_fc if fc_w > 0 else None,
            "best_20min_fc_epoch": best_fc_epoch if fc_w > 0 else None,
            "fc_loss_weight": fc_w,
            "n_train_windows": n_train,
            "n_val_windows": n_val,
            "epochs_run": epoch + 1,
            "n_lr_halves": n_halves,
            "stale": stale,
            "elapsed_sec": time.perf_counter() - t0,
        }
        _write_json(metrics_path, metrics)
        bits = [
            f"{lab} yMSE {va['horizons'][lab]['yield_mse']:.4f} "
            f"MAYE {va['horizons'][lab]['maye']:.4f} "
            f"η_FC {va['horizons'][lab]['false_clear_063']:.3f} "
            f"pNorm {va['horizons'][lab]['persist_norm_mse']:.3f}"
            for lab in (_lab(h) for h in horizons_sec)
        ]
        print(
            f"epoch {epoch:3d}  train {tr_loss:.4f}  joint {joint:.4f}  "
            + "  ".join(bits)
            + f"  best {best_joint:.4f}@{best_epoch}  lr {lr_now:.2e}  stale {stale}"
            + ("  [full val]" if ran_full else ""),
            flush=True,
        )
        at_floor = lr_now <= cfg.min_lr * 1.01
        cannot_reduce = at_floor or n_halves >= cfg.max_lr_halves
        if (not use_cosine) and stale >= cfg.patience:
            if cannot_reduce:
                stop_reason = (
                    "val_plateau_min_lr" if at_floor else "val_plateau_max_halves"
                )
                break
            new_lr = max(lr_now * 0.5, float(cfg.min_lr))
            for group in opt.param_groups:
                group["lr"] = new_lr
            n_halves += 1
            stale = 0
            if best_state is not None:
                model.load_state_dict(best_state)
            print(
                f"  plateau at epoch {epoch}: joint {joint:.5f}  "
                f"lr {lr_now:.2e} → {new_lr:.2e}  "
                f"(half {n_halves}/{cfg.max_lr_halves}, restored best_joint)",
                flush=True,
            )

    if use_cosine and ema is not None:
        ema.copy_to(model)
        stop_reason = "cosine_complete"
        extra_end = {
            "loss": LOSS_NAME,
            "objective": OBJECTIVE,
            "label_sigma": label_sigma,
            "schedule": "cosine",
            "ema": ema_decay,
            "best_epoch": int(epoch) if epoch >= 0 else 0,
            "persist_scales": {
                _lab(h): float(scales[k]) for k, h in enumerate(horizons_sec)
            },
        }
        scores_end = {"validation": best_va}
        _save(alias_ckpt, "ema", extra_end, scores_end)
        _save(joint_ckpt, "ema", extra_end, scores_end)
        _save(h20_ckpt, "ema", extra_end, scores_end)
        _save(h30_ckpt, "ema", extra_end, scores_end)
        print(f"  saved EMA → {alias_ckpt}", flush=True)
    elif best_state is not None:
        model.load_state_dict(best_state)
    if n_val != n_val_full:
        va_end = eval_horizons(
            model,
            full_loader,
            sky_norm,
            device,
            horizons_sec=horizons_sec,
            scales=scales,
            label_sigma=label_sigma,
            amp_dtype=amp_dtype,
            use_amp=use_amp,
            verbose=True,
        )
        print(
            f"val-full  joint {va_end['joint_yield_mse']:.5f}  "
            f"pNorm {va_end['joint_persist_norm_mse']:.5f}  n={n_val_full}",
            flush=True,
        )
    else:
        va_end = best_va
    info = {
        "task": task,
        "run": run,
        "horizons_sec": list(horizons_sec),
        "n_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "loss": LOSS_NAME,
        "objective": OBJECTIVE,
        "best_joint_yield_mse": best_joint,
        "best_20min_yield_mse": best_h20,
        "best_30min_yield_mse": best_h30,
        "best_20min_fc": best_fc if fc_w > 0 else None,
        "best_20min_fc_epoch": best_fc_epoch if fc_w > 0 else None,
        "fc_loss_weight": fc_w,
        "best_20min_fc_ckpt": str(fc_ckpt) if fc_w > 0 else None,
        "best_epoch": best_epoch,
        "best_horizons": best_va.get("horizons"),
        "val_full": va_end,
        "epochs_run": int(epoch + 1) if epoch >= 0 else 0,
        "stop_reason": stop_reason,
        "n_train_windows": n_train,
        "n_val_windows": n_val,
        "n_val_full_windows": n_val_full,
        "n_lr_halves": n_halves,
        "elapsed_sec": time.perf_counter() - t0,
        "best_ckpt": str(joint_ckpt),
        "best_20min_ckpt": str(h20_ckpt),
        "best_30min_ckpt": str(h30_ckpt),
        "last_ckpt": str(last_ckpt),
        "multihorizon": True,
        "h20_weight": float(h20_weight),
        "band_mix": mix_info,
        "curriculum": curriculum,
        "persist_scales": {
            _lab(h): float(scales[k]) for k, h in enumerate(horizons_sec)
        },
    }
    _write_json(
        out / "metrics.json", {**json.loads(metrics_path.read_text()), "final": info}
    )
    _write_json(out / "summary.json", info)
    return info


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument(
        "--init-ckpt",
        type=Path,
        default=None,
        help=(
            "Optional warm-start. Default is from scratch. "
            "Without --init-full, residual_gate is left uncopied."
        ),
    )
    p.add_argument(
        "--init-full",
        action="store_true",
        help=(
            "With --init-ckpt, copy every matching tensor including "
            "residual_gate. Use for same-architecture fine-tune."
        ),
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Continue from <out>/last.pt (weights, epoch, plateau LR, "
            "stale count). Keeps best*.pt and epochs.jsonl. "
            "Overrides --init-ckpt when last.pt exists."
        ),
    )
    p.add_argument(
        "--fc-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Add λ·mean(relu(T̂−thresh)) on cloudy tiles at every "
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
    p.add_argument("--dropout", type=float, default=0.3)
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
            "all-night 40/night cap. 0 (default) is the paper multihorizon keep. "
            "Does not change val or write keep files. On a multihorizon cache "
            "the band is any lead whose mean observed transmission is in range, not +20 min only."
        ),
    )
    p.add_argument("--band-t-min", type=float, default=0.1)
    p.add_argument("--band-t-max", type=float, default=0.9)
    p.add_argument(
        "--ablate",
        default="",
        help=(
            "Zero inputs at train and eval (same architecture). "
            "Comma list: sat, wind, meta, dream. Aliases: multimodal "
            "(=sat,wind), camera (=sat,wind,meta), sat-only (=dream)."
        ),
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
    try:
        ablate_set = parse_ablate(args.ablate)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    out = Path(args.out) if args.out is not None else DEFAULT_OUT / f"seed_{args.seed}" / "model"
    cache_dir = Path(args.cache_dir)
    keep_dir = Path(args.keep_dir)
    if keep_dir.name != "keep_train_nights_val20":
        raise SystemExit(f"keep_dir must be keep_train_nights_val20, got {keep_dir}")
    if not (keep_dir / "validation.npy").is_file():
        print(f"  carving inner val → {keep_dir}", flush=True)
        carve_inner_val(out_dir=keep_dir)
    out.mkdir(parents=True, exist_ok=True)
    curriculum = make_curriculum_record(args, trainer="multihorizon")
    write_curriculum_record(out, curriculum)
    print(format_curriculum_line(curriculum), flush=True)

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        n_frames=args.n_frames,
        train_frac=args.train_frac,
        seed=args.seed,
        lr=args.lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        patience=args.patience,
        max_lr_halves=args.max_lr_halves,
        device=args.device,
        verbose=True,
        d_model=64,
        n_heads=4,
        n_layers=2,
        sky_width=16,
    )
    seed_everything(cfg.seed)
    pairs_dir = resolve_pairs(args.pairs)
    init_note = str(args.init_ckpt) if args.init_ckpt is not None else "scratch"
    print(
        f"multihorizon forecaster  leads={list(HORIZONS_SEC)}  "
        f"loss={LOSS_NAME}  init={init_note}  → {out}",
        flush=True,
    )
    data, _val_unused, test_data, norm, nside, device, cache_root = load_data(
        cfg,
        pairs_dir,
        cache_dir,
        score_test=False,
        val_from_train=True,
    )
    cfg.wind_age_mean = float(norm.wind[-1].mean)
    cfg.wind_age_std = float(norm.wind[-1].std)
    attach_wind_decode_stats(cfg, norm)
    print(
        f"  wind pierce denorm  u={cfg.wind_u_med:.3f}/{cfg.wind_u_iqr:.3f}  "
        f"v={cfg.wind_v_med:.3f}/{cfg.wind_v_iqr:.3f}  "
        f"h={cfg.wind_h_med:.3f}/{cfg.wind_h_iqr:.3f}  (cache median/IQR)",
        flush=True,
    )
    if "sat" in ablate_set:
        print("  ablate sat  skip rich-sat overlay (zeros at train/eval)", flush=True)
    if "dream" in ablate_set:
        print(
            "  ablate dream  zero camera history; persist prior T=1 "
            "(sat/wind/meta stay)",
            flush=True,
        )
    if "sat" not in ablate_set:
        data, _val_unused, test_data = attach_rich_sat(
            data,
            data,
            test_data,
            pairs_dir,
            cache_root,
            cfg,
            val_from_train=True,
        )
    mh = wrap_multihorizon(data, cache_root / "train")
    mh = attach_horizon_lock(mh, cache_root / "train")

    model = cand.build_model(nside, cfg)
    model.set_ablate(ablate_set)
    last_pt = out / "last.pt"
    skip_init = bool(args.resume) and last_pt.is_file()
    if skip_init:
        print(f"  resume {last_pt}  skip --init-ckpt", flush=True)
    elif args.init_ckpt is not None:
        payload = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
        loader = cand.load_matching if args.init_full else cand.load_backbone
        missing, unexpected = loader(model, payload["state_dict"])
        print(
            f"  init {args.init_ckpt}  full={bool(args.init_full)}  "
            f"missing={len(missing)}  unexpected={len(unexpected)}",
            flush=True,
        )
    else:
        print("  init from scratch (random weights)", flush=True)
    if ablate_set:
        print(f"  ablate  {','.join(sorted(ablate_set))}  (train+eval)", flush=True)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    source = json.loads(SOURCE_PATH.read_text()) if SOURCE_PATH.is_file() else {}
    n_train_keep = int(np.load(keep_dir / "train.npy").size)
    n_val_keep = int(np.load(keep_dir / "validation.npy").size)
    _write_json(
        out / "run.json",
        {
            "source": source,
            "started": dt.datetime.now(dt.timezone.utc).isoformat(),
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
            "ablate": sorted(ablate_set),
            "curriculum": curriculum,
            "device": str(device),
            "config": {
                "epochs": cfg.epochs,
                "batch_size": cfg.batch_size,
                "lr": cfg.lr,
                "min_lr": cfg.min_lr,
                "weight_decay": cfg.weight_decay,
                "dropout": cfg.dropout,
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
                "ablate": sorted(ablate_set),
                "wind_u_med": float(getattr(cfg, "wind_u_med", 0.0)),
                "wind_u_iqr": float(getattr(cfg, "wind_u_iqr", 1.0)),
                "wind_v_med": float(getattr(cfg, "wind_v_med", 0.0)),
                "wind_v_iqr": float(getattr(cfg, "wind_v_iqr", 1.0)),
                "wind_h_med": float(getattr(cfg, "wind_h_med", 0.0)),
                "wind_h_iqr": float(getattr(cfg, "wind_h_iqr", 1.0)),
            },
        },
    )
    print(
        f"training  nside={nside}  train={n_train_keep}  val={n_val_keep}  "
        f"params={n_params:,}  device={device}",
        flush=True,
    )
    if float(args.band_mix) > 0:
        print(
            f"  train band-mix  frac={float(args.band_mix):g}  "
            f"T∈[{float(args.band_t_min):g}, {float(args.band_t_max):g}]  "
            f"(val unchanged)",
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
        run="paper_multih_t24",
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
    print(json.dumps(info, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
