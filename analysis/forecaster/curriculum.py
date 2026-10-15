"""Two-stage scratch-to-hinge curriculum for the proposed multihorizon model.

Do not apply the false-clear hinge from epoch 0, and do not ramp λ. The
paper-safe recipe is an explicit curriculum:

1. **calibrate** — train from scratch with airmass-weighted yield MSE,
   averaged over present leads. Plateau to the best validation-yield
   checkpoint (``best_20min.pt``). Default LR ``1e-4``.
2. **risk** — restore that *full* checkpoint (``--init-full``, including
   ``residual_gate``), reset AdamW, drop LR to ``1e-5``, and set
   ``λ_FC=0.01`` immediately for a 25-epoch fine-tune.

Both stages keep two validation-selected outputs: minimum yield, and
minimum η_FC under the predeclared ``1.03×`` yield cap.

Freeze ``λ_FC=0.01``. Any schedule comparison must be selected on inner
validation only, never the paper holdout.
"""

from __future__ import annotations

import json
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

from analysis.our_model.mask_train import FALSE_CLEAR_T

NAME = "scratch_yield_then_fc_hinge"
STAGES = ("calibrate", "risk")

CALIBRATE_LR = 1e-4
RISK_LR = 1e-5
FC_LOSS_WEIGHT = 0.01
FC_LOSS_THRESH = float(FALSE_CLEAR_T)
FC_YIELD_SLACK = 1.03
RISK_EPOCHS = 25
RISK_PATIENCE = 8
RISK_MAX_LR_HALVES = 2
RISK_WARMUP_EPOCHS = 0

T24_YIELD_CKPT = "best.pt"
MH_YIELD_CKPT = "best_20min.pt"
MH_JOINT_CKPT = "best.pt"
MH_FC_CKPT = "best_20min_fc.pt"

_L_FC = (
    "mean(relu(T_hat - thresh)) on gold-cloud, valid-last, altitude>=30 deg tiles"
)


def add_curriculum_args(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--curriculum-stage",
        choices=STAGES,
        default=None,
        help=(
            "calibrate: yield MSE only (λ_FC=0, lr=1e-4). "
            "risk: restore --init-ckpt with --init-full, λ_FC=0.01, "
            "lr=1e-5, 25 epochs. Unset keeps legacy flag defaults."
        ),
    )


def flag_present(argv: list[str] | None, *flags: str) -> bool:
    if not argv:
        return False
    for raw in argv:
        for flag in flags:
            if raw == flag or raw.startswith(flag + "="):
                return True
    return False


def infer_stage(args: Namespace) -> str:
    stage = getattr(args, "curriculum_stage", None)
    if stage in STAGES:
        return str(stage)
    if float(getattr(args, "fc_loss_weight", 0.0) or 0.0) > 0:
        return "risk"
    return "calibrate"


def apply_curriculum_stage(
    parser: ArgumentParser,
    args: Namespace,
    *,
    argv: list[str] | None,
    trainer: str,
) -> Namespace:
    """Fill risk/calibrate defaults when ``--curriculum-stage`` is set."""
    stage = getattr(args, "curriculum_stage", None)
    if stage is None:
        return args
    if stage not in STAGES:
        parser.error(f"unknown --curriculum-stage {stage!r}")

    if stage == "calibrate":
        if float(args.fc_loss_weight) > 0:
            parser.error(
                "--curriculum-stage calibrate forbids --fc-loss-weight > 0; "
                "the hinge is stage risk only"
            )
        if bool(getattr(args, "init_full", False)):
            parser.error(
                "--curriculum-stage calibrate forbids --init-full; "
                "residual_gate is copied only in stage risk"
            )
        args.fc_loss_weight = 0.0
        if not flag_present(argv, "--lr"):
            args.lr = CALIBRATE_LR
        return args

    if args.init_ckpt is None and not bool(getattr(args, "resume", False)):
        parser.error(
            "--curriculum-stage risk requires --init-ckpt "
            "(the calibration-stage yield checkpoint)"
        )
    if flag_present(argv, "--fc-loss-weight") and float(args.fc_loss_weight) <= 0:
        parser.error("--curriculum-stage risk requires --fc-loss-weight > 0")
    if not flag_present(argv, "--fc-loss-weight"):
        args.fc_loss_weight = FC_LOSS_WEIGHT
    if not flag_present(argv, "--fc-loss-thresh"):
        args.fc_loss_thresh = FC_LOSS_THRESH
    if not flag_present(argv, "--fc-yield-slack"):
        args.fc_yield_slack = FC_YIELD_SLACK
    if not flag_present(argv, "--lr"):
        args.lr = RISK_LR
    if not flag_present(argv, "--epochs"):
        args.epochs = RISK_EPOCHS
    if not flag_present(argv, "--patience"):
        args.patience = RISK_PATIENCE
    if not flag_present(argv, "--max-lr-halves"):
        args.max_lr_halves = RISK_MAX_LR_HALVES
    if not flag_present(argv, "--warmup-epochs"):
        args.warmup_epochs = RISK_WARMUP_EPOCHS
    if args.init_ckpt is not None:
        args.init_full = True
    if trainer != "multihorizon":
        parser.error("curriculum trainer must be multihorizon")
    return args


def parent_checkpoint(calibrate_out: Path, trainer: str = "multihorizon") -> Path:
    """Yield-selected calibration checkpoint that stage risk restores."""
    del trainer
    out = Path(calibrate_out)
    path = out / MH_YIELD_CKPT
    if path.is_file():
        return path
    fallback = out / T24_YIELD_CKPT
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(
        f"no calibration yield checkpoint in {out} "
        f"(looked for {MH_YIELD_CKPT} and {T24_YIELD_CKPT})"
    )


def make_record(args: Namespace, *, trainer: str = "multihorizon") -> dict[str, Any]:
    stage = infer_stage(args)
    lam = float(getattr(args, "fc_loss_weight", 0.0) or 0.0)
    parent = getattr(args, "init_ckpt", None)
    slack = float(getattr(args, "fc_yield_slack", FC_YIELD_SLACK))
    return {
        "name": NAME,
        "stage": stage,
        "trainer": "multihorizon",
        "parent_checkpoint": None if parent is None else str(parent),
        "init_full": bool(getattr(args, "init_full", False)),
        "fc_loss_weight": lam,
        "fc_loss_thresh": float(
            getattr(args, "fc_loss_thresh", FC_LOSS_THRESH) or FC_LOSS_THRESH
        ),
        "fc_yield_slack": slack,
        "lr": float(args.lr),
        "min_lr": float(getattr(args, "min_lr", 1e-6)),
        "epochs": int(args.epochs),
        "patience": int(args.patience),
        "max_lr_halves": int(args.max_lr_halves),
        "warmup_epochs": int(getattr(args, "warmup_epochs", 0)),
        "objective": "L_yield + lambda_FC * L_FC" if lam > 0 else "L_yield",
        "L_FC": _L_FC,
        "yield_ckpt": MH_YIELD_CKPT,
        "joint_ckpt": MH_JOINT_CKPT,
        "fc_ckpt": MH_FC_CKPT if lam > 0 else None,
        "yield_selection": "min +20 min validation yield MSE",
        "fc_selection": (
            None
            if lam <= 0
            else (
                f"min validation eta_FC subject to yield MSE "
                f"<= {slack:g} x init-checkpoint yield"
            )
        ),
    }


def write_record(out: Path, record: dict[str, Any]) -> Path:
    path = Path(out) / "curriculum.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(record, indent=2) + "\n")
    tmp.replace(path)
    return path


def format_line(record: dict[str, Any]) -> str:
    parent = record.get("parent_checkpoint") or "none"
    lam = float(record.get("fc_loss_weight") or 0.0)
    return (
        f"curriculum  name={record.get('name')}  "
        f"stage={record.get('stage')}  trainer={record.get('trainer')}  "
        f"λ={lam:g}  thresh={record.get('fc_loss_thresh')}  "
        f"lr={record.get('lr')}  cap={record.get('fc_yield_slack')}  "
        f"parent={parent}  init_full={record.get('init_full')}"
    )


def strip_flags(argv: list[str], *flags: str) -> list[str]:
    """Drop ``--flag`` / ``--flag=value`` so a later stage can set them."""
    out: list[str] = []
    skip = False
    for raw in argv:
        if skip:
            skip = False
            continue
        dropped = False
        for flag in flags:
            if raw == flag:
                skip = True
                dropped = True
                break
            if raw.startswith(flag + "="):
                dropped = True
                break
        if not dropped:
            out.append(raw)
    return out


def _self_check() -> None:
    parser = ArgumentParser()
    parser.add_argument("--init-ckpt", type=Path, default=None)
    parser.add_argument("--init-full", action="store_true")
    parser.add_argument("--fc-loss-weight", type=float, default=0.0)
    parser.add_argument("--fc-loss-thresh", type=float, default=FALSE_CLEAR_T)
    parser.add_argument("--fc-yield-slack", type=float, default=1.03)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--max-lr-halves", type=int, default=4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--resume", action="store_true")
    add_curriculum_args(parser)

    cal_argv = ["--curriculum-stage", "calibrate"]
    cal = apply_curriculum_stage(
        parser, parser.parse_args(cal_argv), argv=cal_argv, trainer="multihorizon"
    )
    assert cal.fc_loss_weight == 0.0
    assert abs(float(cal.lr) - CALIBRATE_LR) < 1e-12
    rec = make_record(cal, trainer="multihorizon")
    assert rec["stage"] == "calibrate"
    assert rec["fc_ckpt"] is None
    assert rec["parent_checkpoint"] is None

    risk_argv = [
        "--curriculum-stage",
        "risk",
        "--init-ckpt",
        "/tmp/best_20min.pt",
    ]
    risk = apply_curriculum_stage(
        parser,
        parser.parse_args(risk_argv),
        argv=risk_argv,
        trainer="multihorizon",
    )
    assert risk.init_full is True
    assert abs(float(risk.fc_loss_weight) - FC_LOSS_WEIGHT) < 1e-12
    assert abs(float(risk.lr) - RISK_LR) < 1e-12
    assert int(risk.epochs) == RISK_EPOCHS
    assert int(risk.patience) == RISK_PATIENCE
    assert int(risk.max_lr_halves) == RISK_MAX_LR_HALVES
    rec_r = make_record(risk, trainer="multihorizon")
    assert rec_r["stage"] == "risk"
    assert rec_r["fc_ckpt"] == MH_FC_CKPT
    assert rec_r["parent_checkpoint"] == "/tmp/best_20min.pt"

    resume_argv = ["--curriculum-stage", "risk", "--resume"]
    resumed = apply_curriculum_stage(
        parser,
        parser.parse_args(resume_argv),
        argv=resume_argv,
        trainer="multihorizon",
    )
    assert abs(float(resumed.fc_loss_weight) - FC_LOSS_WEIGHT) < 1e-12
    assert resumed.init_ckpt is None

    try:
        bad = ["--curriculum-stage", "calibrate", "--fc-loss-weight", "0.01"]
        apply_curriculum_stage(
            parser, parser.parse_args(bad), argv=bad, trainer="multihorizon"
        )
    except SystemExit:
        pass
    else:
        raise AssertionError("calibrate + hinge should fail")

    stripped = strip_flags(
        ["--out", "a", "--init-ckpt", "x", "--seed", "42"],
        "--out",
        "--init-ckpt",
    )
    assert stripped == ["--seed", "42"]
    print("curriculum self-check ok")


if __name__ == "__main__":
    _self_check()
