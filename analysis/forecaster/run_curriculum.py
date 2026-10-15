"""Run the two-stage scratch-to-hinge curriculum for the proposed model.

The proposed network is multihorizon. Stage 1 writes ``<out>/calibrate``
(yield MSE, from scratch). Stage 2 restores that full checkpoint into
``<out>/risk`` (λ_FC=0.01, lr=1e-5). Remaining flags are forwarded.

    python -m analysis.forecaster.run_curriculum \\
        --out /path/to/seed_42 --seed 42 --band-mix 0.5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from analysis.forecaster.curriculum import (
    NAME,
    parent_checkpoint,
    strip_flags,
)
from analysis.forecaster import train_multihorizon as trainer


def _stage_done(out: Path) -> bool:
    return (out / "summary.json").is_file()


def _maybe_resume(out: Path, argv: list[str]) -> list[str]:
    if (out / "last.pt").is_file() and "--resume" not in argv:
        argv = strip_flags(argv, "--init-ckpt", "--init-full")
        return argv + ["--resume"]
    return argv


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Parent directory. Writes <out>/calibrate then <out>/risk.",
    )
    p.add_argument("--calibrate-out", type=Path, default=None)
    p.add_argument("--risk-out", type=Path, default=None)
    p.add_argument(
        "--skip-calibrate",
        action="store_true",
        help="Only run stage risk (requires an existing calibrate yield ckpt).",
    )
    p.add_argument(
        "--skip-risk",
        action="store_true",
        help="Only run stage calibrate.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-run a stage even if summary.json already exists.",
    )
    args, rest = p.parse_known_args(argv)

    cal_out = Path(args.calibrate_out or (Path(args.out) / "calibrate"))
    risk_out = Path(args.risk_out or (Path(args.out) / "risk"))
    rest = strip_flags(rest, "--out", "--curriculum-stage", "--trainer")
    # Calibrate is from scratch. Risk restores calibrate/best_20min.pt.
    cal_rest = strip_flags(
        rest,
        "--init-ckpt",
        "--init-full",
        "--fc-loss-weight",
        "--fc-loss-thresh",
        "--fc-yield-slack",
        "--resume",
    )
    risk_rest = strip_flags(
        rest,
        "--init-ckpt",
        "--init-full",
        "--resume",
        "--lr",
        "--epochs",
        "--patience",
        "--max-lr-halves",
        "--warmup-epochs",
        "--fc-loss-weight",
        "--fc-loss-thresh",
        "--fc-yield-slack",
    )

    print(
        f"run_curriculum  name={NAME}  trainer=multihorizon  "
        f"calibrate={cal_out}  risk={risk_out}",
        flush=True,
    )

    if not args.skip_calibrate:
        if _stage_done(cal_out) and not args.force:
            print(f"  skip calibrate (summary.json exists)  {cal_out}", flush=True)
        else:
            cal_cmd = cal_rest + [
                "--out",
                str(cal_out),
                "--curriculum-stage",
                "calibrate",
            ]
            cal_argv = cal_cmd if args.force else _maybe_resume(cal_out, cal_cmd)
            print(f"  stage calibrate  argv={cal_argv}", flush=True)
            trainer.main(cal_argv)
            if not _stage_done(cal_out):
                raise SystemExit(f"calibrate finished without summary.json in {cal_out}")
    elif not _stage_done(cal_out) and not (cal_out / "best.pt").is_file():
        raise SystemExit(f"--skip-calibrate needs an existing run at {cal_out}")

    if args.skip_risk:
        print("  skip risk (--skip-risk)", flush=True)
        return
    if _stage_done(risk_out) and not args.force:
        print(f"  skip risk (summary.json exists)  {risk_out}", flush=True)
        return

    parent = parent_checkpoint(cal_out, "multihorizon")
    risk_cmd = risk_rest + [
        "--out",
        str(risk_out),
        "--curriculum-stage",
        "risk",
        "--init-ckpt",
        str(parent),
        "--init-full",
    ]
    risk_argv = risk_cmd if args.force else _maybe_resume(risk_out, risk_cmd)
    print(f"  stage risk  parent={parent}  argv={risk_argv}", flush=True)
    trainer.main(risk_argv)
    if not _stage_done(risk_out):
        raise SystemExit(f"risk finished without summary.json in {risk_out}")


if __name__ == "__main__":
    main()
