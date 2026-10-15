"""Minimal torch helpers shared by trained models (not baselines)."""

from __future__ import annotations

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed RNGs. Call before model construction so init is reproducible."""
    import os

    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


def resolve_device(device: str | None = None) -> "torch.device":
    """Pick CUDA when available unless ``device`` is set explicitly."""
    import torch

    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
