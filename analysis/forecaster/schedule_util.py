"""Fixed cosine schedule + EMA. No plateau / best-checkpoint hunting."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def cosine_lr(epoch: int, *, base: float, min_lr: float, warmup: int, total: int) -> float:
    if epoch < int(warmup):
        return float(base) * float(epoch + 1) / max(int(warmup), 1)
    t = (epoch - int(warmup)) / max(int(total) - int(warmup), 1)
    t = min(max(t, 0.0), 1.0)
    return float(min_lr) + 0.5 * (float(base) - float(min_lr)) * (1.0 + math.cos(math.pi * t))


class ModelEma:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = float(decay)
        self.shadow = {
            k: v.detach().clone() for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if k not in self.shadow or self.shadow[k].shape != v.shape:
                self.shadow[k] = v.detach().clone()
                continue
            if not v.is_floating_point():
                self.shadow[k].copy_(v)
                continue
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=False)
