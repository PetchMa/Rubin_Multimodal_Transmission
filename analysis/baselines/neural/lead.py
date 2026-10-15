"""Centered lead embedding shared by SimVP and ConvLSTM.

Same ``e(h)=MLP(φ(h))−MLP(φ(20 min))`` as forecaster. Last layer is zero, so
every lead is an identity residual on the shared trunk until the MLP learns.
``+20 min`` stays the unconditioned decode by construction.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from analysis.forecaster.geom import LEAD_SEC
from analysis.forecaster.program import LeadQueryMLP

LEAD_SKIP_PREFIXES: tuple[str, ...] = ("net.lead.", "net.lead_query.")


class CenteredLead(nn.Module):
    """Add a centered lead delta to a ``(B, npix, C)`` feature map."""

    def __init__(self, dim: int, *, lead_ref_sec: float = LEAD_SEC) -> None:
        super().__init__()
        self.dim = int(dim)
        self.lead_ref_sec = float(lead_ref_sec)
        self.lead_query = LeadQueryMLP(self.dim)

    def as_lead(
        self, lead_sec, *, batch: int, ref: torch.Tensor
    ) -> torch.Tensor:
        if lead_sec is None:
            return ref.new_full((batch,), float(self.lead_ref_sec))
        lead = torch.as_tensor(lead_sec, device=ref.device, dtype=ref.dtype).reshape(-1)
        if int(lead.numel()) == 1:
            lead = lead.expand(batch)
        if int(lead.numel()) != int(batch):
            raise ValueError(f"lead_sec n={int(lead.numel())} != batch {batch}")
        return lead.clamp(min=1.0)

    def feat(self, lead: torch.Tensor) -> torch.Tensor:
        ref = float(self.lead_ref_sec)
        ratio = (lead / ref).clamp(min=1e-3, max=8.0)
        return torch.stack([ratio.log(), ratio], dim=-1)

    def delta(self, lead_sec, *, batch: int, ref: torch.Tensor) -> torch.Tensor:
        lead = self.as_lead(lead_sec, batch=batch, ref=ref)
        feat = self.feat(lead)
        ref_feat = self.feat(lead.new_full(lead.shape, float(self.lead_ref_sec)))
        return self.lead_query(feat) - self.lead_query(ref_feat)

    def add_to(self, z: torch.Tensor, lead_sec) -> torch.Tensor:
        """``z: (B, npix, C)``."""
        d = self.delta(lead_sec, batch=int(z.shape[0]), ref=z)
        return z + d.view(z.shape[0], 1, -1)
