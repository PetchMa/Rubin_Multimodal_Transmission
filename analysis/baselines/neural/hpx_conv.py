"""DeepSphere geometry for the neural baselines.

Same Chebyshev graph convolution as the forecaster: filters live on
the nested HEALPix neighbour graph (geodesic edge weights), not on a 2D
face raster. Nested 4-child pool / upsample is the HEALPix hierarchy
(nside → nside/2), not a Euclidean image pyramid.

The Laplacian apply uses a cached CSR ``torch.sparse.mm`` path that is
mathematically identical to the previous ``index_add_`` scatter
(``y[src] += w * x[dst]``), including autograd.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from analysis.healpix_torch import (
    apply_chebyshev_laplacian,
    chebyshev_basis,
    chebyshev_laplacian_csr,
    neighbour_graph,
)


class ChebConv(nn.Module):
    """Chebyshev spectral graph convolution of order ``K``.

    ``y = sum_k theta_k T_k(L~) x`` on the nested HEALPix neighbour graph.
    Rotation-equivariant in the DeepSphere sense.
    """

    def __init__(
        self,
        nside: int,
        in_ch: int,
        out_ch: int,
        K: int = 3,
        *,
        bias: bool = True,
    ) -> None:
        super().__init__()
        edge_index, edge_weight = neighbour_graph(nside)
        self.nside = int(nside)
        self.K = int(K)
        # Kept for checkpoint compatibility with the pre-CSR implementation.
        self.register_buffer("src", torch.as_tensor(edge_index[0]))
        self.register_buffer("dst", torch.as_tensor(edge_index[1]))
        self.register_buffer("w", torch.as_tensor(edge_weight, dtype=torch.float32))
        self.lin = nn.Linear(self.K * in_ch, out_ch, bias=bias)

    def _laplacian(self, x: torch.Tensor) -> torch.Tensor:
        lap = chebyshev_laplacian_csr(self.nside, device=x.device, dtype=x.dtype)
        return apply_chebyshev_laplacian(x, lap)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x: (B, npix, C) → (B, npix, C_out)``."""
        lap = chebyshev_laplacian_csr(self.nside, device=x.device, dtype=x.dtype)
        return self.lin(torch.cat(chebyshev_basis(x, lap, self.K), dim=-1))


def nest_pool(x: torch.Tensor) -> torch.Tensor:
    """Average the four nested children. ``(B, N, C) → (B, N/4, C)``."""
    b, n, c = x.shape
    if n % 4 != 0:
        raise ValueError(f"npix={n} is not a nested parent of nside/2")
    return x.reshape(b, n // 4, 4, c).mean(dim=2)


def nest_upsample(x: torch.Tensor) -> torch.Tensor:
    """Copy each coarse tile onto its four nested children."""
    b, n, c = x.shape
    return x.unsqueeze(2).expand(b, n, 4, c).reshape(b, n * 4, c)
