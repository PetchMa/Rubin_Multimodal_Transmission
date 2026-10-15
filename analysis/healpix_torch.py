"""HEALPix geometry primitives for spherical CNNs.

- **Faces.** Nested HEALPix splits into 12 square faces of side ``nside``.
- **Graph.** Pixel adjacency with a normalized Laplacian (DeepSphere / Chebyshev).

Both are built once and cached; they depend only on ``nside``.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np


def _morton_decode(j: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split interleaved Morton (z-order) bits into ``(x, y)`` face coordinates.

    Within a face, the nested index is the Morton code of the pixel's position,
    with x in the even bits and y in the odd bits.
    """
    x = np.zeros_like(j)
    y = np.zeros_like(j)
    for b in range(32):
        x |= ((j >> (2 * b)) & 1) << b
        y |= ((j >> (2 * b + 1)) & 1) << b
    return x, y


@lru_cache(maxsize=8)
def face_index(nside: int) -> np.ndarray:
    """Map ``(face, y, x) -> nested pixel index``, shape ``(12, nside, nside)``.

    Nested ordering guarantees ``ipix = face * nside**2 + morton(x, y)``, so
    this is an exact rearrangement with no interpolation.
    """
    npix_face = nside * nside
    ipix = np.arange(12 * npix_face, dtype=np.int64)
    face = ipix // npix_face
    x, y = _morton_decode(ipix % npix_face)

    idx = np.empty((12, nside, nside), dtype=np.int64)
    idx[face, y, x] = ipix
    return idx


@lru_cache(maxsize=8)
def padded_face_index(nside: int, pad: int = 1) -> np.ndarray:
    """Gather indices for faces padded with real neighbouring sky.

    Returns shape ``(12, nside + 2*pad, nside + 2*pad)`` of nested pixel
    indices. Interior entries are the face's own pixels; border entries point at
    genuine neighbouring pixels across the face boundary, so a convolution never
    sees a zero-padded or wrapped-around edge.

    Border positions are located by linearly extrapolating the 3D positions of
    the two nearest in-face pixels, renormalizing onto the unit sphere, and
    taking the HEALPix pixel containing that direction. This is approximate at
    the corners, where the sphere's curvature makes the extrapolation ambiguous,
    but it is geometrically grounded everywhere and avoids the flat-image
    artifacts that motivated the constraint in ``NORTH_STAR.md``.
    """
    import healpy as hp

    base = face_index(nside)
    n = nside + 2 * pad
    out = np.empty((12, n, n), dtype=np.int64)
    out[:, pad : pad + nside, pad : pad + nside] = base

    vec = np.stack(hp.pix2vec(nside, np.arange(12 * nside * nside), nest=True), axis=1)

    def in_face(iy: int, ix: int) -> tuple[int, int]:
        return int(np.clip(iy, 0, nside - 1)), int(np.clip(ix, 0, nside - 1))

    for f in range(12):
        for yy in range(n):
            for xx in range(n):
                if pad <= yy < pad + nside and pad <= xx < pad + nside:
                    continue
                # Face coordinates of this padded cell, possibly outside [0, nside).
                fy, fx = yy - pad, xx - pad
                ay, ax = in_face(fy, fx)
                # Step back toward the face interior to get a second anchor.
                by = ay + (1 if fy < 0 else (-1 if fy > nside - 1 else 0))
                bx = ax + (1 if fx < 0 else (-1 if fx > nside - 1 else 0))
                by, bx = in_face(by, bx)

                va = vec[base[f, ay, ax]]
                vb = vec[base[f, by, bx]]
                steps = max(abs(fy - ay), abs(fx - ax), 1)
                v = va + steps * (va - vb)
                norm = np.linalg.norm(v)
                if norm < 1e-12:
                    out[f, yy, xx] = base[f, ay, ax]
                else:
                    out[f, yy, xx] = hp.vec2pix(nside, *(v / norm), nest=True)
    return out


@lru_cache(maxsize=8)
def neighbour_graph(nside: int) -> tuple[np.ndarray, np.ndarray]:
    """Sparse normalized graph Laplacian over HEALPix pixels.

    Returns ``(edge_index, edge_weight)`` for the symmetric normalized Laplacian
    ``L = I - D^-1/2 A D^-1/2``, rescaled to spectral range ``[-1, 1]`` as
    Chebyshev convolution requires. Edge weights follow DeepSphere: a Gaussian
    of the geodesic (chord) distance between adjacent pixel centres.
    """
    import healpy as hp

    npix = 12 * nside * nside
    pix = np.arange(npix)
    nb = hp.get_all_neighbours(nside, pix, nest=True)  # (8, npix)

    src = np.repeat(pix, 8)
    dst = nb.T.reshape(-1)
    ok = dst >= 0
    src, dst = src[ok], dst[ok]

    vec = np.stack(hp.pix2vec(nside, pix, nest=True), axis=1)
    d2 = np.sum((vec[src] - vec[dst]) ** 2, axis=1)
    w = np.exp(-d2 / (2.0 * np.mean(d2)))

    # Symmetrize, since get_all_neighbours is not guaranteed reciprocal.
    key = np.minimum(src, dst) * npix + np.maximum(src, dst)
    _, first = np.unique(key, return_index=True)
    s, d, ww = src[first], dst[first], w[first]
    src = np.concatenate([s, d])
    dst = np.concatenate([d, s])
    w = np.concatenate([ww, ww])

    deg = np.zeros(npix)
    np.add.at(deg, src, w)
    dinv = 1.0 / np.sqrt(np.maximum(deg, 1e-12))

    # L = I - D^-1/2 A D^-1/2. Chebyshev convolution needs the spectrum
    # rescaled to [-1, 1] as L~ = 2L/lmax - I, and lmax <= 2 for a normalized
    # Laplacian, so L~ = -D^-1/2 A D^-1/2: the identity cancels and the
    # diagonal vanishes, leaving only the off-diagonal edges below.
    edge_weight = -w * dinv[src] * dinv[dst]
    return np.stack([src, dst]), edge_weight


# Device-local CSR cache for ``L~``. Same operator as the scatter form of
# ``neighbour_graph``; cuSPARSE SpMM is much faster than index_add_ for training.
_CSR_LAPLACIAN_CACHE: dict[tuple[int, str, int, str], "torch.Tensor"] = {}


def chebyshev_laplacian_csr(
    nside: int,
    *,
    device: "torch.device | str",
    dtype: "torch.dtype | None" = None,
):
    """Cached sparse CSR ``L~`` for Chebyshev graph convolution.

    Mathematically identical to applying the ``(edge_index, edge_weight)``
    scatter from :func:`neighbour_graph` (``y[src] += w * x[dst]``).
    """
    import torch

    device = torch.device(device)
    if dtype is None:
        dtype = torch.float32
    index = device.index
    if device.type == "cuda" and index is None:
        index = torch.cuda.current_device()
    if index is None:
        index = -1
    key = (int(nside), device.type, int(index), str(dtype))
    cached = _CSR_LAPLACIAN_CACHE.get(key)
    if cached is not None:
        return cached

    edge_index, edge_weight = neighbour_graph(int(nside))
    npix = 12 * int(nside) * int(nside)
    idx = torch.as_tensor(edge_index, device=device, dtype=torch.long)
    vals = torch.as_tensor(edge_weight, device=device, dtype=dtype)
    lap = (
        torch.sparse_coo_tensor(idx, vals, (npix, npix)).coalesce().to_sparse_csr()
    )
    _CSR_LAPLACIAN_CACHE[key] = lap
    return lap


def apply_chebyshev_laplacian(x: "torch.Tensor", lap: "torch.Tensor") -> "torch.Tensor":
    """``y = L~ x`` for ``x`` shaped ``(B, npix, C)``.

    ``cusparseSpMM`` does not mix a fp32 CSR Laplacian with bf16/fp16
    features, so the multiply always runs in the Laplacian dtype.
    """
    import torch

    b, n, c = x.shape
    compute = lap.dtype
    src = x.dtype
    with torch.autocast(device_type=x.device.type, enabled=False):
        flat = x.to(dtype=compute).transpose(0, 1).reshape(n, b * c).contiguous()
        out = torch.sparse.mm(lap, flat)
    return out.reshape(n, b, c).transpose(0, 1).contiguous().to(dtype=src)


def chebyshev_basis(
    x: "torch.Tensor", lap: "torch.Tensor", K: int
) -> list:
    """Chebyshev polynomials ``T_0(L~)x … T_{K-1}(L~)x`` (same recurrence as ChebConv)."""
    K = int(K)
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")
    terms = [x]
    if K > 1:
        terms.append(apply_chebyshev_laplacian(x, lap))
    for _ in range(2, K):
        terms.append(
            2.0 * apply_chebyshev_laplacian(terms[-1], lap) - terms[-2]
        )
    return terms


def to_faces(x, nside: int):
    """``(..., npix) -> (..., 12, nside, nside)`` for torch tensors."""
    import torch

    idx = torch.as_tensor(face_index(nside), device=x.device).reshape(-1)
    return x[..., idx].reshape(*x.shape[:-1], 12, nside, nside)


def from_faces(x, nside: int):
    """``(..., 12, nside, nside) -> (..., npix)``, inverse of :func:`to_faces`."""
    import torch

    idx = torch.as_tensor(face_index(nside), device=x.device).reshape(-1)
    flat = x.reshape(*x.shape[:-3], 12 * nside * nside)
    out = torch.empty_like(flat)
    out[..., idx] = flat
    return out


def nside_from_npix(npix: int) -> int:
    nside = int(round((float(npix) / 12.0) ** 0.5))
    if 12 * nside * nside != int(npix):
        raise ValueError(f"npix={npix} is not 12·nside²")
    return nside


@lru_cache(maxsize=16)
def _gauss_kernel_1d(sigma: float, truncate: float) -> np.ndarray:
    radius = int(np.ceil(float(truncate) * float(sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / float(sigma)) ** 2)
    k /= k.sum()
    return k


def gaussian_smooth_healpix(
    field: "torch.Tensor",
    valid: "torch.Tensor | None" = None,
    *,
    nside: int | None = None,
    sigma: float,
    truncate: float = 3.0,
) -> "torch.Tensor":
    """Masked Gaussian on the 12 nested HEALPix faces.

    ``sigma`` is in face-grid pixels. Invalid tiles have zero weight so they
    do not bleed into neighbours. The original valid mask is not changed;
    this only fills a smoothed value on every tile.
    """
    import torch
    import torch.nn.functional as F

    if float(sigma) <= 0.0:
        return field
    npix = int(field.shape[-1])
    nside = int(nside) if nside is not None else nside_from_npix(npix)
    if 12 * nside * nside != npix:
        raise ValueError(f"npix={npix} is not 12·nside² (nside={nside})")

    k1d = _gauss_kernel_1d(float(sigma), float(truncate))
    radius = int((k1d.size - 1) // 2)
    pad = max(radius, 1)
    idx = torch.as_tensor(padded_face_index(nside, pad=pad), device=field.device)
    kernel = torch.as_tensor(k1d, device=field.device, dtype=torch.float32)
    kh = kernel.view(1, 1, -1, 1)
    kw = kernel.view(1, 1, 1, -1)

    leading = field.shape[:-1]
    flat = field.reshape(-1, npix)
    if valid is None:
        vflat = torch.ones_like(flat)
    else:
        vflat = valid.to(dtype=field.dtype).reshape(-1, npix).expand_as(flat)

    side = nside + 2 * pad
    gathered = flat[:, idx]
    weight = vflat[:, idx]
    b12 = flat.shape[0] * 12
    num = (gathered * weight).to(dtype=torch.float32).reshape(b12, 1, side, side)
    den = weight.to(dtype=torch.float32).reshape(b12, 1, side, side)
    num = F.conv2d(F.conv2d(num, kh), kw)
    den = F.conv2d(F.conv2d(den, kh), kw)
    # pad == radius → conv output is exactly nside × nside.
    if num.shape[-1] != nside:
        c0 = (num.shape[-1] - nside) // 2
        num = num[..., c0 : c0 + nside, c0 : c0 + nside]
        den = den[..., c0 : c0 + nside, c0 : c0 + nside]
    sm = (num / den.clamp_min(1e-6)).reshape(flat.shape[0], 12, nside, nside)
    out = from_faces(sm, nside).to(dtype=field.dtype)
    return out.reshape(*leading, npix)
