"""Camera ↔ GOES geometry for the forecaster model.

Camera maps are gathered onto a horizon sphere at load (zenith = pole,
φ = 0 = north). Clouds and GOES live in local alt-az / east-north.
This module converts without a learned warp.

The sat cutout is a ±1.5° lat–lon box (native ABI 135×149, or a 48×48
bake): row 0 is north, col 0 is west.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F

from analysis.satellite.goes import SITE_LAT, SITE_LON

JD_UNIX_EPOCH = 2440587.5
KM_PER_DEG_LAT = 111.195
HALF_WIDTH_DEG = 1.5
HALF_KM = HALF_WIDTH_DEG * KM_PER_DEG_LAT
LEAD_SEC = 1200.0
MIN_ALT_DEG = 10.0
DECKS_KM = (2.5, 6.0, 10.0)

# Fallback ChannelNorm (median / IQR) if a caller does not pass cache stats.
# Prefer ``attach_wind_decode_stats`` from the memmap ``norm.wind`` used to
# encode the points. These numbers match ``dream_20min_encoded_t24``.
WIND_U_MED, WIND_U_IQR = 3.544867515563965, 15.922544986009598
WIND_V_MED, WIND_V_IQR = 1.9015477895736694, 12.357436656951904
WIND_H_MED, WIND_H_IQR = 5.9928483963012695, 7.8826568722724915


def attach_wind_decode_stats(cfg, wind_norm) -> None:
    """Copy cache ChannelNorm median/IQR (stored as mean/std) onto ``cfg``.

    ``wind_norm`` is ``BundleNorm`` or the ``norm.wind`` 6-tuple. Indices
    0, 1, 2 are u, v, height — the same channels ``decode_wind_uvh`` reads.
    """
    ch = wind_norm.wind if hasattr(wind_norm, "wind") else wind_norm
    for med_k, iqr_k, i in (
        ("wind_u_med", "wind_u_iqr", 0),
        ("wind_v_med", "wind_v_iqr", 1),
        ("wind_h_med", "wind_h_iqr", 2),
    ):
        setattr(cfg, med_k, float(ch[i].mean))
        setattr(cfg, iqr_k, float(max(float(ch[i].std), 1e-3)))


def _as_wind_scale(value, default: float, like: torch.Tensor) -> torch.Tensor:
    if value is None:
        return like.new_tensor(float(default))
    if torch.is_tensor(value):
        return value.to(device=like.device, dtype=like.dtype).reshape(())
    return like.new_tensor(float(value))


def unix_to_jd(unix: torch.Tensor) -> torch.Tensor:
    return unix.to(dtype=torch.float64) / 86400.0 + JD_UNIX_EPOCH


def lst_deg_from_unix(unix: torch.Tensor) -> torch.Tensor:
    """Local mean sidereal time [deg] from unix seconds. Vectorized, no astropy."""
    jd = unix_to_jd(unix)
    d = jd - 2451545.0
    gmst = 280.46061837 + 360.98564736629 * d
    return torch.remainder(gmst + SITE_LON, 360.0).to(dtype=torch.float32)


@lru_cache(maxsize=4)
def pix_radec(nside: int) -> tuple[np.ndarray, np.ndarray]:
    import healpy as hp

    theta, phi = hp.pix2ang(int(nside), np.arange(12 * nside * nside), nest=True)
    ra = np.rad2deg(phi).astype(np.float32)
    dec = (90.0 - np.rad2deg(theta)).astype(np.float32)
    return ra, dec


@lru_cache(maxsize=4)
def local_pix_altaz(nside: int) -> tuple[np.ndarray, np.ndarray]:
    """Nested HEALPix as a horizon sphere: pole = zenith, φ = 0 = north.

    ``alt`` is degrees above the horizon; ``az`` is degrees from north,
    east-positive, in ``[0, 360)``. Tile index is then a fixed sightline
    relative to the site, not a (RA, Dec).
    """
    import healpy as hp

    nside = int(nside)
    theta, phi = hp.pix2ang(nside, np.arange(12 * nside * nside), nest=True)
    alt = (90.0 - np.rad2deg(theta)).astype(np.float32)
    az = np.rad2deg(phi).astype(np.float32)
    return alt, az


def nside_from_npix(npix: int) -> int:
    nside2 = int(npix) // 12
    nside = int(round(nside2**0.5))
    if 12 * nside * nside != int(npix):
        raise ValueError(f"npix={npix} is not 12·nside²")
    return nside


def gather_equatorial_to_local(
    field: torch.Tensor,
    lst_deg: torch.Tensor,
    nside: int | None = None,
) -> torch.Tensor:
    """Nearest-neighbour reindex of an equatorial nested map onto local alt-az.

    ``field[..., npix]`` is RA/Dec HEALPix. The output uses the same ``nside``
    with pole = zenith and φ = 0 = north, so a given index is always the same
    horizon direction. ``lst_deg`` is the LST of *this* camera frame.
    """
    npix = int(field.shape[-1])
    nside = int(nside) if nside is not None else nside_from_npix(npix)
    alt, az = local_pix_altaz(nside)
    alt_t = torch.as_tensor(alt, device=field.device, dtype=torch.float32)
    az_t = torch.as_tensor(az, device=field.device, dtype=torch.float32)
    lst = lst_deg.to(device=field.device, dtype=torch.float32)
    if lst.numel() != 1:
        raise ValueError(f"lst_deg must be a scalar, got shape {tuple(lst.shape)}")
    ra, dec = altaz_to_equatorial(alt_t, az_t, lst.reshape(()))
    theta = torch.deg2rad(90.0 - dec)
    phi = torch.deg2rad(ra)
    idx = ang2pix_nest(nside, theta, phi).to(dtype=torch.int64).clamp(0, npix - 1)
    while idx.dim() < field.dim():
        idx = idx.unsqueeze(0)
    return field.gather(-1, idx.expand_as(field))


def equatorial_to_altaz(
    ra_deg: torch.Tensor,
    dec_deg: torch.Tensor,
    lst_deg: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RA/Dec [deg] and LST [deg] → altitude, azimuth [deg].

    ``ra_deg, dec_deg`` are ``(N,)``. ``lst_deg`` is ``(B, T)`` or ``(B,)``.
    Returns ``alt, az`` broadcast to ``(*lst.shape, N)``. Azimuth is from
    north, east-positive, in ``[0, 360)``.
    """
    if lst_deg.dim() == 1:
        lst_deg = lst_deg.unsqueeze(-1)
        squeeze_t = True
    else:
        squeeze_t = False
    phi = math.radians(SITE_LAT)
    ha = torch.deg2rad(lst_deg.unsqueeze(-1) - ra_deg)
    dec = torch.deg2rad(dec_deg)
    sin_dec, cos_dec = torch.sin(dec), torch.cos(dec)
    sin_alt = sin_dec * math.sin(phi) + cos_dec * math.cos(phi) * torch.cos(ha)
    sin_alt = sin_alt.clamp(-1.0, 1.0)
    alt = torch.asin(sin_alt)
    y = -cos_dec * torch.sin(ha)
    x = (sin_dec - math.sin(phi) * sin_alt) / max(math.cos(phi), 1e-12)
    az = torch.atan2(y, x)
    alt_deg = torch.rad2deg(alt)
    az_deg = torch.remainder(torch.rad2deg(az), 360.0)
    if squeeze_t:
        return alt_deg.squeeze(-2), az_deg.squeeze(-2)
    return alt_deg, az_deg


def altaz_feat(alt_deg: torch.Tensor, az_deg: torch.Tensor) -> torch.Tensor:
    """``(...,)`` alt/az deg → ``(..., 4)`` = sin alt, cos alt, sin az, cos az."""
    alt = torch.deg2rad(alt_deg)
    az = torch.deg2rad(az_deg)
    return torch.stack(
        [torch.sin(alt), torch.cos(alt), torch.sin(az), torch.cos(az)], dim=-1
    )


def pierce_en(
    alt_deg: torch.Tensor, az_deg: torch.Tensor, height_km: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sightline ∩ plane at ``height_km`` → east, north km. Low alt → NaN."""
    alt = torch.deg2rad(alt_deg)
    az = torch.deg2rad(az_deg)
    r = height_km / alt.tan().clamp(min=1e-4)
    bad = alt_deg < MIN_ALT_DEG
    east = torch.where(bad, torch.full_like(r, float("nan")), r * torch.sin(az))
    north = torch.where(bad, torch.full_like(r, float("nan")), r * torch.cos(az))
    return east, north


def sat_coord_maps(
    height: int, width: int, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """``(2, H, W)``: east and north in [-1, 1] (1 = box edge, ~165 km)."""
    r = torch.linspace(1.0, -1.0, height, device=device, dtype=dtype)
    c = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    north, east = torch.meshgrid(r, c, indexing="ij")
    return torch.stack([east, north], dim=0)


def en_to_grid(east_km: torch.Tensor, north_km: torch.Tensor) -> torch.Tensor:
    """East/north km → grid_sample xy in [-1, 1]. Top of the image is north."""
    x = (east_km / HALF_KM).clamp(-1.5, 1.5)
    y = (-north_km / HALF_KM).clamp(-1.5, 1.5)
    return torch.stack([x, y], dim=-1)


def sample_sat_at(sat_bchw: torch.Tensor, grid_xy: torch.Tensor) -> torch.Tensor:
    """Bilinear sample. ``sat`` ``(B, C, H, W)``, ``grid_xy`` ``(B, N, 2)`` → ``(B, C, N)``."""
    b, n, _ = grid_xy.shape
    grid = grid_xy.view(b, 1, n, 2)
    out = F.grid_sample(
        sat_bchw,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return out.view(b, sat_bchw.shape[1], n)


def decode_wind_uvh(
    wind_set: torch.Tensor,
    wind_mask: torch.Tensor,
    *,
    u_med=None,
    u_iqr=None,
    v_med=None,
    v_iqr=None,
    h_med=None,
    h_iqr=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Masked mean u, v (m/s) and height (km) from z-scored wind_set."""
    m = (wind_mask > 0).to(dtype=wind_set.dtype)
    n_valid = m.sum(dim=-1)
    den = n_valid.clamp_min(1.0)
    u_z = (wind_set[..., 0] * m).sum(dim=-1) / den
    v_z = (wind_set[..., 1] * m).sum(dim=-1) / den
    h_z = (wind_set[..., 2] * m).sum(dim=-1) / den
    empty = n_valid < 1
    u_med_t = _as_wind_scale(u_med, WIND_U_MED, u_z)
    u_iqr_t = _as_wind_scale(u_iqr, WIND_U_IQR, u_z)
    v_med_t = _as_wind_scale(v_med, WIND_V_MED, v_z)
    v_iqr_t = _as_wind_scale(v_iqr, WIND_V_IQR, v_z)
    h_med_t = _as_wind_scale(h_med, WIND_H_MED, h_z)
    h_iqr_t = _as_wind_scale(h_iqr, WIND_H_IQR, h_z)
    u = torch.where(empty, torch.zeros_like(u_z), u_z * u_iqr_t + u_med_t)
    v = torch.where(empty, torch.zeros_like(v_z), v_z * v_iqr_t + v_med_t)
    h = torch.where(empty, torch.full_like(h_z, float(h_med_t)), h_z * h_iqr_t + h_med_t)
    return u, v, h.clamp(1.0, 16.0)


def decode_wind_uvh_decks(
    wind_set: torch.Tensor,
    wind_mask: torch.Tensor,
    decks_km: tuple[float, ...] = DECKS_KM,
    deck_tau_km: float = 1.5,
    *,
    u_med=None,
    u_iqr=None,
    v_med=None,
    v_iqr=None,
    h_med=None,
    h_iqr=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-deck soft-mean u, v (m/s) from z-scored DMW points.

    Each deck weights points by ``exp(-|h − h_deck| / τ)``. Empty DMW falls
    back to the global mean (then zeros if that is empty too).

    ``wind_set`` is ``(..., N, 6)``, ``wind_mask`` ``(..., N)``.
    Returns ``u, v`` each ``(..., n_decks)``.
    """
    m = (wind_mask > 0).to(dtype=wind_set.dtype)
    u_z = wind_set[..., 0]
    v_z = wind_set[..., 1]
    h_z = wind_set[..., 2]
    u_med_t = _as_wind_scale(u_med, WIND_U_MED, u_z)
    u_iqr_t = _as_wind_scale(u_iqr, WIND_U_IQR, u_z)
    v_med_t = _as_wind_scale(v_med, WIND_V_MED, v_z)
    v_iqr_t = _as_wind_scale(v_iqr, WIND_V_IQR, v_z)
    h_med_t = _as_wind_scale(h_med, WIND_H_MED, h_z)
    h_iqr_t = _as_wind_scale(h_iqr, WIND_H_IQR, h_z)
    u_phys = u_z * u_iqr_t + u_med_t
    v_phys = v_z * v_iqr_t + v_med_t
    h_phys = (h_z * h_iqr_t + h_med_t).clamp(1.0, 16.0)

    n_valid = m.sum(dim=-1)
    den = n_valid.clamp_min(1.0)
    u_glob = (u_phys * m).sum(dim=-1) / den
    v_glob = (v_phys * m).sum(dim=-1) / den
    empty = n_valid < 1
    u_glob = torch.where(empty, torch.zeros_like(u_glob), u_glob)
    v_glob = torch.where(empty, torch.zeros_like(v_glob), v_glob)

    decks = torch.as_tensor(
        decks_km, device=wind_set.device, dtype=wind_set.dtype
    )
    # (..., N, D)
    w = torch.exp(-(h_phys.unsqueeze(-1) - decks).abs() / float(deck_tau_km))
    w = w * m.unsqueeze(-1)
    w_sum = w.sum(dim=-2).clamp_min(1e-6)
    u_d = (u_phys.unsqueeze(-1) * w).sum(dim=-2) / w_sum
    v_d = (v_phys.unsqueeze(-1) * w).sum(dim=-2) / w_sum
    no_deck = w.sum(dim=-2) < 1e-6
    u_d = torch.where(no_deck, u_glob.unsqueeze(-1).expand_as(u_d), u_d)
    v_d = torch.where(no_deck, v_glob.unsqueeze(-1).expand_as(v_d), v_d)
    return u_d, v_d


def _spread_bits(v: torch.Tensor) -> torch.Tensor:
    """Interleave zeros between the low 16 bits (HEALPix ``xyf2nest``)."""
    x = v.to(dtype=torch.int64) & 0xFFFF
    x = (x | (x << 8)) & 0x00FF00FF
    x = (x | (x << 4)) & 0x0F0F0F0F
    x = (x | (x << 2)) & 0x33333333
    x = (x | (x << 1)) & 0x55555555
    return x


def ang2pix_nest(nside: int, theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """Nested HEALPix pixel index. Stays on ``theta``'s device (no healpy)."""
    nside = int(nside)
    if nside < 1 or (nside & (nside - 1)) != 0:
        raise ValueError(f"nested ang2pix needs power-of-two nside, got {nside}")
    order = nside.bit_length() - 1
    twopi = 2.0 * math.pi
    halfpi = 0.5 * math.pi
    twothird = 2.0 / 3.0
    z = torch.cos(theta)
    za = z.abs()
    tt = torch.remainder(phi, twopi)
    tt = torch.where(tt < 0, tt + twopi, tt) / halfpi
    n = torch.as_tensor(float(nside), device=theta.device, dtype=theta.dtype)

    jp_e = torch.floor(n * (0.5 + tt - z * 0.75)).to(dtype=torch.int64)
    jm_e = torch.floor(n * (0.5 + tt + z * 0.75)).to(dtype=torch.int64)
    ifp = jp_e >> order
    ifm = jm_e >> order
    face_eq = torch.where(
        ifp == ifm, ifp | 4, torch.where(ifp < ifm, ifp, ifm + 8)
    )
    ix_eq = jm_e & (nside - 1)
    iy_eq = nside - (jp_e & (nside - 1)) - 1

    ntt = tt.to(dtype=torch.int64).clamp(0, 3)
    tp = tt - ntt.to(dtype=tt.dtype)
    tmp = n * torch.sqrt((3.0 * (1.0 - za)).clamp_min(0.0))
    jp_p = (tp * tmp).to(dtype=torch.int64).clamp(0, nside - 1)
    jm_p = ((1.0 - tp) * tmp).to(dtype=torch.int64).clamp(0, nside - 1)
    north = z >= 0
    face_p = torch.where(north, ntt, ntt + 8)
    ix_p = torch.where(north, nside - jm_p - 1, jp_p)
    iy_p = torch.where(north, nside - jp_p - 1, jm_p)

    equatorial = za <= twothird
    face = torch.where(equatorial, face_eq, face_p)
    ix = torch.where(equatorial, ix_eq, ix_p)
    iy = torch.where(equatorial, iy_eq, iy_p)
    pix = (face << (2 * order)) + _spread_bits(ix) + (_spread_bits(iy) << 1)
    return pix


def _ang2pix_nest(nside: int, theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return ang2pix_nest(nside, theta, phi)


def altaz_to_equatorial(
    alt_deg: torch.Tensor, az_deg: torch.Tensor, lst_deg: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of ``equatorial_to_altaz``. ``lst_deg`` broadcasts as ``alt``[:-1]."""
    phi = math.radians(SITE_LAT)
    alt = torch.deg2rad(alt_deg)
    az = torch.deg2rad(az_deg)
    sin_dec = torch.sin(alt) * math.sin(phi) + torch.cos(alt) * math.cos(phi) * torch.cos(az)
    sin_dec = sin_dec.clamp(-1.0, 1.0)
    dec = torch.asin(sin_dec)
    y = -torch.cos(alt) * torch.sin(az)
    x = (torch.sin(alt) - math.sin(phi) * sin_dec) / max(math.cos(phi), 1e-12)
    ha = torch.atan2(y, x)
    lst = lst_deg
    while lst.dim() < alt_deg.dim():
        lst = lst.unsqueeze(-1)
    ra = torch.remainder(lst - torch.rad2deg(ha), 360.0)
    return ra, torch.rad2deg(dec)


def ground_to_altaz(
    east_km: torch.Tensor, north_km: torch.Tensor, height_km: float
) -> tuple[torch.Tensor, torch.Tensor]:
    r = torch.hypot(east_km, north_km)
    alt = torch.rad2deg(torch.atan2(torch.full_like(r, height_km), r.clamp_min(1e-3)))
    az = torch.remainder(torch.rad2deg(torch.atan2(east_km, north_km)), 360.0)
    return alt, az


def advect_gather_index(
    nside: int,
    east_km: torch.Tensor,
    north_km: torch.Tensor,
    u_ms: torch.Tensor,
    v_ms: torch.Tensor,
    lst_deg: torch.Tensor,
    height_km: float,
    dt_sec: float = LEAD_SEC,
) -> torch.Tensor:
    """Pixel that advects into each tile over ``dt_sec`` at deck ``height_km``.

    ``east_km, north_km`` are the *current* pierce points ``(B, N)``.
    ``u_ms, v_ms, lst_deg`` are ``(B,)``. Returns ``(B, N)`` int64 indices
    into an *equatorial* nested map. Prefer ``advect_gather_index_local`` for
    horizon-locked camera fields.
    """
    dx = (u_ms * dt_sec / 1000.0).unsqueeze(-1)
    dy = (v_ms * dt_sec / 1000.0).unsqueeze(-1)
    src_e = east_km - dx
    src_n = north_km - dy
    alt, az = ground_to_altaz(src_e, src_n, height_km)
    ra, dec = altaz_to_equatorial(alt, az, lst_deg)
    theta = torch.deg2rad(90.0 - dec)
    phi = torch.deg2rad(ra)
    npix = 12 * nside * nside
    pix = _ang2pix_nest(nside, theta, phi)
    return pix.clamp(0, npix - 1)


def advect_gather_index_local(
    nside: int,
    east_km: torch.Tensor,
    north_km: torch.Tensor,
    u_ms: torch.Tensor,
    v_ms: torch.Tensor,
    height_km: float,
    dt_sec: float | torch.Tensor = LEAD_SEC,
    bad: torch.Tensor | None = None,
) -> torch.Tensor:
    """Horizon-sphere pixel that advects into each tile over ``dt_sec``.

    Field layout is local alt-az HEALPix (pole = zenith, φ = 0 = north), so
    no LST is needed. ``east_km, north_km`` are ``(B, N)`` pierce points;
    ``u_ms, v_ms`` are ``(B,)`` or ``(B, 1)``. ``dt_sec`` is a float or
    ``(B,)``. Returns ``(B, N)`` int64. When ``bad`` is set (or pierce is
    non-finite), the index is identity.
    """
    if torch.is_tensor(dt_sec):
        dt = dt_sec.to(device=east_km.device, dtype=east_km.dtype)
    else:
        dt = east_km.new_tensor(float(dt_sec))
    dx = u_ms * (dt / 1000.0)
    dy = v_ms * (dt / 1000.0)
    while dx.dim() < east_km.dim():
        dx = dx.unsqueeze(-1)
        dy = dy.unsqueeze(-1)
    src_e = east_km - dx
    src_n = north_km - dy
    alt, az = ground_to_altaz(src_e, src_n, height_km)
    theta = torch.deg2rad((90.0 - alt).clamp(1e-3, 180.0 - 1e-3))
    phi = torch.deg2rad(torch.remainder(az, 360.0))
    npix = 12 * int(nside) * int(nside)
    pix = _ang2pix_nest(int(nside), theta, phi).clamp(0, npix - 1)
    identity = torch.arange(east_km.shape[-1], device=east_km.device, dtype=torch.int64)
    identity = identity.unsqueeze(0).expand(east_km.shape[0], -1)
    invalid = ~torch.isfinite(east_km) | ~torch.isfinite(north_km)
    if bad is not None:
        invalid = invalid | bad
    return torch.where(invalid, identity, pix)
