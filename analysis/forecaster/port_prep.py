"""Horizon lock, moon cut, and per-night cap for the encoded cache.

Used by the forecaster and the camera-only SimVP / ConvLSTM trainers.
Does not rewrite the DREAM memmap.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, Subset

from analysis.baselines.neural.filter_overcast import load_keep_indices, window_moon_ok
from analysis.eval_harness.dream_data import pick_evenly
from analysis.forecaster.geom import (
    LEAD_SEC,
    altaz_to_equatorial,
    ang2pix_nest,
    gather_equatorial_to_local,
    local_pix_altaz,
    lst_deg_from_unix,
    nside_from_npix,
)
from analysis.forecaster.sat_ingest import load_times_sidecar
from analysis.satellite.sites import MOON_ALT_MIN_DEG, MOON_ILLUM_MAX

LABEL_SMOOTH_SIGMA = 3.5
MAX_TRAIN_PER_NIGHT = 40
FILTER_MOON = True


class HorizonLockPairs(Dataset):
    """Reindex camera maps onto a horizon sphere (zenith = pole, φ = 0 = north).

    ``sky`` / ``last`` use each history frame's LST. ``y`` uses the +lead
    target instant (``unix[-1] + lead_sec``). Memmaps stay equatorial.
    """

    def __init__(
        self,
        inner: Dataset,
        *,
        lead_sec: float = LEAD_SEC,
        unix: np.ndarray | None = None,
        dates: np.ndarray | None = None,
    ) -> None:
        self._inner = inner
        self.lead_sec = float(lead_sec)
        self._unix = None if unix is None else np.asarray(unix, dtype=np.float64)
        self.dates = dates if dates is not None else getattr(inner, "dates", None)

    def __len__(self) -> int:
        return len(self._inner)

    def reopen_extra(self) -> None:
        fn = getattr(self._inner, "reopen_extra", None)
        if fn is not None:
            fn()

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = dict(self._inner[int(idx)])
        unix = row.get("unix")
        if unix is None and self._unix is not None:
            unix = torch.as_tensor(self._unix[int(idx)], dtype=torch.float64)
            row["unix"] = unix
        if unix is None:
            raise KeyError("horizon-lock needs unix on the row or a times sidecar")
        unix_t = unix.to(dtype=torch.float64).reshape(-1)
        nside = nside_from_npix(int(row["sky"].shape[-1]))
        lst_hist = lst_deg_from_unix(unix_t)
        sky = row["sky"].clone()
        sky_mask = row["sky_mask"].clone()
        for t in range(int(sky.shape[0])):
            lst = lst_hist[t : t + 1]
            sky[t] = gather_equatorial_to_local(sky[t], lst, nside)
            sky_mask[t] = gather_equatorial_to_local(sky_mask[t], lst, nside)
        row["sky"] = sky
        row["sky_mask"] = sky_mask
        lst_last = lst_hist[-1:]
        if "last" in row:
            row["last"] = gather_equatorial_to_local(row["last"], lst_last, nside)
        if "last_mask" in row:
            row["last_mask"] = gather_equatorial_to_local(
                row["last_mask"], lst_last, nside
            )
        y = row["y"]
        if y.ndim == 2:
            leads = row.get("horizon_sec")
            if leads is None:
                leads = y.new_full((int(y.shape[0]),), float(self.lead_sec))
            locked = []
            locked_m = []
            ym = row["y_mask"]
            for k in range(int(y.shape[0])):
                lst_y = lst_deg_from_unix(unix_t[-1:] + float(leads[k]))
                locked.append(gather_equatorial_to_local(y[k], lst_y, nside))
                locked_m.append(gather_equatorial_to_local(ym[k], lst_y, nside))
            row["y"] = torch.stack(locked, dim=0)
            row["y_mask"] = torch.stack(locked_m, dim=0)
        else:
            if "lead_sec" in row:
                lead = float(torch.as_tensor(row["lead_sec"]).reshape(-1)[0])
            else:
                lead = float(self.lead_sec)
            lst_y = lst_deg_from_unix(unix_t[-1:] + lead)
            row["y"] = gather_equatorial_to_local(row["y"], lst_y, nside)
            row["y_mask"] = gather_equatorial_to_local(row["y_mask"], lst_y, nside)
        if "occ" in row:
            row["occ"] = gather_equatorial_to_local(row["occ"], lst_y, nside)
        if "occ_n" in row:
            row["occ_n"] = gather_equatorial_to_local(row["occ_n"], lst_y, nside)
        return row

    def __getattr__(self, name: str):
        return getattr(self._inner, name)


def lock_numpy(
    field: np.ndarray,
    unix: float,
    nside: int | None = None,
) -> np.ndarray:
    """Nearest-neighbour gather of an equatorial map onto the horizon sphere."""
    t = torch.as_tensor(np.ascontiguousarray(field))
    lst = lst_deg_from_unix(torch.as_tensor([float(unix)], dtype=torch.float64))
    return np.asarray(gather_equatorial_to_local(t, lst, nside))


def lock_numpy_batch(
    field: np.ndarray,
    unix: np.ndarray,
    nside: int | None = None,
) -> np.ndarray:
    """Horizon-lock a stack of equatorial maps. ``field`` is ``(N, npix)``."""
    t = torch.as_tensor(np.ascontiguousarray(field))
    if t.ndim != 2:
        raise ValueError(f"lock_numpy_batch wants (N, npix), got {tuple(t.shape)}")
    n, npix = int(t.shape[0]), int(t.shape[1])
    nside = int(nside) if nside is not None else nside_from_npix(npix)
    lst = lst_deg_from_unix(
        torch.as_tensor(np.asarray(unix, dtype=np.float64).reshape(-1))
    )
    if int(lst.numel()) != n:
        raise ValueError(f"unix n={lst.numel()} != field n={n}")
    alt, az = local_pix_altaz(nside)
    alt_t = torch.as_tensor(alt, dtype=torch.float32)
    az_t = torch.as_tensor(az, dtype=torch.float32)
    # lst (N,) with alt (npix,): pass lst as (N, 1) so ra is (N, npix).
    ra, dec = altaz_to_equatorial(alt_t, az_t, lst.to(dtype=torch.float32).unsqueeze(-1))
    if ra.shape != (n, npix):
        ra = ra.expand(n, npix)
        dec = dec.expand(n, npix) if dec.ndim == 2 else dec.unsqueeze(0).expand(n, npix)
    elif dec.shape == (npix,):
        dec = dec.unsqueeze(0).expand(n, npix)
    theta = torch.deg2rad(90.0 - dec)
    phi = torch.deg2rad(ra)
    idx = ang2pix_nest(nside, theta, phi).to(dtype=torch.int64).clamp(0, npix - 1)
    return np.asarray(t.gather(-1, idx))


def attach_horizon_lock(
    data,
    cache_split: Path | None = None,
    *,
    lead_sec: float = LEAD_SEC,
):
    """Wrap ``data`` so camera maps are horizon-locked. Loads times if needed."""
    unix = getattr(data, "unix", None)
    dates = getattr(data, "dates", None)
    if cache_split is not None and (unix is None or dates is None):
        dest = Path(cache_split)
        if (dest / "times.npz").is_file():
            times = load_times_sidecar(dest)
            unix = times["unix"] if unix is None else unix
            dates = times["dates"] if dates is None else dates
    return HorizonLockPairs(data, lead_sec=lead_sec, unix=unix, dates=dates)


def cap_keep_per_night(
    keep: np.ndarray,
    dates: np.ndarray,
    max_per_night: int,
) -> np.ndarray:
    keep = np.asarray(keep, dtype=np.int64)
    cap = int(max_per_night)
    if cap <= 0 or keep.size == 0:
        return keep
    by: dict[str, list[int]] = {}
    for i in keep:
        by.setdefault(str(dates[int(i)]), []).append(int(i))
    out: list[np.ndarray] = []
    for night in sorted(by):
        rows = np.unique(np.asarray(by[night], dtype=np.int64))
        if rows.size > cap:
            pick = np.asarray(pick_evenly(np.arange(rows.size), cap), dtype=np.int64)
            rows = rows[pick]
        out.append(rows)
    return np.concatenate(out) if out else keep


def filter_keep_moon(keep: np.ndarray, cache: Path, n: int) -> np.ndarray:
    keep = np.asarray(keep, dtype=np.int64)
    ok = window_moon_ok(cache, n)
    return keep[ok[keep]]


def apply_port_keeps(
    data,
    keep_dir: Path | None,
    cache_split: Path,
    *,
    filter_moon: bool = FILTER_MOON,
    max_train_per_night: int = MAX_TRAIN_PER_NIGHT,
    val_from_train: bool = True,
) -> tuple:
    """Moon-dark ∩ keep-set, then cap train windows per night. Val uncapped."""
    n = int(getattr(data, "n", len(data)))
    if keep_dir is None:
        train_keep = np.arange(n, dtype=np.int64)
        val_keep = train_keep.copy() if val_from_train else train_keep
    else:
        train_keep = load_keep_indices(Path(keep_dir) / "train.npy", n, "train")
        val_keep = load_keep_indices(Path(keep_dir) / "validation.npy", n, "validation")
    n_train_in, n_val_in = int(train_keep.size), int(val_keep.size)
    if filter_moon:
        train_keep = filter_keep_moon(train_keep, cache_split, n)
        val_keep = filter_keep_moon(val_keep, cache_split, n)
    n_train_moon, n_val_moon = int(train_keep.size), int(val_keep.size)
    dates = getattr(data, "dates", None)
    if dates is None and (Path(cache_split) / "times.npz").is_file():
        dates = load_times_sidecar(cache_split)["dates"]
    if dates is None:
        raise SystemExit("port keep: need dates on the dataset or a times sidecar")
    n_train_full = int(train_keep.size)
    if int(max_train_per_night) > 0:
        train_keep = cap_keep_per_night(train_keep, dates, int(max_train_per_night))
    moon_note = (
        f" ∩ moon dark (illum<{MOON_ILLUM_MAX} or alt≤{MOON_ALT_MIN_DEG:g}°)"
        if filter_moon
        else ""
    )
    print(
        f"  port keep  train={train_keep.size}/{n_train_in}  "
        f"val={val_keep.size}/{n_val_in}  (inner nights{moon_note})",
        flush=True,
    )
    if int(max_train_per_night) > 0:
        n_nights = len({str(dates[int(i)]) for i in train_keep})
        print(
            f"  train night cap  {train_keep.size}/{n_train_full}  "
            f"≤{int(max_train_per_night)} / night  nights={n_nights}  "
            f"(val uncapped {val_keep.size}; moon {n_train_moon}/{n_val_moon})",
            flush=True,
        )
    return (
        Subset(data, train_keep.tolist()),
        Subset(data, val_keep.tolist()),
        train_keep,
        val_keep,
    )


def _memmap_arrays(data) -> dict[str, np.ndarray]:
    """Camera memmaps on the innermost PairsDataset (no recache)."""
    cur = data
    seen: set[int] = set()
    while id(cur) not in seen:
        seen.add(id(cur))
        dest = getattr(cur, "_dest", None)
        if dest is not None and hasattr(cur, "_ensure_arrays"):
            return cur._ensure_arrays()
        inner = getattr(cur, "_inner", None)
        if inner is None:
            inner = getattr(cur, "dataset", None)
        if inner is None:
            break
        cur = inner
    fn = getattr(cur, "_ensure_arrays", None)
    if fn is None:
        raise RuntimeError("band mix needs a memmap PairsDataset")
    return fn()


def window_mean_t_keep(
    data,
    keep: np.ndarray,
    *,
    mean: float,
    std: float,
    chunk: int = 512,
) -> np.ndarray:
    """Per-window mean observed transmission for ``keep`` memmap rows. NaN if no valid tiles."""
    arrays = _memmap_arrays(data)
    y = arrays["y"]
    ym = arrays["y_mask"]
    keep = np.asarray(keep, dtype=np.int64)
    out = np.full(keep.size, np.nan, dtype=np.float64)
    mu, sd = float(mean), float(std)
    step = max(1, int(chunk))
    for j in range(0, keep.size, step):
        ix = keep[j : j + step]
        yy = np.asarray(y[ix])
        valid = np.asarray(ym[ix]) > 0
        t = yy * sd + mu
        n_valid = valid.reshape(valid.shape[0], -1).sum(axis=1).astype(np.float64)
        acc = np.where(valid, t, 0.0).reshape(t.shape[0], -1).sum(axis=1)
        ok = n_valid > 0
        chunk_out = np.full(yy.shape[0], np.nan, dtype=np.float64)
        chunk_out[ok] = acc[ok] / n_valid[ok]
        out[j : j + ix.size] = chunk_out
    return out


def build_band_mix_pools(
    moon_keep: np.ndarray,
    dates: np.ndarray,
    mean_t: np.ndarray,
    *,
    max_per_night: int = MAX_TRAIN_PER_NIGHT,
    t_min: float = 0.1,
    t_max: float = 0.9,
    band_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """All-night 40/night pool and night-capped in-band pool.

    ``mean_t`` is aligned with ``moon_keep``. If ``band_mask`` is set it
    selects the band pool (multihorizon: any lead in range). Val is not
    filtered here.
    """
    moon_keep = np.asarray(moon_keep, dtype=np.int64)
    mean_t = np.asarray(mean_t, dtype=np.float64)
    if mean_t.shape[0] != moon_keep.size:
        raise ValueError(
            f"mean_t n={mean_t.shape[0]} != moon_keep n={moon_keep.size}"
        )
    all_cap = cap_keep_per_night(moon_keep, dates, int(max_per_night))
    if band_mask is None:
        band_mask = (
            np.isfinite(mean_t)
            & (mean_t >= float(t_min))
            & (mean_t <= float(t_max))
        )
    else:
        band_mask = np.asarray(band_mask, dtype=bool)
        if band_mask.shape[0] != moon_keep.size:
            raise ValueError(
                f"band_mask n={band_mask.shape[0]} != moon_keep n={moon_keep.size}"
            )
    band_keep = moon_keep[band_mask]
    band_cap = cap_keep_per_night(band_keep, dates, int(max_per_night))
    info = {
        "t_min": float(t_min),
        "t_max": float(t_max),
        "max_per_night": int(max_per_night),
        "n_moon": int(moon_keep.size),
        "n_moon_band": int(band_keep.size),
        "n_all_cap": int(all_cap.size),
        "n_band_cap": int(band_cap.size),
        "n_all_nights": int(len({str(dates[int(i)]) for i in all_cap})) if all_cap.size else 0,
        "n_band_nights": int(len({str(dates[int(i)]) for i in band_cap})) if band_cap.size else 0,
        "n_moon_nights": int(len({str(dates[int(i)]) for i in moon_keep})) if moon_keep.size else 0,
    }
    return all_cap, band_cap, info


def mix_all_and_band_keep(
    all_cap: np.ndarray,
    band_cap: np.ndarray,
    *,
    band_frac: float = 0.5,
    seed: int = 42,
    n_target: int | None = None,
) -> tuple[np.ndarray, dict]:
    """One epoch of memmap rows: ``(1-frac)`` all-night cap + ``frac`` in-band.

    Band slots cycle the smaller pool with replacement so the epoch stays
    ``len(all_cap)`` windows. Duplicate rows are intentional upsampling.
    """
    all_cap = np.asarray(all_cap, dtype=np.int64)
    band_cap = np.asarray(band_cap, dtype=np.int64)
    if all_cap.size == 0:
        raise ValueError("band mix needs a non-empty all-night pool")
    n_target = int(all_cap.size if n_target is None else n_target)
    frac = float(band_frac)
    if not (0.0 < frac <= 1.0):
        raise ValueError(f"--band-mix must be in (0, 1], got {frac}")
    n_band = int(round(n_target * frac))
    n_all = int(n_target - n_band)
    rng = np.random.default_rng(int(seed))
    if n_all <= 0:
        all_pick = np.zeros(0, dtype=np.int64)
    elif n_all <= all_cap.size:
        all_pick = rng.choice(all_cap, size=n_all, replace=False)
    else:
        all_pick = rng.choice(all_cap, size=n_all, replace=True)
    n_replace = 0
    if band_cap.size == 0 or n_band <= 0:
        band_pick = np.zeros(0, dtype=np.int64)
        n_band = 0
        n_all = n_target
        all_pick = (
            rng.choice(all_cap, size=n_target, replace=False)
            if n_target <= all_cap.size
            else rng.choice(all_cap, size=n_target, replace=True)
        )
    elif band_cap.size >= n_band:
        band_pick = rng.choice(band_cap, size=n_band, replace=False)
    else:
        extra = rng.choice(band_cap, size=n_band - band_cap.size, replace=True)
        band_pick = np.concatenate([band_cap, extra])
        n_replace = int(n_band - band_cap.size)
    mixed = np.concatenate([all_pick, band_pick])
    mixed = mixed[np.argsort(mixed, kind="stable")]
    info = {
        "band_frac": frac,
        "n_target": n_target,
        "n_all_slots": int(all_pick.size),
        "n_band_slots": int(band_pick.size),
        "n_band_replace": int(n_replace),
        "n_unique": int(np.unique(mixed).size),
    }
    return mixed, info


class BandMixEpochSampler(Sampler[int]):
    """Each epoch redraws the 50/50 mix, then streams nights in memmap order."""

    def __init__(
        self,
        *,
        all_local: np.ndarray,
        band_local: np.ndarray,
        dates_local: np.ndarray,
        band_frac: float,
        n_target: int,
        seed: int,
    ) -> None:
        self.all_local = np.asarray(all_local, dtype=np.int64)
        self.band_local = np.asarray(band_local, dtype=np.int64)
        self.dates_local = np.asarray(dates_local)
        self.band_frac = float(band_frac)
        self.n_target = int(n_target)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.n_target

    def __iter__(self):
        mixed_rows, _ = mix_all_and_band_keep(
            self.all_local,
            self.band_local,
            band_frac=self.band_frac,
            seed=self.seed + self.epoch,
            n_target=self.n_target,
        )
        by: dict[str, list[int]] = {}
        for loc in mixed_rows:
            i = int(loc)
            by.setdefault(str(self.dates_local[i]), []).append(i)
        rng = np.random.default_rng(self.seed + 1_000_003 * (self.epoch + 1))
        nights = sorted(by)
        for j in rng.permutation(len(nights)):
            idxs = np.asarray(by[nights[int(j)]], dtype=np.int64)
            yield from idxs[np.argsort(idxs, kind="stable")].tolist()


def apply_band_mix_train(
    data,
    moon_keep: np.ndarray,
    *,
    cache_split: Path | None,
    sky_mean: float,
    sky_std: float,
    max_per_night: int = MAX_TRAIN_PER_NIGHT,
    band_frac: float = 0.5,
    t_min: float = 0.1,
    t_max: float = 0.9,
    seed: int = 42,
) -> tuple[Subset, BandMixEpochSampler, dict]:
    """Moon-dark train → unique union Subset + per-epoch two-stream sampler.

    On a multihorizon cache the in-band pool is windows whose mean observed transmission
    is in ``[t_min, t_max]`` at *any* cached lead. Single-horizon caches
    still use ``y`` (+20 min). Val is unchanged. Does not write keep files.
    """
    dates = getattr(data, "dates", None)
    if dates is None and cache_split is not None and (Path(cache_split) / "times.npz").is_file():
        dates = load_times_sidecar(cache_split)["dates"]
    if dates is None:
        raise SystemExit("band mix: need dates on the dataset or a times sidecar")
    moon_keep = np.asarray(moon_keep, dtype=np.int64)
    print(
        f"  band mix  mean T on {moon_keep.size} moon-dark train windows…",
        flush=True,
    )
    band_mask = None
    band_rule = "gold_y"
    mh_info: dict = {}
    cache_path = None if cache_split is None else Path(cache_split)
    from analysis.forecaster.multihorizon import (
        is_multihorizon_cache,
        window_mean_t_horizons,
    )

    if cache_path is not None and is_multihorizon_cache(cache_path):
        mean_hk, horizons = window_mean_t_horizons(
            cache_path,
            moon_keep,
            mean=float(sky_mean),
            std=float(sky_std),
        )
        in_band = (
            np.isfinite(mean_hk)
            & (mean_hk >= float(t_min))
            & (mean_hk <= float(t_max))
        )
        band_mask = np.any(in_band, axis=1)
        mean_t = np.where(
            np.any(np.isfinite(mean_hk), axis=1),
            np.nanmean(mean_hk, axis=1),
            np.nan,
        )
        band_rule = "any_horizon"
        mh_info = {
            "horizons_sec": list(horizons),
            "n_moon_band_by_lead": {
                str(h): int(in_band[:, k].sum()) for k, h in enumerate(horizons)
            },
            "n_moon_band_any": int(band_mask.sum()),
            "n_moon_band_all": int(np.all(in_band, axis=1).sum()),
        }
        print(
            f"  band mix  any-lead T∈[{t_min:g}, {t_max:g}]  "
            + "  ".join(
                f"{h}s={mh_info['n_moon_band_by_lead'][str(h)]}"
                for h in horizons
            )
            + f"  any={int(band_mask.sum())}",
            flush=True,
        )
    else:
        mean_t = window_mean_t_keep(
            data, moon_keep, mean=float(sky_mean), std=float(sky_std)
        )
    all_cap, band_cap, pool_info = build_band_mix_pools(
        moon_keep,
        dates,
        mean_t,
        max_per_night=int(max_per_night),
        t_min=float(t_min),
        t_max=float(t_max),
        band_mask=band_mask,
    )
    pool_info["band_rule"] = band_rule
    pool_info.update(mh_info)
    if band_cap.size == 0:
        raise SystemExit(
            f"band mix: no train windows with mean T in [{t_min:g}, {t_max:g}]"
        )
    n_target = int(all_cap.size)
    _, mix_once = mix_all_and_band_keep(
        all_cap, band_cap, band_frac=float(band_frac), seed=int(seed), n_target=n_target
    )
    unique = np.unique(np.concatenate([all_cap, band_cap]))
    row_to_local = {int(r): i for i, r in enumerate(unique)}
    train_set = Subset(data, unique.tolist())
    dates_local = np.asarray([str(dates[int(r)]) for r in unique], dtype=object)
    all_local = np.asarray([row_to_local[int(i)] for i in all_cap], dtype=np.int64)
    band_local = np.asarray([row_to_local[int(i)] for i in band_cap], dtype=np.int64)
    sampler = BandMixEpochSampler(
        all_local=all_local,
        band_local=band_local,
        dates_local=dates_local,
        band_frac=float(band_frac),
        n_target=n_target,
        seed=int(seed),
    )
    band_in_all = 0
    if all_cap.size:
        if band_mask is not None:
            in_band_by_row = {int(r): bool(m) for r, m in zip(moon_keep, band_mask)}
            band_in_all = int(sum(in_band_by_row[int(i)] for i in all_cap))
        else:
            mt_by_row = {int(r): float(t) for r, t in zip(moon_keep, mean_t)}
            all_mt = np.asarray([mt_by_row[int(i)] for i in all_cap], dtype=np.float64)
            band_in_all = int(
                np.sum(
                    np.isfinite(all_mt)
                    & (all_mt >= float(t_min))
                    & (all_mt <= float(t_max))
                )
            )
    expected_band = mix_once["n_all_slots"] * (
        band_in_all / max(all_cap.size, 1)
    ) + mix_once["n_band_slots"]
    info = {
        **pool_info,
        **mix_once,
        "resample_each_epoch": True,
        "n_unique_union": int(unique.size),
        "n_band_in_all_cap": int(band_in_all),
        "epoch_band_instances_seed": int(round(expected_band)),
        "epoch_band_frac_seed": float(expected_band / max(n_target, 1)),
    }
    print(
        f"  band mix  T∈[{t_min:g}, {t_max:g}]  frac={band_frac:g}  "
        f"all_cap={all_cap.size}/{pool_info['n_all_nights']}n  "
        f"band_cap={band_cap.size}/{pool_info['n_band_nights']}n  "
        f"epoch={n_target}  unique_union={unique.size}",
        flush=True,
    )
    print(
        f"  band mix  slots all={mix_once['n_all_slots']}  "
        f"band={mix_once['n_band_slots']} "
        f"(upsample +{mix_once['n_band_replace']})  "
        f"~{100.0 * info['epoch_band_frac_seed']:.0f}% band instances/epoch  "
        f"val stays full inner",
        flush=True,
    )
    return train_set, sampler, info
