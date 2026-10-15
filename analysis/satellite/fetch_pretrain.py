"""Download night-time GOES-East cutouts used by the forecaster.

Each ABI full-disk scan is read once and sliced to the sites in
``analysis.satellite.sites``. Products are C07, C13, C15, ACM, ACHA, ACTP,
LST, and TPW. Brightness-temperature differences are computed when the cube
is loaded. Files land under ``$TRANSMISSIONS_MNT/goes_pretrain``.
"""

from __future__ import annotations

from analysis.paths import MNT

import argparse
import datetime as dt
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from analysis.satellite.goes import (
    PRODUCTS,
    acha_npz_is_2km,
    bucket_name,
    find_scans,
    read_patches,
    satellite_id,
)
from analysis.satellite.sites import SITE_BY_NAME, SITES, Site, night_window

DEFAULT_OUT = Path(str(MNT) + "/goes_pretrain")
DEFAULT_START = dt.date(2025, 6, 27)
DEFAULT_END = dt.date(2026, 5, 25)
MODEL_PRODUCTS = ("c07", "c13", "c15", "acm", "acha", "actp", "lst", "tpw")
TARGETS = {
    "c07": ("c07", "CMI", 7),
    "c08": ("c08", "CMI", 8),
    "c09": ("c09", "CMI", 9),
    "c10": ("c10", "CMI", 10),
    "c13": ("c13", "CMI", 13),
    "c15": ("c15", "CMI", 15),
    "acm": ("acm", "BCM", None),
    "acha": ("acha", "HT", None),
    "actp": ("actp", "Phase", None),
    "lst": ("lst", "LST", None),
    "tpw": ("tpw", "TPW", None),
}
INDEX_NAME = "index.json"


def _dates(start: dt.date, end: dt.date) -> list[dt.date]:
    out = []
    d = start
    while d <= end:
        out.append(d)
        d += dt.timedelta(days=1)
    return out


def site_night_dir(root: Path, site: Site, evening: dt.date) -> Path:
    return root / site.name / evening.isoformat()


def product_path(root: Path, site: Site, evening: dt.date, product: str) -> Path:
    return site_night_dir(root, site, evening) / f"{product}.npz"


def night_complete(root: Path, site: Site, evening: dt.date, products: tuple[str, ...]) -> bool:
    return all(product_path(root, site, evening, p).is_file() for p in products)


def missing_products(
    root: Path, site: Site, evening: dt.date, products: tuple[str, ...], *, force: bool
) -> list[str]:
    if force:
        return list(products)
    missing = []
    for p in products:
        path = product_path(root, site, evening, p)
        if not path.is_file():
            missing.append(p)
        elif p == "acha" and not acha_npz_is_2km(path):
            missing.append(p)
    return missing


def _write_product(path: Path, frames: list, site: Site, product: str, var: str) -> dict:
    frames.sort(key=lambda kv: kv[0])
    lat = frames[0][1].lat
    lon = frames[0][1].lon
    values = []
    unix = []
    shape0 = None
    for t, patch in frames:
        if shape0 is None:
            shape0 = patch.values.shape
        if patch.values.shape != shape0:
            continue
        values.append(patch.values.astype(np.float32))
        unix.append(t.timestamp())
    if not values:
        return {"missing": True}
    payload = {
        "values": np.stack(values),
        "unix": np.asarray(unix, dtype=np.float64),
        "lat": lat.astype(np.float32),
        "lon": lon.astype(np.float32),
        "units": frames[0][1].units,
        "long_name": frames[0][1].long_name,
        "product": PRODUCTS[product][0],
        "var": var,
        "site": site.name,
        "site_lat": site.lat,
        "site_lon": site.lon,
        "satellite": satellite_id(frames[0][0]),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return {
        "path": str(path),
        "n": int(payload["values"].shape[0]),
        "shape": list(payload["values"].shape),
        "size_mb": round(path.stat().st_size / 1e6, 3),
    }


def fetch_evening(
    evening: dt.date,
    *,
    products: tuple[str, ...] = MODEL_PRODUCTS,
    out_dir: Path = DEFAULT_OUT,
    sites: tuple[Site, ...] = SITES,
    workers: int = 4,
    force: bool = False,
    max_alt_deg: float = -12.0,
) -> dict:
    """Download one Chilean evening for every site still missing products."""
    need = {s.name: missing_products(out_dir, s, evening, products, force=force) for s in sites}
    pending = [s for s in sites if need[s.name]]
    rec: dict = {"date": evening.isoformat(), "sites": {}}
    if not pending:
        rec["skipped"] = True
        return rec

    windows: dict[str, tuple[dt.datetime, dt.datetime]] = {}
    for site in pending:
        span = night_window(site, evening, max_alt_deg=max_alt_deg)
        if span is None:
            rec["sites"][site.name] = {"missing_night": True}
            continue
        windows[site.name] = span
    if not windows:
        rec["no_dark"] = True
        return rec

    t0 = min(a for a, _ in windows.values())
    t1 = max(b for _, b in windows.values())
    sat = satellite_id(t0)
    bucket = bucket_name(sat)
    rec["satellite"] = sat
    rec["union"] = [t0.strftime("%Y-%m-%dT%H:%M:%SZ"), t1.strftime("%Y-%m-%dT%H:%M:%SZ")]
    print(
        f"  {evening.isoformat()}  {sat}  {t0:%H:%M}-{t1:%H:%M} UTC  "
        f"sites={list(windows)}",
        flush=True,
    )

    want_by_prod: dict[str, list[Site]] = {}
    for product in products:
        want = [s for s in pending if product in need[s.name] and s.name in windows]
        if want:
            want_by_prod[product] = want

    scan_map: dict[str, list[tuple[dt.datetime, str]]] = {}
    with ThreadPoolExecutor(max_workers=min(8, max(len(want_by_prod), 1))) as pool:
        futs = {
            pool.submit(
                find_scans,
                TARGETS[product][0],
                t0,
                t1,
                TARGETS[product][2],
                bucket=bucket,
            ): product
            for product in want_by_prod
        }
        for fut in as_completed(futs):
            product = futs[fut]
            scan_map[product] = fut.result()
            print(
                f"    {product}: {len(scan_map[product])} scans  "
                f"sites={[s.name for s in want_by_prod[product]]}",
                flush=True,
            )

    jobs: list[tuple[str, str, list[tuple[str, float, float]], dict, dt.datetime, str]] = []
    for product, want in want_by_prod.items():
        scans = scan_map.get(product) or []
        if not scans:
            for site in want:
                rec["sites"].setdefault(site.name, {})[product] = {"missing": True}
            continue
        var = TARGETS[product][1]
        site_tuples = [(s.name, s.lat, s.lon) for s in want]
        want_windows = {s.name: windows[s.name] for s in want}
        for t, key in scans:
            jobs.append((product, var, site_tuples, want_windows, t, key))

    site_frames: dict[str, dict[str, list]] = {
        product: {s.name: [] for s in want} for product, want in want_by_prod.items()
    }

    def _one(job):
        product, var, site_tuples, want_windows, t, key = job
        last_exc: Exception | None = None
        for _attempt in range(2):
            try:
                patches = read_patches(key, var, site_tuples, bucket=bucket)
                keep = {}
                for name, (a, b) in want_windows.items():
                    if a <= t <= b:
                        keep[name] = (t, patches[name])
                return product, keep
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
        raise last_exc  # type: ignore[misc]

    n_jobs = len(jobs)
    if n_jobs:
        print(f"    download {n_jobs} files  workers={max(workers, 1)}", flush=True)
        done = 0
        with ThreadPoolExecutor(max_workers=max(workers, 1)) as pool:
            futs = [pool.submit(_one, job) for job in jobs]
            for fut in as_completed(futs):
                try:
                    product, got = fut.result()
                    for name, pair in got.items():
                        site_frames[product][name].append(pair)
                except Exception as exc:  # noqa: BLE001
                    print(f"      FAILED {type(exc).__name__}: {exc}", flush=True)
                done += 1
                if done % 40 == 0 or done == n_jobs:
                    print(f"      {done}/{n_jobs}", flush=True)

    for product, want in want_by_prod.items():
        var = TARGETS[product][1]
        for site in want:
            frames = site_frames.get(product, {}).get(site.name, [])
            path = site_night_dir(out_dir, site, evening) / f"{product}.npz"
            info = _write_product(path, frames, site, product, var) if frames else {"missing": True}
            rec["sites"].setdefault(site.name, {})[product] = info
            if not info.get("missing"):
                print(
                    f"      {site.name}/{product} n={info['n']} {info['size_mb']:.2f} MB",
                    flush=True,
                )

    for site in pending:
        if site.name not in windows:
            continue
        meta = {
            "date": evening.isoformat(),
            "site": site.name,
            "lat": site.lat,
            "lon": site.lon,
            "kind": site.kind,
            "split": site.pretrain_split,
            "night": [
                windows[site.name][0].strftime("%Y-%m-%dT%H:%M:%SZ"),
                windows[site.name][1].strftime("%Y-%m-%dT%H:%M:%SZ"),
            ],
            "products": rec["sites"].get(site.name, {}),
        }
        dest = site_night_dir(out_dir, site, evening)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return rec


def _run_evening(job: tuple) -> tuple[str, dict]:
    """One evening in a child process. Args are plain types so spawn can pickle."""
    iso, i, n, products, dest, site_names, workers, force = job
    print(f"[{i}/{n}] {iso}  pid={os.getpid()}", flush=True)
    rec = fetch_evening(
        dt.date.fromisoformat(iso),
        products=tuple(products),
        out_dir=Path(dest),
        sites=tuple(SITE_BY_NAME[name] for name in site_names),
        workers=int(workers),
        force=bool(force),
    )
    return iso, rec


def pairing_status(out_dir: Path, products: tuple[str, ...], dates: list[dt.date]) -> dict:
    per = {}
    for site in SITES:
        have = [d.isoformat() for d in dates if night_complete(out_dir, site, d, products)]
        per[site.name] = {"n": len(have), "split": site.pretrain_split}
    return {
        "out": str(out_dir),
        "products": list(products),
        "n_dates": len(dates),
        "sites": per,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", default=DEFAULT_START.isoformat())
    p.add_argument("--end", default=DEFAULT_END.isoformat())
    p.add_argument("--products", nargs="+", default=list(MODEL_PRODUCTS), choices=list(TARGETS))
    p.add_argument("--sites", nargs="+", default=[s.name for s in SITES])
    p.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Parallel HTTPS downloads of ABI full-disk files (I/O bound).",
    )
    p.add_argument(
        "--nights-parallel",
        type=int,
        default=2,
        help="Evenings fetched at once as processes (own GIL / HDF5 lock). "
        "Each evening uses --workers download threads.",
    )
    p.add_argument("--max-nights", type=int, default=None)
    p.add_argument(
        "--dates-file",
        type=Path,
        default=None,
        help="ISO dates, one per line. Overrides --start/--end.",
    )
    p.add_argument("--force", action="store_true")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--status", action="store_true")
    args = p.parse_args(argv)

    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    if args.dates_file is not None:
        dates = [
            dt.date.fromisoformat(line.strip())
            for line in Path(args.dates_file).read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    else:
        dates = _dates(start, end)
    products = tuple(args.products)
    sites = tuple(SITE_BY_NAME[n] for n in args.sites)
    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "sites.json").write_text(
        json.dumps(
            [
                {
                    "name": s.name,
                    "lat": s.lat,
                    "lon": s.lon,
                    "kind": s.kind,
                    "split": s.pretrain_split,
                    "note": s.note,
                }
                for s in SITES
            ],
            indent=2,
        )
        + "\n"
    )

    if args.status:
        st = pairing_status(dest, products, dates)
        print(json.dumps(st, indent=2))
        return

    if args.max_nights is not None:
        dates = dates[: int(args.max_nights)]
    print(
        f"Night GOES pretrain fetch → {dest}\n"
        f"  {dates[0]} … {dates[-1]}  ({len(dates)} evenings)\n"
        f"  sites={[s.name for s in sites]}  products={list(products)}  "
        f"workers={args.workers}  nights_parallel={args.nights_parallel}",
        flush=True,
    )
    index = {"name": "goes_pretrain_night", "nights": {}}
    idx_path = dest / INDEX_NAME
    if idx_path.is_file():
        index = json.loads(idx_path.read_text())
    n_ok = n_skip = n_fail = 0
    nights_p = max(1, int(args.nights_parallel))
    jobs = [
        (
            evening.isoformat(),
            i,
            len(dates),
            list(products),
            str(dest),
            [s.name for s in sites],
            int(args.workers),
            bool(args.force),
        )
        for i, evening in enumerate(dates, 1)
    ]
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=nights_p, mp_context=ctx) as pool:
        futs = {pool.submit(_run_evening, job): job[0] for job in jobs}
        for fut in as_completed(futs):
            iso = futs[fut]
            try:
                iso, rec = fut.result()
            except Exception as exc:  # noqa: BLE001
                print(f"  FAIL {iso}: {type(exc).__name__}: {exc}", flush=True)
                n_fail += 1
                continue
            n_skip += int(bool(rec.get("skipped")))
            n_ok += int(not rec.get("skipped") and not rec.get("no_dark"))
            index.setdefault("nights", {})[iso] = {
                k: rec[k] for k in rec if k != "sites"
            }
            idx_path.write_text(json.dumps(index, indent=2) + "\n")
    print(
        f"done: wrote {n_ok}  skipped {n_skip}  failed {n_fail}\n"
        f"  {pairing_status(dest, products, _dates(start, end))}"
    )


if __name__ == "__main__":
    main()
