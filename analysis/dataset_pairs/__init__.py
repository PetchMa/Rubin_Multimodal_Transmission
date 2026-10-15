"""Portable all-sky training pairs: 1 h camera + 1 h GOES + +20 min target.

Each Parquet file is one self-contained bundle. Arrays travel as little-endian
binary blobs plus shape metadata so any language can unpack them.

See ``README.md`` and ``python -m analysis.dataset_pairs.pack``.
"""

from .pack import (
    DEFAULT_OUT,
    PROTOCOL,
    decode_sample,
    iter_samples,
    load_table,
    pack_night,
    pairs_dir,
    write_bundle,
)

__all__ = [
    "DEFAULT_OUT",
    "PROTOCOL",
    "decode_sample",
    "iter_samples",
    "load_table",
    "pack_night",
    "pairs_dir",
    "write_bundle",
]
