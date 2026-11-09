"""Name the Zarr chunk objects behind the missing-data blocks of nan_report.csv, and check them on Swift.

For each block (member, field, time chunk) it builds the key of every chunk covering that member
and time chunk, sends a HEAD request for each, and prints the keys that return 404. It then
downloads and decodes every chunk of the block at the levels with missing points (the heights_m
column of the report, or --zchunks) and prints those that fail to download or decode, or that hold
NaN inside the array (the padding of edge chunks is left out of the count).

    python scripts/cloud_botany_find_chunks.py [NAN_REPORT] [--members 1,11] [--max-blocks 5] [--zchunks 5,6]

About 700 HEADs and 36-72 chunk downloads of about 8 MB per block, 4 at a time: a few hundred MB
per block, so run more than a few blocks in a batch job rather than on a login node.
"""
import argparse
import getpass
import itertools
import json
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numcodecs
import numpy as np
import pandas as pd
import xarray as xr

STORE = ("https://swift.dkrz.de/v1/dkrz_b42c6bc6-439f-4efb-9491-db3de70ad9ba/"
         "botany_1536_3D/botany_1536_3D.zarr")  # eurec4a-intake simulations.DALES.botany.dx100m.nx1536.3D


def head(url):
    """HTTP status and object size in bytes (-1 when the server gives none)."""
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, int(r.headers.get("Content-Length", -1))
    except urllib.error.HTTPError as err:
        return err.code, -1


def valid(a, idx, chunks, shape):
    """The part of an edge chunk that lies inside the array."""
    return a[tuple(slice(0, min(c, n - i * c)) for i, c, n in zip(idx, chunks, shape))]


def decode(url, zarray):
    """Download one Zarr v2 chunk and return it as an array."""
    raw = urllib.request.urlopen(url, timeout=120).read()
    comp = zarray.get("compressor") or {}
    if comp.get("id") == "blosc":  # blosc can crash on a truncated buffer: check its header first
        nbytes, cbytes = np.frombuffer(raw[4:16], "<u4")[[0, 2]] if len(raw) >= 16 else (0, -1)
        if cbytes != len(raw):
            raise ValueError(f"blosc header says {cbytes} compressed bytes, object has {len(raw)}")
    if comp:
        raw = numcodecs.get_codec(zarray["compressor"]).decode(raw)
    for f in reversed(zarray.get("filters") or []):
        raw = numcodecs.get_codec(f).decode(raw)
    return np.frombuffer(raw, dtype=zarray["dtype"]).reshape(zarray["chunks"], order=zarray["order"])


def main():
    user = getpass.getuser()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("report", nargs="?", type=Path,
                   default=Path(f"/scratch/{user[0]}/{user}/ISO/Cloud_Botany/nan_report.csv"))
    p.add_argument("--members", default=None, help="comma-separated members to check (default: all)")
    p.add_argument("--max-blocks", type=int, default=5, help="blocks to check, in report order")
    p.add_argument("--zchunks", default=None,
                   help="comma-separated z-chunk indices to decode (default: from the heights_m column)")
    p.add_argument("--out", type=Path, default=None, help="CSV of the keys checked (default: next to the report)")
    p.add_argument("--store", default=STORE, help="Zarr store URL (default: the Botany nx1536 3D store)")
    args = p.parse_args()
    store = args.store

    meta = json.load(urllib.request.urlopen(f"{store}/.zmetadata", timeout=60))["metadata"]
    ds = xr.open_zarr(store, consolidated=True)
    members = ds["member"].values if "member" in ds.coords else np.arange(1, ds.sizes["member"] + 1)

    blocks = pd.read_csv(args.report)
    if args.members:
        blocks = blocks[blocks.member.isin([int(m) for m in args.members.split(",")])]
    blocks = blocks.head(args.max_blocks)
    rows = []
    with ThreadPoolExecutor(16) as pool, ThreadPoolExecutor(4) as getpool:
        for b in blocks.itertuples():
            zarray, zattrs = meta[f"{b.field}/.zarray"], meta[f"{b.field}/.zattrs"]
            dims, chunks, shape = zattrs["_ARRAY_DIMENSIONS"], zarray["chunks"], zarray["shape"]
            sep = zarray.get("dimension_separator", ".")
            fixed = {"member": int(np.flatnonzero(members == b.member)[0]),
                     "time": b.first_index // chunks[dims.index("time")]}
            ranges = [[fixed[d]] if d in fixed else range(-(-n // c)) for d, n, c in zip(dims, shape, chunks)]
            keys = [f"{b.field}/{sep.join(map(str, idx))}" for idx in itertools.product(*ranges)]
            codes, sizes = zip(*pool.map(head, (f"{store}/{k}" for k in keys)))
            missing = [k for k, c in zip(keys, codes) if c == 404]
            other = sorted({c for c in codes if c not in (200, 404)})
            print(f"member {b.member} {b.field} {b.status} times {b.first_index}-{b.last_index}: "
                  f"{len(missing)} of {len(keys)} chunks missing" + (f", other HTTP codes {other}" if other else ""))
            for k in missing[:10]:
                print(f"  {store}/{k}")
            if len(missing) > 10:
                print(f"  ... {len(missing) - 10} more")
            zdim = next(d for d in dims if d.startswith("z"))
            iz = dims.index(zdim)
            if args.zchunks:
                zc = {int(z) for z in args.zchunks.split(",")}
            elif isinstance(getattr(b, "heights_m", None), str) and "-" in b.heights_m:
                lo, hi = map(float, b.heights_m.split("-"))
                z = ds[zdim].values
                zc = {int(i) // chunks[iz] for i in np.flatnonzero((z >= lo - 50) & (z <= hi + 50))}
            else:
                zc = set()
                print("  no heights for this block; pass --zchunks to decode it")
            todo = [(k, idx) for k, idx, c in zip(keys, itertools.product(*ranges), codes)
                    if c == 200 and idx[iz] in zc]

            def scan(item):
                k, idx = item
                try:
                    return int(np.isnan(valid(decode(f"{store}/{k}", zarray), idx, chunks, shape)).sum()), ""
                except Exception as err:  # report any download or decode failure
                    return -1, repr(err)[:200]

            result = dict(zip((k for k, _ in todo), getpool.map(scan, todo)))
            bad = {k: r for k, r in result.items() if r[0] != 0}
            print(f"  decoded {len(result)} chunks at z-chunks {sorted(zc)}: {len(bad)} with NaN or errors")
            for k, (n, err) in sorted(bad.items()):
                print(f"  {store}/{k}  " + (f"ERROR {err}" if err else f"{n} NaN"))
            nnan = {k: r[0] for k, r in result.items()}
            errors = {k: r[1] for k, r in result.items() if r[1]}
            rows += [dict(member=b.member, field=b.field, first_index=b.first_index, key=k, http=c, bytes=n,
                          nnan=nnan.get(k, ""), error=errors.get(k, "")) for k, c, n in zip(keys, codes, sizes)]
    out = args.out or args.report.with_name("missing_chunks.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"dims {dims}, chunks {chunks}; all keys checked in {out}")


if __name__ == "__main__":
    main()
