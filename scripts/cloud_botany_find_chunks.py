"""Name the Zarr chunk objects behind the missing-data blocks of nan_report.csv, and check them on Swift.

For each block (member, field, time chunk) it builds the key of every chunk covering that member
and time chunk, sends a HEAD request for each, and prints the keys that return 404. When the chunks
all exist, the NaNs are stored in the data: the script then downloads and decodes the --decode
smallest objects of the block (a chunk of NaN compresses far better than one of model output) and
prints their NaN counts, which names the chunks that hold the NaNs.

    python scripts/cloud_botany_find_chunks.py [NAN_REPORT] [--members 1,11] [--max-blocks 5] [--decode 3]

One HEAD per chunk (about 700 per block), 16 at a time, plus a few chunk downloads of a few MB:
light enough for a login node.
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


def decode(url, zarray):
    """Download one Zarr v2 chunk and return it as an array."""
    raw = urllib.request.urlopen(url, timeout=120).read()
    if zarray.get("compressor"):
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
    p.add_argument("--decode", type=int, default=3, help="smallest chunks of each block to decode")
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
    with ThreadPoolExecutor(16) as pool:
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
            nnan = {}
            present = [(n, k) for k, c, n in zip(keys, codes, sizes) if c == 200 and n >= 0]
            if present:
                med = np.median([n for n, _ in present])
                print(f"  median object size {med / 1e6:.2f} MB; smallest:")
                for n, k in sorted(present)[:args.decode]:
                    a = decode(f"{store}/{k}", zarray)
                    nnan[k] = int(np.isnan(a).sum())
                    print(f"  {store}/{k}  {n / 1e6:.3f} MB, {nnan[k]} of {a.size} NaN")
            rows += [dict(member=b.member, field=b.field, first_index=b.first_index, key=k, http=c, bytes=n,
                          nnan=nnan.get(k, "")) for k, c, n in zip(keys, codes, sizes)]
    out = args.out or args.report.with_name("missing_chunks.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"dims {dims}, chunks {chunks}; all keys checked in {out}")


if __name__ == "__main__":
    main()
