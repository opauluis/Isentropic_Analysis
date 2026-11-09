"""Name the Zarr chunk objects behind the missing-data blocks of nan_report.csv, and check them on Swift.

For each block of the report (member, field, run of times) it takes every chunk covering that
member and those time chunks at the levels with missing points (the heights_m column, or
--zchunks), and reads it from Swift. With --check range (the default) it reads only the first
16 bytes of each chunk, which is enough to catch objects that are listed but cannot be downloaded
(HEAD 200, GET 404: zarr fills those with NaN); with --check full it downloads and decodes each
chunk and also counts NaN stored inside the array (the padding of edge chunks is left out).
Unreadable chunks are then described by a HEAD request (size, ETag, Last-Modified).

    python scripts/cloud_botany_find_chunks.py [NAN_REPORT] [--members 1,11] [--max-blocks 0]
        [--status masked] [--zchunks 6,7] [--check range|full] [--out bad_chunks.csv]

--check range costs one small request per chunk (about 100 per block); --check full downloads
about 8 MB per chunk. scripts/cloud_botany_find_chunks.sbatch runs every masked block.
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
    """HTTP status and the headers that describe the object."""
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            h = r.headers
            return r.status, h.get("Content-Length", ""), h.get("Etag", ""), h.get("Last-Modified", "")
    except urllib.error.HTTPError as err:
        return err.code, "", "", ""


def get(url, nbytes=None, retries=3):
    """GET the object, or only its first nbytes; errors other than 404 are retried."""
    req = urllib.request.Request(url, headers={"Range": f"bytes=0-{nbytes - 1}"} if nbytes else {})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as err:
            if err.code == 404 or attempt == retries - 1:
                return err.code, b""
        except OSError:
            if attempt == retries - 1:
                raise


def valid(a, idx, chunks, shape):
    """The part of an edge chunk that lies inside the array."""
    return a[tuple(slice(0, min(c, n - i * c)) for i, c, n in zip(idx, chunks, shape))]


def decode(raw, zarray):
    """Decode one Zarr v2 chunk."""
    comp = zarray.get("compressor") or {}
    if comp.get("id") == "blosc":  # blosc can crash on a truncated buffer: check its header first
        cbytes = int(np.frombuffer(raw[12:16], "<u4")[0]) if len(raw) >= 16 else -1
        if cbytes != len(raw):
            raise ValueError(f"blosc header says {cbytes} compressed bytes, object has {len(raw)}")
    if comp:
        raw = numcodecs.get_codec(comp).decode(raw)
    for f in reversed(zarray.get("filters") or []):
        raw = numcodecs.get_codec(f).decode(raw)
    return np.frombuffer(raw, dtype=zarray["dtype"]).reshape(zarray["chunks"], order=zarray["order"])


def check(url, idx, zarray, full):
    """(GET status, NaN count inside the array or -1, error text) for one chunk."""
    status, raw = get(url, None if full else 16)
    if status not in (200, 206):
        return status, -1, f"GET {status}"
    if not full:
        return status, -1, ""
    try:
        a = valid(decode(raw, zarray), idx, zarray["chunks"], zarray["shape"])
        return status, int(np.isnan(a).sum()), ""
    except Exception as err:  # a damaged object: report it rather than stop
        return status, -1, repr(err)[:200]


def main():
    user = getpass.getuser()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("report", nargs="?", type=Path,
                   default=Path(f"/scratch/{user[0]}/{user}/ISO/Cloud_Botany/nan_report.csv"))
    p.add_argument("--members", default=None, help="comma-separated members to check (default: all)")
    p.add_argument("--status", default="masked", help="comma-separated block statuses to check")
    p.add_argument("--max-blocks", type=int, default=5, help="blocks to check, in report order; 0 for all")
    p.add_argument("--zchunks", default=None,
                   help="comma-separated z-chunk indices to read (default: from the heights_m column)")
    p.add_argument("--check", choices=["range", "full"], default="range")
    p.add_argument("--workers", type=int, default=8, help="requests in flight")
    p.add_argument("--out", type=Path, default=None,
                   help="CSV of the unreadable or NaN chunks (default: bad_chunks.csv next to the report)")
    p.add_argument("--store", default=STORE, help="Zarr store URL (default: the Botany nx1536 3D store)")
    args = p.parse_args()
    store, full = args.store, args.check == "full"

    meta = json.load(urllib.request.urlopen(f"{store}/.zmetadata", timeout=60))["metadata"]
    ds = xr.open_zarr(store, consolidated=True)
    members = ds["member"].values if "member" in ds.coords else np.arange(1, ds.sizes["member"] + 1)

    blocks = pd.read_csv(args.report)
    blocks = blocks[blocks.status.isin(args.status.split(","))]
    if args.members:
        blocks = blocks[blocks.member.isin([int(m) for m in args.members.split(",")])]
    if args.max_blocks:
        blocks = blocks.head(args.max_blocks)
    print(f"{len(blocks)} blocks, --check {args.check}", flush=True)

    bad, nread = [], 0
    with ThreadPoolExecutor(args.workers) as pool:
        for b in blocks.itertuples():
            zarray, zattrs = meta[f"{b.field}/.zarray"], meta[f"{b.field}/.zattrs"]
            dims, chunks, shape = zattrs["_ARRAY_DIMENSIONS"], zarray["chunks"], zarray["shape"]
            sep = zarray.get("dimension_separator", ".")
            zdim = next(d for d in dims if d.startswith("z"))
            ct, cz = chunks[dims.index("time")], chunks[dims.index(zdim)]
            if args.zchunks:
                zc = sorted({int(z) for z in args.zchunks.split(",")})
            elif isinstance(getattr(b, "heights_m", None), str) and "-" in b.heights_m:
                lo, hi = map(float, b.heights_m.split("-"))
                z = ds[zdim].values  # heights_m are zt values; 50 m of slack covers zm fields
                zc = sorted({int(i) // cz for i in np.flatnonzero((z >= lo - 50) & (z <= hi + 50))})
            else:
                print(f"member {b.member} {b.field} times {b.first_index}-{b.last_index}: "
                      "no heights in the report, skipped (pass --zchunks)", flush=True)
                continue
            fixed = {"member": [int(np.flatnonzero(members == b.member)[0])],
                     "time": range(b.first_index // ct, b.last_index // ct + 1), zdim: zc}
            ranges = [fixed.get(d, range(-(-n // c))) for d, n, c in zip(dims, shape, chunks)]
            idxs = list(itertools.product(*ranges))
            keys = [f"{b.field}/{sep.join(map(str, i))}" for i in idxs]
            res = list(pool.map(lambda ki: check(f"{store}/{ki[0]}", ki[1], zarray, full), zip(keys, idxs)))
            nread += len(keys)
            found = [(k, r) for k, r in zip(keys, res) if r[2] or r[1] > 0]
            print(f"member {b.member} {b.field} times {b.first_index}-{b.last_index}, z-chunks {zc}: "
                  f"{len(found)} of {len(keys)} chunks bad", flush=True)
            for k, (status, nnan, err) in found:
                hstatus, size, etag, modified = head(f"{store}/{k}")
                print(f"  {store}/{k}  {err or f'{nnan} NaN'}  (HEAD {hstatus}, {size} B)", flush=True)
                bad.append(dict(member=b.member, field=b.field, first_index=b.first_index,
                                last_index=b.last_index, key=k, url=f"{store}/{k}", get=status, nnan=nnan,
                                error=err, head=hstatus, bytes=size, etag=etag, last_modified=modified))
    out = args.out or args.report.with_name("bad_chunks.csv")
    pd.DataFrame(bad).to_csv(out, index=False)
    print(f"{len(bad)} bad chunks out of {nread} read; written to {out}")


if __name__ == "__main__":
    main()
