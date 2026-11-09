"""Name the Zarr chunk objects behind the missing-data blocks of nan_report.csv, and check them on Swift.

For each block (member, field, time chunk) it builds the key of every chunk covering that member
and time chunk, sends a HEAD request for each, and prints the keys that return 404. A block whose
chunks all exist holds NaN in the data itself rather than missing objects.

    python scripts/cloud_botany_find_chunks.py [NAN_REPORT] [--max-blocks 5] [--out missing_chunks.csv]

One HEAD per chunk (792 per block for the 3D fields), 16 at a time: light enough for a login node.
"""
import argparse
import getpass
import itertools
import json
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

STORE = ("https://swift.dkrz.de/v1/dkrz_b42c6bc6-439f-4efb-9491-db3de70ad9ba/"
         "botany_1536_3D/botany_1536_3D.zarr")  # eurec4a-intake simulations.DALES.botany.dx100m.nx1536.3D


def status(url):
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status
    except urllib.error.HTTPError as err:
        return err.code


def main():
    user = getpass.getuser()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("report", nargs="?", type=Path,
                   default=Path(f"/scratch/{user[0]}/{user}/ISO/Cloud_Botany/nan_report.csv"))
    p.add_argument("--max-blocks", type=int, default=5, help="blocks to check, in report order")
    p.add_argument("--out", type=Path, default=None, help="CSV of the keys checked (default: next to the report)")
    p.add_argument("--store", default=STORE, help="Zarr store URL (default: the Botany nx1536 3D store)")
    args = p.parse_args()
    store = args.store

    meta = json.load(urllib.request.urlopen(f"{store}/.zmetadata", timeout=60))["metadata"]
    ds = xr.open_zarr(store, consolidated=True)
    members = ds["member"].values if "member" in ds.coords else np.arange(1, ds.sizes["member"] + 1)

    blocks = pd.read_csv(args.report).head(args.max_blocks)
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
            codes = list(pool.map(status, (f"{store}/{k}" for k in keys)))
            missing = [k for k, c in zip(keys, codes) if c == 404]
            other = sorted({c for c in codes if c not in (200, 404)})
            print(f"member {b.member} {b.field} {b.status} times {b.first_index}-{b.last_index}: "
                  f"{len(missing)} of {len(keys)} chunks missing" + (f", other HTTP codes {other}" if other else ""))
            for k in missing[:10]:
                print(f"  {store}/{k}")
            if len(missing) > 10:
                print(f"  ... {len(missing) - 10} more")
            rows += [dict(member=b.member, field=b.field, first_index=b.first_index, key=k, http=c)
                     for k, c in zip(keys, codes)]
    out = args.out or args.report.with_name("missing_chunks.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"dims {dims}, chunks {chunks}; all keys checked in {out}")


if __name__ == "__main__":
    main()
