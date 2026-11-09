"""Merge the per-time output of cloud_botany_iso.py into one netCDF file per member, off scratch.

Reads SAVE_DIR/EXP_<m>/CB_ISO_EXP_<m>_TIME_*.nc, concatenates them along time, turns the per-file
attributes time_index, masked_points and edge_bin_area_fraction into variables on time, and writes
WORK_DIR/CB_ISO_EXP_<m>.nc (zlib-compressed) with a copy of the member's nan_log.csv next to it.
The merged file is read back and compared with the inputs before the script reports success;
the scratch files are left in place.

    python scripts/cloud_botany_merge.py [MEMBERS] [--save-dir SAVE_DIR] [--work-dir WORK_DIR] [--force]

MEMBERS is a comma list or a range such as 1-103 (default: every EXP_* directory). Times with no
data (member 11, the 48 h runs after 48 h) have no input file and are absent from the merged time
axis; the time_index variable gives the position of each kept time in the 60 hourly outputs.
"""
import argparse
import getpass
import shutil
from pathlib import Path

import numpy as np
import xarray as xr

PER_TIME_ATTRS = {
    "time_index": ("index of the time in the hourly 3D output", "1"),
    "masked_points": ("grid points masked because a field was NaN", "1"),
    "edge_bin_area_fraction": ("fraction of the area in the first and last theta_e bins", "1"),
}


def parse_members(text, save_dir):
    if not text:
        return sorted(int(p.name[4:]) for p in save_dir.glob("EXP_*") if p.is_dir())
    out = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out += range(int(lo), int(hi or lo) + 1)
    return out


def merge(member, save_dir, work_dir, force):
    src = save_dir / f"EXP_{member:03d}"
    files = sorted(src.glob(f"CB_ISO_EXP_{member:03d}_TIME_*.nc"))
    out = work_dir / f"CB_ISO_EXP_{member:03d}.nc"
    if not files:
        return f"member {member}: no files in {src}, skipped"
    if out.exists() and not force:
        return f"member {member}: {out.name} exists, skipped (--force to redo)"

    parts = []
    for f in files:
        with xr.open_dataset(f) as d:
            d = d.load()
        for name, (long_name, units) in PER_TIME_ATTRS.items():
            d[name] = ("time", [d.attrs.pop(name)], {"long_name": long_name, "units": units})
        parts.append(d)
    ds = xr.concat(parts, "time", combine_attrs="drop_conflicts").sortby("time")
    ds.attrs["source_files"] = f"{len(files)} files {src}/CB_ISO_EXP_{member:03d}_TIME_*.nc"
    enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    enc["time"] = {"units": "hours since 2020-02-01 00:00:00", "dtype": "f8"}  # same axis for every member
    tmp = out.with_suffix(".nc.tmp")
    ds.to_netcdf(tmp, encoding=enc)

    with xr.open_dataset(tmp) as back:  # read back and compare before replacing any earlier file
        same = back.sizes["time"] == len(files) and all(
            np.array_equal(back[v].values, ds[v].values, equal_nan=True) for v in ds.data_vars)
    if not same:
        tmp.unlink()
        raise SystemExit(f"member {member}: merged file does not match the inputs; nothing written")
    tmp.replace(out)
    log = src / "nan_log.csv"
    if log.exists():
        shutil.copy2(log, work_dir / f"CB_ISO_EXP_{member:03d}_nan_log.csv")
    return (f"member {member}: {len(files)} times -> {out} "
            f"({out.stat().st_size / 1e6:.0f} MB, time_index {int(ds.time_index.min())}-{int(ds.time_index.max())})")


def main():
    user = getpass.getuser()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("members", nargs="?", default=None)
    p.add_argument("--save-dir", type=Path, default=Path(f"/scratch/{user[0]}/{user}/ISO/Cloud_Botany"))
    p.add_argument("--work-dir", type=Path, default=Path(f"/work/mh1569/{user}/cloud_botany_iso"))
    p.add_argument("--force", action="store_true", help="redo members whose merged file exists")
    args = p.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    for name in ("nan_report.csv", "bad_chunks.csv"):  # the run-wide reports, if present
        if (args.save_dir / name).exists():
            shutil.copy2(args.save_dir / name, args.work_dir / name)
    for m in parse_members(args.members, args.save_dir):
        print(merge(m, args.save_dir, args.work_dir, args.force), flush=True)


if __name__ == "__main__":
    main()
