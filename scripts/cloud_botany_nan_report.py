"""List the blocks of missing data found by cloud_botany_iso.py, for a report to the data providers.

Reads SAVE_DIR/EXP_*/nan_log.csv, keeps the last read of each (member, time, field), and groups
the times whose last read was `masked` (NaN at the same points on two reads) or `no_data` (NaN
everywhere) into runs of consecutive time indices. For masked blocks the heights with missing
points come from the AREA deficit in the first output file of the block; that deficit is the union
over the five fields, so it can include levels of another field masked at the same time.

    python scripts/cloud_botany_nan_report.py [SAVE_DIR]        # writes SAVE_DIR/nan_report.{csv,md}

Light enough for a login node: it reads the CSV logs and one small netCDF per masked block.
"""
import argparse
import getpass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

CHUNK_POINTS = 8 * 256 * 256  # one (z, y, x) Zarr chunk of the 3D fields at one time


def last_reads(save_dir):
    logs = sorted(save_dir.glob("EXP_*/nan_log.csv"))
    if not logs:
        raise SystemExit(f"no nan_log.csv under {save_dir}")
    d = pd.concat(pd.read_csv(f) for f in logs)
    # `attempt` restarts at 1 when a task is resubmitted; keep file order within equal attempts
    d = d.reset_index(drop=True).reset_index(names="row")
    return d.sort_values(["attempt", "row"]).groupby(["member", "time_index", "field"]).tail(1)


def blocks(last):
    bad = last[last.status.isin(["masked", "no_data"])].sort_values(["member", "field", "time_index"])
    out = []
    for (m, fld, st), g in bad.groupby(["member", "field", "status"]):
        run = (g.time_index.diff() != 1).cumsum()
        for _, b in g.groupby(run):
            out.append(dict(member=m, field=fld, status=st,
                            first_index=b.time_index.min(), last_index=b.time_index.max(),
                            n_times=len(b), first_time=b.time.iloc[0], last_time=b.time.iloc[-1],
                            nan_points=int(b.nnan.max())))
    return pd.DataFrame(out)


def missing_levels(save_dir, member, it):
    f = save_dir / f"EXP_{member:03d}" / f"CB_ISO_EXP_{member:03d}_TIME_{it:02d}.nc"
    if not f.exists():
        return "", np.nan
    with xr.open_dataset(f) as ds:
        a = ds.AREA.isel(time=0).sum("theta_e")
        full = float(a.max())
        low = a < 0.999 * full
        if not low.any():
            return "", 0.0
        z = ds.height.values[low.values]
        return f"{z.min():.0f}-{z.max():.0f}", float(1 - a[low].mean() / full)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    user = getpass.getuser()
    p.add_argument("save_dir", nargs="?", type=Path,
                   default=Path(f"/scratch/{user[0]}/{user}/ISO/Cloud_Botany"))
    args = p.parse_args()

    last = last_reads(args.save_dir)
    b = blocks(last)
    if b.empty:
        print("no masked or no_data reads")
        return
    b["chunks_per_time"] = (b.nan_points / CHUNK_POINTS).round(2)
    lev = [missing_levels(args.save_dir, r.member, r.first_index) if r.status == "masked" else ("", np.nan)
           for r in b.itertuples()]
    b["heights_m"] = [x[0] for x in lev]
    b["area_missing_at_those_levels"] = [round(x[1], 3) for x in lev]

    members = sorted(last.member.unique())
    print(f"{len(members)} members logged ({members[0]}-{members[-1]}), "
          f"{last.groupby('member').time_index.nunique().sum()} snapshots")
    print(b.groupby("status").size().to_string())
    print(b.to_string(index=False))
    b.to_csv(args.save_dir / "nan_report.csv", index=False)
    try:
        (args.save_dir / "nan_report.md").write_text(b.to_markdown(index=False) + "\n")
    except ImportError:  # to_markdown needs tabulate
        pass
    print(f"wrote {args.save_dir / 'nan_report.csv'}")


if __name__ == "__main__":
    main()
