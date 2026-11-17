"""Copy the surviving per-time output of a Cloud Botany run to a new directory, leaving out bad times.

Copies SRC/EXP_<m>/CB_ISO_EXP_<m>_TIME_<t>.nc to DST/EXP_<m>/, except the (member, time) pairs of
the masked blocks in SRC/nan_report.csv, which had unreadable chunks. A rerun of
cloud_botany_iso.py into DST (e.g. with --refs) then skips the copied times and redoes the rest.
Each member's nan_log.csv is copied as nan_log_first_run.csv, so the rerun starts its own log.

    python scripts/cloud_botany_salvage.py SRC DST [--dry-run]
"""
import argparse
import re
import shutil
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src", type=Path)
    p.add_argument("dst", type=Path)
    p.add_argument("--dry-run", action="store_true", help="only count what would be copied")
    args = p.parse_args()

    report = pd.read_csv(args.src / "nan_report.csv")
    bad = {(int(r.member), t) for r in report[report.status == "masked"].itertuples()
           for t in range(int(r.first_index), int(r.last_index) + 1)}
    ncopy = nskip = 0
    for d in sorted(args.src.glob("EXP_*")):
        out = args.dst / d.name
        for f in sorted(d.glob("CB_ISO_EXP_*_TIME_*.nc")):
            m, t = map(int, re.findall(r"\d+", f.name)[-2:])
            if (m, t) in bad:
                nskip += 1
                continue
            ncopy += 1
            if not args.dry_run and not (out / f.name).exists():
                out.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, out / f.name)
        if not args.dry_run and (d / "nan_log.csv").exists():
            out.mkdir(parents=True, exist_ok=True)
            shutil.copy2(d / "nan_log.csv", out / "nan_log_first_run.csv")
    print(f"{'would copy' if args.dry_run else 'copied'} {ncopy} files, left out {nskip} with bad chunks")


if __name__ == "__main__":
    main()
