"""Isentropic integrals (conditional sums) of a SAM run, written to one netCDF file.

Usage:
    python sam_iso_integrals.py RUN_DIR RUN_NAME [-o OUTPUT] [--exp ID] [--times START:STOP:STEP]

RUN_DIR holds SAM's OUT_3D and OUT_STAT directories; RUN_NAME is <case>_<caseid>, so that the
3D snapshots are OUT_3D/<RUN_NAME>_<nproc>_<step>.nc and the statistics file is
OUT_STAT/<RUN_NAME>.nc (RHO is used as the density profile; without it the density is
computed from each snapshot).

For each snapshot and for both theta_e and frozen theta_e, the output has the conditional sums
MF (rho*w), AREA (number of points) and ATH (the coordinate itself), and the same sums over the
cloudy points (QN > --qn-cloud g/kg) as MF_CLD and AREA_CLD, with dimensions (time, z, coord).
HIST_JOINT(theta_e, theta_ef) is the joint distribution of the two coordinates summed over all
processed snapshots. Read the output with notebooks/isentropic_diagnostics.ipynb.

Example (every 4th snapshot):
    python sam_iso_integrals.py /scratch/op13/SAM-Torch/RCE_REF RCE_REF_300K --times ::4
"""

import argparse
import re
import time as clock
from pathlib import Path

import numpy as np
import xarray as xr

from isentropic import sam

VAR = dict(w="W", T="TABS", p="p", pp="PP", qv="QV", qn="QN", qi="QI")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path, help="SAM run directory (with OUT_3D and OUT_STAT)")
    ap.add_argument("run_name", help="SAM run name <case>_<caseid>, e.g. RCE_REF_300K")
    ap.add_argument("-o", "--output", type=Path, help="output file (default: iso_sums_<exp>.nc)")
    ap.add_argument("--exp", help="experiment ID stored in the output (default: name of RUN_DIR)")
    ap.add_argument("--times", default="::", help="snapshots to process, as a Python slice START:STOP:STEP")
    ap.add_argument("--qn-cloud", type=float, default=sam.QN_CLOUD, help="cloud threshold on QN (g/kg)")
    ap.add_argument("--theta-e-bins", default="300,370,141", help="th_min,th_max,nbin for theta_e (K)")
    ap.add_argument("--theta-ef-bins", default="300,390,181", help="th_min,th_max,nbin for frozen theta_e (K)")
    ap.add_argument("--no-stat", action="store_true", help="compute the density instead of reading RHO")
    return ap.parse_args()


def parse_bins(s):
    th_min, th_max, nbin = s.split(",")
    return dict(th_min=float(th_min), th_max=float(th_max), nbin=int(nbin))


def run_files(run_dir, run_name):
    """3D files <run_name>_<nproc>_<step>.nc sorted by name, and the statistics file (or None)."""
    pattern = re.compile(rf"{re.escape(run_name)}_\d+_\d+\.nc")
    files = sorted(f for f in (run_dir / "OUT_3D").glob(f"{run_name}_*.nc") if pattern.fullmatch(f.name))
    stat = run_dir / "OUT_STAT" / f"{run_name}.nc"
    if not stat.exists():  # look for it anywhere in the run directory
        stat = next(iter(sorted(run_dir.rglob(f"{run_name}.nc"))), None)
    return files, stat


def snapshots(files, time_slice):
    """(file, time index) of every snapshot, sorted in time; a time in several files is used once."""
    snaps = {}
    for f in files:
        with xr.open_dataset(f) as ds:
            for it, t in enumerate(ds["time"].values):
                if t in snaps:
                    print(f"time {t} in {snaps[t][0].name} and {f.name}, using {f.name}")
                snaps[t] = (f, it)
    return [snaps[t] for t in sorted(snaps)][time_slice]


def main():
    args = parse_args()
    exp = args.exp or args.run_dir.name
    output = args.output or Path(f"iso_sums_{exp}.nc")
    bins = {"theta_e": parse_bins(args.theta_e_bins), "theta_ef": parse_bins(args.theta_ef_bins)}
    centers = {name: b["th_min"] + (b["th_max"] - b["th_min"]) / (b["nbin"] - 1) * np.arange(b["nbin"])
               for name, b in bins.items()}

    files, stat = run_files(args.run_dir, args.run_name)
    if not files:
        raise SystemExit(f"no 3D files {args.run_name}_<nproc>_<step>.nc in {args.run_dir / 'OUT_3D'}")
    snaps = snapshots(files, slice(*(int(s) if s else None for s in args.times.split(":"))))
    rho_ref = None
    if stat is not None and not args.no_stat:
        with xr.open_dataset(stat) as st:
            rho_ref = st["RHO"].load() if "RHO" in st else None
    print(f"{exp}: {len(files)} 3D files, {len(snaps)} snapshots to process")
    print(f"density: {'RHO from ' + str(stat) if rho_ref is not None else 'computed from each snapshot'}")

    with xr.open_dataset(snaps[0][0]) as ds:
        z = ds["z"].values
    if rho_ref is not None and rho_ref.sizes["z"] != len(z):
        raise SystemExit("statistics file and 3D output have different vertical levels")
    nt = len(snaps)
    out = {f"{v}_{name}": np.zeros((nt, len(z), b["nbin"]))
           for name, b in bins.items() for v in ("MF", "AREA", "ATH", "MF_CLD", "AREA_CLD")}
    rho_out = np.zeros((nt, len(z)))
    cld_frac = np.zeros((nt, len(z)))
    times = np.zeros(nt)
    edges = {name: np.linspace(b["th_min"], b["th_max"], b["nbin"] + 1) for name, b in bins.items()}
    hist = np.zeros((bins["theta_e"]["nbin"], bins["theta_ef"]["nbin"]))

    t0 = clock.time()
    for k, (f, it) in enumerate(snaps):
        with xr.open_dataset(f) as ds:
            snap = ds.isel(time=it)
            get = lambda name: snap[name].values.astype(float)
            times[k] = float(snap["time"])
            rho = None if rho_ref is None else rho_ref.sel(time=times[k], method="nearest").values
            rhow, coords, rho_out[k], cloudy = sam.snapshot(
                get(VAR["w"]), get(VAR["T"]), get(VAR["qv"]), get(VAR["qn"]), get(VAR["p"]), get(VAR["pp"]),
                qi=get(VAR["qi"]) if VAR["qi"] in snap else None, rho=rho, qn_cloud=args.qn_cloud)
        for key, val in sam.conditional_sums(rhow, coords, bins, cloudy).items():
            out[key][k] = val
        cld_frac[k] = cloudy.mean((1, 2))
        hist += np.histogram2d(coords["theta_e"].ravel(), coords["theta_ef"].ravel(),
                               bins=[edges["theta_e"], edges["theta_ef"]])[0]
        if k % 20 == 0 or k == nt - 1:
            print(f"snapshot {k + 1}/{nt} ({f.name}), {clock.time() - t0:.0f} s")

    data = {key: (("time", "z", name), val) for name in bins for key, val in out.items() if key.endswith("_" + name)}
    data["RHO"] = (("time", "z"), rho_out)
    data["CLD_FRAC"] = (("time", "z"), cld_frac)
    data["HIST_JOINT"] = (("theta_e", "theta_ef"), hist)
    ds = xr.Dataset(data, coords={"time": times, "z": z, **centers})
    ds["time"].attrs.update(units="d")
    ds["z"].attrs.update(units="m")
    for name in bins:
        ds[f"MF_{name}"].attrs.update(long_name="conditional sum of rho*w over grid points", units="kg m-2 s-1")
        ds[f"AREA_{name}"].attrs.update(long_name="conditional sum of grid points")
        ds[f"ATH_{name}"].attrs.update(long_name=f"conditional sum of {name}", units="K")
        ds[f"MF_CLD_{name}"].attrs.update(long_name="MF over cloudy points", units="kg m-2 s-1")
        ds[f"AREA_CLD_{name}"].attrs.update(long_name="AREA over cloudy points")
    ds["RHO"].attrs.update(long_name="density profile", units="kg m-3")
    ds["CLD_FRAC"].attrs.update(long_name=f"fraction of points with QN > {args.qn_cloud} g/kg")
    ds["HIST_JOINT"].attrs.update(long_name="joint distribution of theta_e and theta_ef, summed over time")
    ds.attrs.update(experiment=exp, run_dir=str(args.run_dir), run_name=args.run_name, qn_cloud=args.qn_cloud,
                    density="RHO from " + str(stat) if rho_ref is not None else "computed",
                    **{f"bins_{name}": f"{b['th_min']},{b['th_max']},{b['nbin']}" for name, b in bins.items()})
    ds.to_netcdf(output)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
