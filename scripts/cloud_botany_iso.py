"""Isentropic integrals (conditional sums in z and theta_e) of one Cloud Botany member.

Usage:
    python cloud_botany_iso.py MEMBER [TIMES] [SAVE_DIR] [--domain nx1536] [--theta-e-bins 300,350,101]

MEMBER is the ensemble member (1-103). TIMES is one index or a Python slice START:STOP:STEP of the
hourly 3D output (60 times, 2020-02-01 01:00 to 2020-02-03 12:00); the default is all of them.
Each time is written to SAVE_DIR/EXP_<member>/CB_ISO_EXP_<member>_TIME_<time>.nc; times whose file
already exists are skipped, so a job that stopped can be resubmitted as is.

The data is read from the DKRZ Swift object store through the EUREC4A intake catalog (needs
intake, intake-xarray, jinja2, zarr and aiohttp). theta_e is computed at p_ref = 1000 hPa from
theta_l and q_t, assuming the parcel is unsaturated there. The density is DALES' base state rhobf,
taken at the same time as the 3D snapshot. Points where theta_e, w, q_l or q_r are not finite are
left out. Values outside the theta_e bins are put in the first/last bin (see isentropic.binning).

Runs 7, 15, 38, 39 and 47 crashed and runs 11, 14, 43, 46 and 87 end after 48 h (Jansson et al.
2023, JAMES, doi:10.1029/2023MS003796, Sec. 6.4): the crashed runs are refused unless --force is
given, and times with no data are skipped.

Example (member 1, all times):
    python cloud_botany_iso.py 1 :: /scratch/b/$USER/ISO/Cloud_Botany
"""

import argparse
import getpass
import time as clock
from pathlib import Path

import numpy as np
import xarray as xr
import intake

from isentropic import binning
from isentropic import thermo

CATALOG = "https://raw.githubusercontent.com/eurec4a/eurec4a-intake/master/catalog.yml"
CRASHED = {7, 15, 38, 39, 47}
P_REF = 1.0e5


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("member", type=int, help="ensemble member, 1-103")
    ap.add_argument("times", nargs="?", default="::", help="hourly 3D time index, or slice START:STOP:STEP")
    user = getpass.getuser()
    ap.add_argument("save_dir", nargs="?", type=Path, default=Path(f"/scratch/{user[0]}/{user}/ISO/Cloud_Botany"),
                    help="output directory (default: /scratch/<u>/$USER/ISO/Cloud_Botany)")
    ap.add_argument("--domain", default="nx1536", choices=["nx1536", "nx96"], help="catalog entry")
    ap.add_argument("--theta-e-bins", default="300,350,101", help="th_min,th_max,nbin (K)")
    ap.add_argument("--retries", type=int, default=5, help="attempts to read each field from Swift")
    ap.add_argument("--force", action="store_true", help="process a member that crashed")
    return ap.parse_args()


def parse_times(s, nt):
    if ":" not in s:
        return [int(s)]
    return list(range(nt))[slice(*(int(v) if v else None for v in s.split(":")))]


def load_field(entry, name, sel, retries, backoff=2.0):
    """One 3D field as a numpy array (z, y, x), or None if the snapshot has no data.

    Reading from Swift sometimes drops chunks, which shows up as NaNs: the dataset is then
    reopened and read again. A field that is entirely NaN is missing (run ended), not dropped.
    """
    for attempt in range(1, retries + 1):
        try:
            a = entry.to_dask()[name].sel(sel).values
            nnan = np.isnan(a).sum()
            if nnan == a.size:
                print(f"  {name}: no data")
                return None
            # qt tolerates a few NaNs, and ql/qr any; the points are masked out later
            if nnan == 0 or name in ("ql", "qr") or (name == "qt" and nnan <= 100):
                if nnan:
                    print(f"  {name}: {nnan} NaNs, masked")
                return a
            print(f"  {name}: {nnan} NaNs on attempt {attempt}/{retries}")
        except Exception as err:
            print(f"  {name}: read error on attempt {attempt}/{retries}: {err}")
        if attempt < retries:
            clock.sleep(backoff ** attempt)
    raise RuntimeError(f"no valid '{name}' after {retries} attempts")


def iso_sums(thl, qt, ql, qr, w, rho, area_cell, bins):
    """Conditional sums at every level; returns (nfield, nz, nbin) sums, (nz, nbin) area, masked count."""
    nz = thl.shape[0]
    out = np.zeros((8, nz, bins["nbin"]))
    area = np.zeros((nz, bins["nbin"]))
    nmask = 0
    for k in range(nz):  # level by level, so that only the 3D inputs are held in memory
        qtk = np.maximum(qt[k], 1e-10)
        the = thermo.equivalent_potential_temperature(P_REF, thl[k], qtk / (1.0 - qtk), 0.0, 0.0)
        rhow = rho[k] * 0.5 * (w[k] + w[k + 1])  # w is on the half levels zm
        mask = ~(np.isfinite(the) & np.isfinite(rhow) & np.isfinite(ql[k]) & np.isfinite(qr[k]))
        nmask += mask.sum()
        qlk = np.where(mask, 0.0, np.maximum(ql[k], 0.0))
        qrk = np.where(mask, 0.0, np.maximum(qr[k], 0.0))
        field = np.stack([rhow, the, thl[k], qtk, qlk, qrk, rhow * (qtk - qlk - qrk), rhow * qtk])[:, None]
        o, a = binning.iso_multi_integral(field, the[None], mask[None], area_cell, **bins)
        out[:, k], area[k] = o[:, 0], a[0]
    return out, area, nmask


def main():
    args = parse_args()
    th_min, th_max, nbin = args.theta_e_bins.split(",")
    bins = dict(th_min=float(th_min), th_max=float(th_max), nbin=int(nbin))
    theta_e = bins["th_min"] + (bins["th_max"] - bins["th_min"]) / (bins["nbin"] - 1) * np.arange(bins["nbin"])
    if args.member in CRASHED and not args.force:
        raise SystemExit(f"member {args.member} crashed (Jansson et al. 2023, Sec. 6.4); use --force to run it")

    cat = intake.open_catalog(CATALOG).simulations.DALES.botany.dx100m[args.domain]
    entry3d = cat["3D"]
    ds3d = entry3d.to_dask().sel(member=args.member)
    rhobf = cat["profiles"].to_dask()["rhobf"].sel(member=args.member)
    nz = ds3d.sizes["zt"] - 1  # w is needed on the level above
    zt = ds3d["zt"].values[:nz]
    area_cell = float(np.diff(ds3d["xt"].values[:2])[0] * np.diff(ds3d["yt"].values[:2])[0])
    times = parse_times(args.times, ds3d.sizes["time"])
    out_dir = args.save_dir / f"EXP_{args.member:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"member {args.member} ({args.domain}), {len(times)} times, output in {out_dir}")

    for it in times:
        out_file = out_dir / f"CB_ISO_EXP_{args.member:03d}_TIME_{it:02d}.nc"
        if out_file.exists():
            print(f"time {it}: {out_file.name} exists, skipped")
            continue
        t0 = clock.time()
        t = ds3d["time"].values[it]
        sel = dict(member=args.member, time=t)
        print(f"time {it} ({t}): reading")
        fields = {}
        for name in ("thl", "qt", "ql", "qr", "w"):
            fields[name] = load_field(entry3d, name, sel, args.retries)
            if fields[name] is None:
                break
        if any(f is None for f in fields.values()):
            print(f"time {it}: no data, skipped")
            continue
        rho = rhobf.sel(time=t).values[:nz]  # profiles are every 5 min: same time as the 3D snapshot
        sums, area, nmask = iso_sums(fields["thl"][:nz], fields["qt"][:nz], fields["ql"][:nz],
                                     fields["qr"][:nz], fields["w"][:nz + 1], rho, area_cell, bins)
        del fields

        dims = ("time", "height", "theta_e")
        names = ["MF", "ATHE", "ATHL", "AQT", "AQL", "AQR", "RHOWQV", "RHOWQT"]
        ds = xr.Dataset({"AREA": (dims, area[None]), **{n: (dims, s[None]) for n, s in zip(names, sums)}},
                        coords={"time": [t], "height": zt, "theta_e": theta_e})
        attrs = {
            "AREA": ("area", "m2"),
            "MF": ("upward mass flux, sum of rho*w*dA", "kg s-1"),
            "ATHE": ("sum of theta_e*dA", "K m2"),
            "ATHL": ("sum of theta_l*dA", "K m2"),
            "AQT": ("sum of q_t*dA", "kg kg-1 m2"),
            "AQL": ("sum of q_l*dA", "kg kg-1 m2"),
            "AQR": ("sum of q_r*dA", "kg kg-1 m2"),
            "RHOWQV": ("upward water vapor flux, sum of rho*w*q_v*dA", "kg s-1"),
            "RHOWQT": ("upward total water flux, sum of rho*w*q_t*dA", "kg s-1"),
        }
        for n, (long_name, units) in attrs.items():
            ds[n].attrs.update(long_name=long_name, units=units)
        ds["height"].attrs.update(long_name="height of the full levels zt", units="m")
        ds["theta_e"].attrs.update(long_name="equivalent potential temperature at 1000 hPa (bin centers)", units="K")
        ds.attrs.update(member=args.member, domain=args.domain, time_index=it, cell_area=area_cell,
                        masked_points=int(nmask), catalog=CATALOG,
                        bins_theta_e=f"{bins['th_min']},{bins['th_max']},{bins['nbin']}")
        edge = (area[:, 0].sum() + area[:, -1].sum()) / area.sum()
        ds.attrs["edge_bin_area_fraction"] = float(edge)
        ds.to_netcdf(out_file)
        print(f"time {it}: wrote {out_file.name} in {clock.time() - t0:.0f} s, "
              f"{nmask} points masked, {100 * edge:.3f}% of the area in the edge bins")


if __name__ == "__main__":
    main()
