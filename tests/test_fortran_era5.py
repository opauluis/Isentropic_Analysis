# End-to-end test of fortran/era5 on a small synthetic ERA5 day, against a numpy version
# of the same algorithm (including the original O(n^2) zonal filter)

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")
if shutil.which("nf-config") is None or shutil.which("gfortran") is None:
    pytest.skip("gfortran and netCDF-Fortran are needed", allow_module_level=True)

SRC = Path(__file__).parents[1] / "fortran" / "era5"

LEVELS = np.array([1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400,
                   450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000],
                  dtype=np.float64)
NLON, NLAT, NT = 48, 25, 2
FILES = {"W": "135_w", "T": "130_t", "Z": "129_z", "Q": "133_q", "CIWC": "247_ciwc", "CLWC": "246_clwc",
         "CSWC": "076_cswc", "CRWC": "075_crwc"}
DATE = "20200115"

NBIN, THE_MIN, THE_MAX = 181, 200.0, 380.0
RADIUS = 6378e3

# constants and esat of iso_thermo.F90
CPAIR, RD, CPV, CL, CI = 1004.0, 287.0, 1880.0, 4218.0, 2106.0
LF0, LV0, TREF, RVAP, PREF = 333600.0, 2510400.0, 273.15, 461.0, 1.0e5
A_ESAT = [6.11239921, 0.443987641, 0.142986287e-1, 0.264847430e-3, 0.302950461e-5, 0.206739458e-7,
          0.640689451e-10, -0.952447341e-13, -0.976195544e-15]


def synthetic_fields(rng):
    lat = np.linspace(90, -90, NLAT)
    lon = np.arange(NLON) * 360.0 / NLON
    p = LEVELS[:, None, None] * 100.0
    phi = np.deg2rad(lat)[None, :, None]
    lam = np.deg2rad(lon)[None, None, :]
    shape = (NT, len(LEVELS), NLAT, NLON)

    def noise(scale):
        return scale * rng.standard_normal(shape)

    t = 300.0 * (p / 1e5) ** 0.19 - 20 * np.sin(phi) ** 2 + 2 * np.cos(2 * lam) + noise(1.0)
    q = 0.018 * np.cos(phi) ** 2 * (p / 1e5) ** 3 * np.exp(noise(0.2))
    w = 0.2 * np.cos(3 * lam + phi) + 0.1 * np.sin(7 * lam) + noise(0.3)
    z = 9.81 * 7000.0 * np.log(1e5 / p) + noise(5.0) * 9.81
    clw = 1e-4 * np.exp(noise(0.5)) * (p > 40000)
    ciw = 1e-4 * np.exp(noise(0.5)) * (p < 50000)
    fields = {"T": t, "Q": q, "W": w, "Z": z, "CLWC": clw, "CIWC": ciw, "CSWC": 0.3 * ciw, "CRWC": 0.2 * clw}
    fields = {k: np.broadcast_to(v, shape).astype(np.float32) for k, v in fields.items()}
    # mountains reaching up to ~ 700 hPa in one sector
    zs = 9.81 * 3000.0 * np.exp(-((lon - 90.0) / 30.0) ** 2)[None, :] * np.cos(phi[0]) ** 2
    zs = zs.astype(np.float32)
    return lat, lon, fields, zs


def write_era5(root, lat, lon, fields, zs):
    month_dir = root / "pl" / DATE[:6]
    month_dir.mkdir(parents=True)
    for name, code in FILES.items():
        path = month_dir / f"e5.oper.an.pl.128_{code}.ll025sc.{DATE}00_{DATE}23.nc"
        with netCDF4.Dataset(path, "w") as nc:
            nc.createDimension("time", NT)
            nc.createDimension("level", len(LEVELS))
            nc.createDimension("latitude", NLAT)
            nc.createDimension("longitude", NLON)
            nc.createVariable("latitude", "f8", ("latitude",))[:] = lat
            nc.createVariable("longitude", "f8", ("longitude",))[:] = lon
            nc.createVariable("level", "f8", ("level",))[:] = LEVELS
            nc.createVariable(name, "f4", ("time", "level", "latitude", "longitude"))[:] = fields[name]
    zs_file = root / "zs.nc"
    with netCDF4.Dataset(zs_file, "w") as nc:
        nc.createDimension("time", 1)
        nc.createDimension("latitude", NLAT)
        nc.createDimension("longitude", NLON)
        nc.createVariable("Z", "f4", ("time", "latitude", "longitude"))[:] = zs[None]
    return zs_file


def reference(lat, lon, fields, zs, nlon_slice=12, nlat_slice=13, p_top=150.0):
    k0 = np.argmax(LEVELS >= p_top)
    p = LEVELS[k0:, None, None] * 100.0
    f = {k: v[:, k0:].astype(np.float64) for k, v in fields.items()}
    nlev = p.shape[0]
    nx, ny = NLON // nlon_slice, (NLAT - 1) // (nlat_slice - 1)
    dthe = (THE_MAX - THE_MIN) / (NBIN - 1)

    dlat = np.deg2rad(abs(lat[1] - lat[0]))
    area_row = RADIUS ** 2 * 2 * np.pi / NLON * dlat * np.maximum(np.cos(np.deg2rad(lat)), 0.0)

    # original convolution filters
    i = np.arange(1 - NLON, NLON)
    filt13 = sum(2 * np.cos(2 * np.pi * k * i / NLON) / NLON for k in range(1, 4))
    filt410 = sum(2 * np.cos(2 * np.pi * k * i / NLON) / NLON for k in range(4, 11))
    conv13 = np.array([filt13[NLON - 1 - ii + np.arange(NLON)] for ii in range(NLON)])  # (i, m)
    conv410 = np.array([filt410[NLON - 1 - ii + np.arange(NLON)] for ii in range(NLON)])

    acc = np.zeros((12, NBIN, nx, ny, nlev))
    accz = np.zeros((4, NBIN, ny, nlev))
    for it in range(NT):
        t, w, gz = f["T"][it], f["W"][it], f["Z"][it]
        qv = np.maximum(f["Q"][it], 1e-10)
        qi = f["CSWC"][it] + f["CIWC"][it]
        ql = f["CRWC"][it] + f["CLWC"][it]

        qt = qv + ql + qi
        r, rl, ri = qv / (1 - qt), ql / (1 - qt), qi / (1 - qt)
        dt = np.maximum(-80.0, t - 273.16)
        es = 100.0 * np.polynomial.polynomial.polyval(dt, A_ESAT)
        e = p * RVAP * r / (RD + RVAP * r)
        pd = p - np.minimum(e, es)
        lv = LV0 + (t - TREF) * (CPV - CL)
        cp = CPAIR + (r + rl) * CL + ri * CI
        s = cp * np.log(t / TREF) - RD * np.log(pd / PREF) + lv * r / t - LF0 * ri / t - RVAP * r * np.log(e / es)
        the = TREF * np.exp(s / (CPAIR + (r + rl + ri) * CL))
        h = cp * (t - TREF) + lv * r - LF0 * ri

        above = gz > zs[None]
        wm = np.where(above, w, 0.0)
        o0 = (wm.sum(-1) / np.maximum(above.sum(-1), 1))[..., None] * np.ones(NLON)
        o13 = wm @ conv13.T
        o410 = wm @ conv410.T
        osw = w - o0 - o13 - o410

        vals = np.stack([np.ones_like(t), w, t, qv, gz, s, h, ql, qi, qv * w, ql * w, qi * w])
        valz = np.stack([o0, o13, o410, osw])
        x = (the - THE_MIN) / dthe
        b1 = np.clip(np.ceil(x), 1, NBIN).astype(int)
        wt = np.clip(b1 - x, 0.0, 1.0)
        for jbin in range(ny):
            for jsl in range(nlat_slice):
                j = jbin * (nlat_slice - 1) + jsl
                a = area_row[j] * (0.5 if jsl in (0, nlat_slice - 1) else 1.0)
                for ii in range(NLON):
                    ibin = ii // nlon_slice
                    for k in range(nlev):
                        if not above[k, j, ii]:
                            continue
                        for b, ww in ((b1[k, j, ii], wt[k, j, ii]), (min(b1[k, j, ii] + 1, NBIN), 1 - wt[k, j, ii])):
                            acc[:, b - 1, ibin, jbin, k] += a * ww * vals[:, k, j, ii]
                            accz[:, b - 1, jbin, k] += a * ww * valz[:, k, j, ii]
    acc /= NT
    accz /= NT

    def mean_w(mf, area):
        return np.where(area > 0, mf / np.where(area > 0, area, 1.0), 0.0)

    area, mf = acc[0], acc[1]
    out = {name: acc[n] for n, name in enumerate(
        ["AREA_THE", "MF_THE", "RHOT_THE", "RHOQ_THE", "RHOZ_THE", "RHOS_THE", "RHOH_THE", "RHOQL_THE",
         "RHOQI_THE", "RHOWQ_THE", "RHOWQL_THE", "RHOWQI_THE"])}
    out["MF_THE"] = mf - mean_w(mf.sum((0, 1, 2)), area.sum((0, 1, 2))) * area
    out["MFC_THE"] = mf - mean_w(mf.sum(0), area.sum(0)) * area
    out["STR_THE"] = np.cumsum(out["MF_THE"].sum((1, 2)), axis=0)
    area_y = area.sum(1)
    mfz = accz.copy()
    mfz[0] -= mean_w(mfz[0].sum((0, 1)), area_y.sum((0, 1))) * area_y
    for n in range(1, 4):
        mfz[n] -= mean_w(mfz[n].sum(0), area_y.sum(0)) * area_y
    for n, name in enumerate(["0", "13", "410", "SW"]):
        out[f"MF{name}_THE"] = mfz[n]
        out[f"STR{name}_THE"] = np.cumsum(mfz[n].sum(1), axis=0)
    out["lat"] = lat[(nlat_slice - 1) * np.arange(ny) + nlat_slice // 2]
    out["lon"] = lon.reshape(nx, nlon_slice).mean(1)
    out["press"] = p[:, 0, 0]
    return out


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    root = tmp_path_factory.mktemp("era5")
    build = root / "build"
    shutil.copytree(SRC, build)
    subprocess.run(["make", "FC=gfortran", "FFLAGS=-O2 -fcheck=all"], cwd=build, check=True,
                   capture_output=True)

    rng = np.random.default_rng(0)
    lat, lon, fields, zs = synthetic_fields(rng)
    zs_file = write_era5(root, lat, lon, fields, zs)
    (root / "out" / DATE[:4]).mkdir(parents=True)
    env = dict(os.environ, ERA5_PL_DIR=str(root / "pl"), ERA5_ZS_FILE=str(zs_file))
    subprocess.run([str(build / "era5_iso.exe"), DATE[:4], DATE[4:6], DATE[6:], str(root / "out")],
                   env=env, check=True, capture_output=True)
    out_file = root / "out" / DATE[:4] / f"ERA5_STR_THE.{DATE[:4]}_{DATE[4:6]}_{DATE[6:]}.nc"
    with netCDF4.Dataset(out_file) as nc:
        # netCDF (day, press, lat, lon, theta_e) -> Fortran order (theta_e, lon, lat, press)
        result = {name: np.asarray(v[:]).squeeze().T for name, v in nc.variables.items()}
    return result, reference(lat, lon, fields, zs), (lat, lon, fields, zs)


def test_coordinates(run):
    out, ref, _ = run
    np.testing.assert_allclose(out["lat"], ref["lat"])
    np.testing.assert_allclose(out["lon"], ref["lon"])
    np.testing.assert_allclose(out["press"], ref["press"])
    assert out["lat"][0] > out["lat"][-1]


def test_matches_reference(run):
    out, ref, _ = run
    for name, r in ref.items():
        if name in ("lat", "lon", "press"):
            continue
        scale = np.abs(r).max()
        np.testing.assert_allclose(out[name], r, rtol=0, atol=2e-4 * scale, err_msg=name)


def test_area_covers_sphere(run):
    out, _, (lat, lon, fields, zs) = run
    # top level is above all topography: its area is the whole sphere, up to the trapezoidal
    # error dlat^2/12 of the half-weighted edge rows (1.6e-6 at 0.25 deg)
    dlat = np.deg2rad(abs(lat[1] - lat[0]))
    rel = out["AREA_THE"][..., 0].sum() / (4 * np.pi * RADIUS ** 2) - 1
    assert abs(rel + dlat ** 2 / 12) < 1e-3 * dlat ** 2


def test_column_mean_removed(run):
    out, _, _ = run
    mfc = out["MFC_THE"].sum(0)
    scale = np.abs(out["MF_THE"]).sum(0).max()
    assert np.abs(mfc).max() < 1e-4 * scale
