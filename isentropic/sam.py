# Isentropic sums for SAM (System for Atmospheric Modeling) 3D output
#
# SAM units: W (m/s), TABS (K), QV, QN, QI (g/kg), p (mb), PP (Pa).
# QN is the non-precipitating condensate (cloud water + cloud ice). With the M2005
# microphysics QI is written separately; otherwise (SAM1MOM) QN is split into liquid
# and ice with SAM's temperature ramp between T_ICE and T_LIQ.

import numpy as np

from . import thermo
from .binning import iso_multi_integral

T_ICE, T_LIQ = 253.16, 273.16
QN_CLOUD = 0.01  # g/kg: SAM's cloud-fraction threshold

COORDS = {
    "theta_e": thermo.equivalent_potential_temperature,
    "theta_ef": thermo.frozen_equivalent_potential_temperature,
}


def condensate(T, qn, qi=None, t_ice=T_ICE, t_liq=T_LIQ):
    '''Split the non-precipitating condensate qn into liquid and ice (same units as qn).

    If qi (cloud ice) is given, liquid = qn - qi; otherwise SAM1MOM's temperature ramp is used.
    '''
    if qi is not None:
        qi = np.clip(qi, 0.0, qn)
        return qn - qi, qi
    liq_frac = np.clip((T - t_ice) / (t_liq - t_ice), 0.0, 1.0)
    return qn * liq_frac, qn * (1.0 - liq_frac)


def snapshot(w, T, qv, qn, p, pp, qi=None, rho=None, qn_cloud=QN_CLOUD, coords=COORDS):
    '''Mass flux, sorting coordinates, density profile and cloud mask of one SAM snapshot.

    w, T, qv, qn, pp, qi: (nz, ny, nx) in SAM units; p: (nz) in mb.
    rho: density profile (nz) in kg/m3, e.g. RHO from the statistics file; if None, the
    horizontal mean of the density computed from the snapshot is used.
    Returns rho*w (nz, ny, nx), {name: coordinate (nz, ny, nx)}, rho (nz) and
    cloudy = qn > qn_cloud (nz, ny, nx).
    '''
    qv = np.maximum(qv * 1e-3, 1e-10)  # g/kg -> kg/kg
    qn = np.maximum(qn * 1e-3, 0.0)
    ql, qi = condensate(T, qn, None if qi is None else qi * 1e-3)
    P = np.asarray(p)[:, None, None] * 100.0 + pp  # mb -> Pa, plus perturbation
    th = {name: f(P, T, qv, ql, qi) for name, f in coords.items()}
    if rho is None:
        rho = thermo.density(P, T, qv, ql, qi).mean((1, 2))
    rho = np.asarray(rho, dtype=float)
    return rho[:, None, None] * w, th, rho, qn > qn_cloud * 1e-3


def conditional_sums(rhow, coords, bins, cloudy=None):
    '''Conditional sums of one snapshot in each sorting coordinate.

    rhow: mass flux (nz, ny, nx); coords: {name: coordinate (nz, ny, nx)};
    bins: {name: dict(th_min=, th_max=, nbin=)}; cloudy: optional mask of cloudy points.
    Returns {f"{var}_{name}": (nz, nbin)} with var = MF (sum of rho*w), AREA (number of points)
    and ATH (sum of the coordinate), and MF_CLD, AREA_CLD over the cloudy points only.
    '''
    out = {}
    for name, th in coords.items():
        mask = ~np.isfinite(th)
        (mf, ath), area = iso_multi_integral(np.stack([rhow, th]), th, mask, 1.0, **bins[name])
        out[f"MF_{name}"], out[f"ATH_{name}"], out[f"AREA_{name}"] = mf, ath, area
        if cloudy is not None:
            (mf,), area = iso_multi_integral(rhow[None], th, mask | ~cloudy, 1.0, **bins[name])
            out[f"MF_CLD_{name}"], out[f"AREA_CLD_{name}"] = mf, area
    return out
