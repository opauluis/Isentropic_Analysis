# Isentropic diagnostics from conditional sums (Pauluis and Mrowiec 2013)
# Olivier Pauluis
#
# All functions take the conditional sums produced by binning.iso_integral and friends,
# with height and theta_e as the last two axes (..., nz, nbin). Sums over several times
# (or ensemble members) can simply be added together before calling these functions.
#
# Notation: AREA = conditional sum of area, MF = conditional sum of rho*w*area,
# A_tot(z) = AREA.sum(-1) = total area at each level.

import numpy as np


def isentropic_distribution(field_sum, area, dth):
    '''Isentropic distribution <f>(z, theta_e) [units of f per K], Eq. (1).

    field_sum: conditional sum of f * area (..., nz, nbin)
    area: conditional sum of area (..., nz, nbin)
    dth: bin width (K)
    '''
    return field_sum / (np.sum(area, -1, keepdims=True) * dth)


def streamfunction(mf_sum, area):
    '''Isentropic streamfunction Psi(z, theta_e) [kg m-2 s-1], Eq. (3).

    Psi[..., k, i] is the net upward mass flux per unit area at level k of all air
    with theta_e below the upper edge of bin i (theta_bin[i] + dth/2).

    mf_sum: conditional sum of rho * w * area (..., nz, nbin)
    area: conditional sum of area (..., nz, nbin)
    '''
    return np.cumsum(mf_sum, -1) / np.sum(area, -1, keepdims=True)


def conditional_mean(field_sum, area, profile=None, tiny=1e-10):
    '''Conditional (isentropic) mean of f at each height and theta_e bin.

    field_sum: conditional sum of f * area (..., nz, nbin)
    area: conditional sum of area (..., nz, nbin)
    profile: value used in bins with (almost) no area, (..., nz); defaults to the
             horizontal mean of f.
    '''
    if profile is None:
        profile = np.sum(field_sum, -1) / np.sum(area, -1)
    return (tiny * np.expand_dims(profile, -1) + field_sum) / (tiny + area)


def diabatic_tendency(psi, z):
    '''Mass-weighted diabatic tendency <rho dtheta_e/dt> = -dPsi/dz, Eq. (10).

    psi: streamfunction (..., nz, nbin)
    z: heights (nz)
    Returns <rho dtheta_e/dt> [kg m-3 s-1] at the upper edge of each bin; summed over
    bins and multiplied by dth, it gives the total diabatic tendency at each level.
    '''
    return -np.gradient(psi, z, axis=-2)


def isentropic_mean_tendency(psi, z, area, rho, dth, tiny=1e-10):
    '''Isentropic-mean diabatic tendency of theta_e [K s-1]: <rho dtheta_e/dt> / <rho>.

    psi: streamfunction (..., nz, nbin)
    z: heights (nz)
    area: conditional sum of area (..., nz, nbin)
    rho: density profile (nz)
    dth: bin width (K)
    Bins with no air are set to NaN.
    '''
    # <rho> [kg m-3 K-1] at the upper bin edges (mean of the two adjacent bins)
    frac = area / np.sum(area, -1, keepdims=True)
    frac_edge = 0.5 * (frac + np.concatenate([frac[..., 1:], frac[..., -1:]], -1))
    rho_iso = np.asarray(rho)[:, None] * frac_edge / dth
    tend = diabatic_tendency(psi, z)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(rho_iso > tiny, tend / rho_iso, np.nan)


def two_stream(mf_sum, area, flux_sums=None):
    '''Two-stream decomposition into a mean ascent and a mean descent, Eqs. (14)-(17).

    mf_sum: conditional sum of rho * w * area (..., nz, nbin)
    area: conditional sum of area (..., nz, nbin)
    flux_sums: optional dict {name: conditional sum of rho * w * f * area}

    Returns a dict with
      'M_up', 'M_down': upward and downward mass transports [kg m-2 s-1] (..., nz)
      'f_up', 'f_down': dicts {name: value of f in the mean ascent / descent} (..., nz)
    '''
    a_tot = np.sum(area, -1)
    up = mf_sum > 0
    mf_up = np.sum(np.where(up, mf_sum, 0.0), -1)
    mf_down = np.sum(np.where(up, 0.0, mf_sum), -1)
    out = {'M_up': mf_up / a_tot, 'M_down': mf_down / a_tot, 'f_up': {}, 'f_down': {}}
    with np.errstate(invalid='ignore', divide='ignore'):
        for name, fs in (flux_sums or {}).items():
            out['f_up'][name] = np.sum(np.where(up, fs, 0.0), -1) / mf_up
            out['f_down'][name] = np.sum(np.where(up, 0.0, fs), -1) / mf_down
    return out
