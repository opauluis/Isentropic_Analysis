# Isentropic binning: conditional sums of a field in terms of a sorting variable (e.g. theta_e)
# Olivier Pauluis
#
# Each data point is shared linearly between the two nearest bins, so that both the area
# and the field integral are conserved. Data below th_min or above th_max is put in the
# first/last bin. Points where the sorting variable is not finite are left out.

import numpy as np


def _bin_sums(field, theta, mask, area_in, th_min, th_max, nbin):
    # field: (nfield, nz, npts), theta/mask/area_in: (nz, npts)
    # returns output (nfield, nz, nbin) and area (nz, nbin)
    nfield, nz, npts = field.shape
    dth = (th_max - th_min) / (nbin - 1)

    keep = ~mask & np.isfinite(theta)
    x = np.where(keep, (theta - th_min) / dth, 0.0)
    ibin = np.clip(np.floor(x), 0, nbin - 2).astype(np.intp)
    weight = np.clip(x - ibin, 0.0, 1.0)
    a = np.where(keep, area_in, 0.0)
    wa = (1.0 - weight) * a
    wa2 = weight * a

    # flat index of (level, bin) for the lower bin; the upper bin is idx + 1
    idx = (np.arange(nz)[:, None] * nbin + ibin)[keep]
    wa = wa[keep]
    wa2 = wa2[keep]
    size = nz * nbin

    area = (np.bincount(idx, wa, size) + np.bincount(idx + 1, wa2, size)).reshape(nz, nbin)
    output = np.zeros((nfield, nz, nbin))
    for n in range(nfield):
        f = field[n][keep]
        output[n] = (np.bincount(idx, wa * f, size) + np.bincount(idx + 1, wa2 * f, size)).reshape(nz, nbin)
    return output, area


def _prepare(field, theta, mask, area_in, nfield_dims):
    # Flatten the horizontal dimensions: field -> (nfield, nz, npts), others -> (nz, npts)
    field = np.asarray(field, dtype=float)
    theta = np.asarray(theta, dtype=float)
    if nfield_dims == 0:
        field = field[None]
    nz = theta.shape[0]
    mask = np.broadcast_to(np.asarray(mask, dtype=bool), theta.shape)
    area_in = np.broadcast_to(np.asarray(area_in, dtype=float), theta.shape)
    return (field.reshape(field.shape[0], nz, -1), theta.reshape(nz, -1),
            mask.reshape(nz, -1), area_in.reshape(nz, -1))


def iso_integral(field, theta, mask, area_in, th_min=300.0, th_max=400.0, nbin=101):
    '''Conditional sum (aka isentropic integral) of "field" binned in terms of theta.

    field: integrating variable (nz, nx, ny)
    theta: sorting variable (nz, nx, ny)
    mask: True/1 = exclude the point, False/0 = include (nz, nx, ny)
    area_in: horizontal grid area (nx, ny), or anything that broadcasts to theta
    th_min, th_max: lowest and highest bin; data outside goes in the first/last bin
    nbin: number of equally spaced bins

    Returns output (nz, nbin), the conditional sum of field, and area (nz, nbin).
    '''
    output, area = _bin_sums(*_prepare(field, theta, mask, area_in, 0), th_min, th_max, nbin)
    return output[0], area


def iso_multi_integral(field, theta, mask, area_in, th_min=300.0, th_max=400.0, nbin=101):
    '''Same as iso_integral for several fields at once.

    field: (nfield, nz, nx, ny); returns output (nfield, nz, nbin) and area (nz, nbin).
    '''
    return _bin_sums(*_prepare(field, theta, mask, area_in, 1), th_min, th_max, nbin)


def iso_multicell_integral(field, theta, mask, area_in, th_min=300.0, th_max=400.0, nbin=101):
    '''Same as iso_multi_integral for unstructured grids.

    field: (nfield, nz, ncell), theta and mask: (nz, ncell), area_in: (ncell)
    Returns output (nfield, nz, nbin) and area (nz, nbin).
    '''
    return _bin_sums(*_prepare(field, theta, mask, area_in, 1), th_min, th_max, nbin)


def iso_average(field_sum, area, profile, nstart=0, nend=None, dt=1, tiny=1e-10):
    '''Conditional average from conditional sums, averaged over blocks of dt time steps.

    field_sum: conditional sums of a field (nt, nz, nbin)
    area: conditional sums of area (nt, nz, nbin)
    profile: horizontal mean of the field (nz); used in bins with (almost) no area
    nstart, nend: time steps to use
    dt: number of consecutive time steps averaged together

    Returns the conditional average (nslice, nz, nbin), with nslice = (nend - nstart) // dt.
    '''
    field_sum = np.asarray(field_sum)[nstart:nend]
    area = np.asarray(area)[nstart:nend]
    nt, nz, nbin = area.shape
    nslice = nt // dt
    fs = field_sum[:nslice * dt].reshape(nslice, dt, nz, nbin).sum(1)
    ar = area[:nslice * dt].reshape(nslice, dt, nz, nbin).sum(1)
    return (tiny * np.asarray(profile)[:, None] + fs) / (tiny + ar)
