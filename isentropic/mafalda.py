# Mean Air Flow As Lagrangian Dynamics Approximation (MAFALDA)
# Pauluis (2016), Pauluis and Zhang (2017)
#
# 1. Mean trajectories are the isolines of the isentropic streamfunction Psi(z, theta_e),
#    Psi = (2k - 1) / (2N) * min(Psi) for k = 1..N, Eq. (5).
# 2. Thermodynamic properties along the trajectories are the conditional means at the same
#    (z, theta_e), Eq. (6).
# 3. Each trajectory is a thermodynamic cycle; the maximum (Carnot) work is decomposed as
#    Wmax = Wb + Wp + Dg, Eq. (12), with
#       Wmax = cyc T ds,  Wb = cyc b dz,  Wp = cyc r_T G dz,  Dg = - sum_w cyc g_w dr_w.

import numpy as np
import contourpy

from .thermo import moist_entropy, Gibbs_vapor, Gibbs_liquid, Gibbs_ice

G = 9.81


def cycle_integral(a, b):
    '''Closed-loop integral cyc a db along a trajectory (trapezoidal rule, last point joins the first).'''
    a = np.asarray(a)
    b = np.asarray(b)
    return np.sum(0.5 * (a + np.roll(a, -1)) * (np.roll(b, -1) - b))


def interp_zth(field, z, theta, zq, thq):
    '''Bilinear interpolation of field (nz, nbin) on the (z, theta) grid at points (zq, thq).'''
    def locate(x, xq):
        i = np.clip(np.searchsorted(x, xq) - 1, 0, len(x) - 2)
        t = np.clip((xq - x[i]) / (x[i + 1] - x[i]), 0.0, 1.0)
        return i, t
    k, tz = locate(np.asarray(z), np.asarray(zq))
    i, tt = locate(np.asarray(theta), np.asarray(thq))
    return ((1 - tz) * (1 - tt) * field[k, i] + (1 - tz) * tt * field[k, i + 1]
            + tz * (1 - tt) * field[k + 1, i] + tz * tt * field[k + 1, i + 1])


def trajectories(psi, z, theta_edge, n=20):
    '''MAFALDA mean trajectories: isolines of Psi at the levels of Eq. (5).

    psi: streamfunction (nz, nbin), from analysis.streamfunction
    z: heights (nz)
    theta_edge: theta_e at which psi is defined (upper bin edges, theta_bin + dth/2)
    n: number of cycles

    Returns a list of n dicts with
      'level': value of Psi on the trajectory
      'dpsi': mass transport of the cycle, -min(Psi) / n [kg m-2 s-1]
      'z', 'theta_e': closed path (last point not repeated), oriented along the flow
                      (upward where Psi increases with theta_e)
    For each level, the longest isoline is kept.
    '''
    psi = np.asarray(psi)
    psi_min = psi.min()
    gen = contourpy.contour_generator(x=theta_edge, y=z, z=psi, line_type='Separate')
    dpsi_dz, dpsi_dth = np.gradient(psi, z, theta_edge)
    out = []
    for k in range(1, n + 1):
        level = (2 * k - 1) / (2 * n) * psi_min
        lines = gen.lines(level)
        if not lines:
            out.append({'level': level, 'dpsi': -psi_min / n, 'z': np.empty(0), 'theta_e': np.empty(0)})
            continue
        line = max(lines, key=len)
        if np.allclose(line[0], line[-1]):
            line = line[:-1]
        th, zz = line[:, 0], line[:, 1]
        # Mean flow in (z, theta_e) is along (dPsi/dtheta, -dPsi/dz)
        gz = interp_zth(dpsi_dth, z, theta_edge, zz, th)
        gth = -interp_zth(dpsi_dz, z, theta_edge, zz, th)
        dz = np.roll(zz, -1) - zz
        dth = np.roll(th, -1) - th
        if np.sum(dz * gz + dth * gth) < 0:
            zz, th = zz[::-1], th[::-1]
        out.append({'level': level, 'dpsi': -psi_min / n, 'z': zz, 'theta_e': th})
    return out


def along(fields, z, theta, traj):
    '''Conditional means along a trajectory, Eq. (6).

    fields: dict {name: conditional mean (nz, nbin)} on the (z, theta) grid (bin centers)
    traj: one element of trajectories()
    '''
    return {name: interp_zth(f, z, theta, traj['z'], traj['theta_e']) for name, f in fields.items()}


def work_budget(path, G=G):
    '''Work decomposition of one cycle, Eq. (12) [J per kg of dry air].

    path: dict of values along the cycle; uses whichever of these are present:
      'T', 's' -> Wmax = cyc T ds
      'b', 'z' -> Wb = cyc b dz
      'rt', 'z' -> Wp = cyc r_T G dz
      ('gv', 'rv'), ('gl', 'rl'), ('gi', 'ri') -> Dg = - sum cyc g_w dr_w
    '''
    out = {}
    if 'T' in path and 's' in path:
        out['Wmax'] = cycle_integral(path['T'], path['s'])
    if 'b' in path:
        out['Wb'] = cycle_integral(path['b'], path['z'])
    if 'rt' in path:
        out['Wp'] = G * cycle_integral(path['rt'], path['z'])
    pairs = [(g, r) for g, r in (('gv', 'rv'), ('gl', 'rl'), ('gi', 'ri')) if g in path and r in path]
    if pairs:
        out['Dg'] = -sum(cycle_integral(path[g], path[r]) for g, r in pairs)
    return out


def mafalda(psi, z, theta_edge, theta, fields, n=20, G=G):
    '''Full MAFALDA analysis.

    psi: streamfunction (nz, nbin) at theta_edge
    z: heights (nz)
    theta_edge: theta_e of the streamfunction (upper bin edges)
    theta: theta_e of the conditional means (bin centers)
    fields: dict {name: conditional mean (nz, nbin)}, e.g. 'T', 's', 'b', 'rt', 'rv', 'rl',
            'ri', 'gv', 'gl', 'gi' (see work_budget)
    n: number of cycles

    Returns a list of n dicts, one per cycle, with the trajectory, the fields along it
    ('path', including 'z' and 'theta_e'), and the work terms of work_budget.
    Multiply the work terms by 'dpsi' and sum over cycles for the total power [W m-2], Eq. (14).
    '''
    cycles = trajectories(psi, z, theta_edge, n)
    for c in cycles:
        c['path'] = {'z': c['z'], 'theta_e': c['theta_e']}
        if len(c['z']):
            c['path'].update(along(fields, z, theta, c))
        c.update(work_budget(c['path'], G) if len(c['z']) else {})
    return cycles


def thermo_fields(P, T, rv, rl, ri):
    '''Point-wise fields needed by MAFALDA, to be binned in (z, theta_e) like any other field.

    P [Pa], T [K], mixing ratios per unit mass of dry air. Returns a dict with
    's' (moist entropy, J/kg/K), 'rt', 'gv', 'gl', 'gi' (Gibbs free energies, J/kg).
    '''
    return {
        's': moist_entropy(P, T, rv, rl, ri),
        'rt': rv + rl + ri,
        'gv': Gibbs_vapor(P, T, rv),
        'gl': Gibbs_liquid(T),
        'gi': Gibbs_ice(T),
    }
