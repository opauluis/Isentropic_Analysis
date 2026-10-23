import numpy as np
import pytest

from isentropic import mafalda as mf

# Quadratic bowl Psi = -(1 - X^2 - Z^2): the isolines Psi = -L are ellipses of area pi a b (1 - L),
# with upward motion at high theta_e, as for convection.
N = 10
A, B = 15.0, 6000.0          # half-widths in theta_e (K) and z (m)
TH0, Z0 = 325.0, 8000.0


@pytest.fixture
def bowl():
    z = np.linspace(0.0, 16000.0, 321)
    theta_edge = np.linspace(300.5, 350.5, 501)
    X = (theta_edge[None, :] - TH0) / A
    Z = (z[:, None] - Z0) / B
    psi = -(1.0 - X**2 - Z**2)
    return psi, z, theta_edge


def areas():
    L = (2 * np.arange(1, N + 1) - 1) / (2 * N)
    return np.pi * A * B * (1.0 - L)


def test_cycle_integral_square():
    # counterclockwise unit square: cyc x dy = +1
    x = np.array([0.0, 1.0, 1.0, 0.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    assert mf.cycle_integral(x, y) == pytest.approx(1.0)
    assert mf.cycle_integral(y, x) == pytest.approx(-1.0)


def test_trajectories_levels_and_orientation(bowl):
    psi, z, theta_edge = bowl
    traj = mf.trajectories(psi, z, theta_edge, N)
    assert len(traj) == N
    for k, t in enumerate(traj, start=1):
        assert t['level'] == pytest.approx(-(2 * k - 1) / (2 * N) * 1.0, rel=1e-3)
        assert t['dpsi'] == pytest.approx(1.0 / N, rel=1e-3)
        # rising on the high theta_e side of the ellipse, sinking on the low side
        dz = np.roll(t['z'], -1) - t['z']
        half_width = A * np.sqrt(1.0 + t['level'])
        assert np.all(dz[t['theta_e'] > TH0 + 0.5 * half_width] > -1e-6)
        assert np.all(dz[t['theta_e'] < TH0 - 0.5 * half_width] < 1e-6)
        assert np.sum(dz[t['theta_e'] > TH0]) > 0
    # deepest cycle first
    assert np.ptp(traj[0]['z']) > np.ptp(traj[-1]['z'])


def test_work_terms_match_ellipse_areas(bowl):
    psi, z, theta_edge = bowl
    dth = theta_edge[1] - theta_edge[0]
    theta = theta_edge - 0.5 * dth
    ZZ, TT = np.meshgrid(z, theta, indexing='ij')
    c, gamma, beta, d, e = 2.0, 6.5e-3, 1.0e-3, 1.0e-3, 50.0
    fields = {
        'T': 300.0 - gamma * ZZ,   # Wmax = c gamma Area
        's': c * TT,
        'b': beta * TT,            # Wb = beta Area
        'rt': d * TT,              # Wp = G d Area
        'rv': d * TT,
        'gv': e * ZZ,              # Dg = e d Area
    }
    cycles = mf.mafalda(psi, z, theta_edge, theta, fields, n=N)
    area = areas()
    got = {key: np.array([cy[key] for cy in cycles]) for key in ('Wmax', 'Wb', 'Wp', 'Dg')}
    np.testing.assert_allclose(got['Wmax'], c * gamma * area, rtol=2e-3)
    np.testing.assert_allclose(got['Wb'], beta * area, rtol=2e-3)
    np.testing.assert_allclose(got['Wp'], mf.G * d * area, rtol=2e-3)
    np.testing.assert_allclose(got['Dg'], e * d * area, rtol=2e-3)
    assert 'Wmax' in cycles[0] and len(cycles[0]['path']['T']) == len(cycles[0]['z'])


def test_thermo_fields():
    T = np.array([300.0, 250.0])
    f = mf.thermo_fields(np.array([1.0e5, 5.0e4]), T, np.array([0.015, 0.001]), np.zeros(2), np.zeros(2))
    assert set(f) == {'s', 'rt', 'gv', 'gl', 'gi'}
    assert all(np.shape(v) == (2,) for v in f.values())
    # unsaturated vapor has a lower Gibbs free energy than liquid water at the same temperature
    assert np.all(f['gv'] < f['gl'])
