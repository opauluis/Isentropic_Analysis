import numpy as np
import pytest

from isentropic import (iso_multi_integral, streamfunction, conditional_mean, isentropic_distribution,
                        diabatic_tendency, isentropic_mean_tendency, two_stream)

TH = dict(th_min=300.0, th_max=350.0, nbin=51)
DTH = 1.0


@pytest.fixture
def random_flow():
    # theta_e exactly at bin centers, so the binning involves no splitting between bins
    rng = np.random.default_rng(0)
    nz, nx, ny = 4, 20, 15
    theta = 300.0 + rng.integers(0, 51, (nz, nx, ny)).astype(float)
    rhow = rng.normal(size=(nz, nx, ny))
    q = rng.uniform(0.0, 0.02, (nz, nx, ny))
    out, area = iso_multi_integral(np.stack([rhow, q, rhow * q]), theta, False, 1.0, **TH)
    return theta, rhow, q, out, area


def test_streamfunction_matches_direct_sum(random_flow):
    theta, rhow, q, out, area = random_flow
    psi = streamfunction(out[0], area)
    npts = theta[0].size
    for k in range(theta.shape[0]):
        for i in (0, 10, 25, 50):
            direct = rhow[k][theta[k] <= 300.0 + i].sum() / npts
            np.testing.assert_allclose(psi[k, i], direct, atol=1e-12)
    # last bin: horizontal-mean mass flux
    np.testing.assert_allclose(psi[:, -1], rhow.mean((1, 2)))


def test_distribution_integrates_to_mean(random_flow):
    theta, rhow, q, out, area = random_flow
    dist = isentropic_distribution(out[1], area, DTH)
    np.testing.assert_allclose(dist.sum(-1) * DTH, q.mean((1, 2)))


def test_conditional_mean(random_flow):
    theta, rhow, q, out, area = random_flow
    qm = conditional_mean(out[1], area)
    k, i = 2, 17
    sel = theta[k] == 300.0 + i
    if sel.any():
        np.testing.assert_allclose(qm[k, i], q[k][sel].mean())
    empty = area == 0
    np.testing.assert_allclose(qm[empty], np.broadcast_to(q.mean((1, 2))[:, None], qm.shape)[empty])


def test_two_stream(random_flow):
    theta, rhow, q, out, area = random_flow
    ts = two_stream(out[0], area, {'q': out[2]})
    np.testing.assert_allclose(ts['M_up'] + ts['M_down'], rhow.mean((1, 2)))
    assert np.all(ts['M_up'] >= 0) and np.all(ts['M_down'] <= 0)
    # q in the mean ascent, Eq. (17a), computed directly
    k = 1
    up = np.zeros(theta[k].shape, bool)
    for i in range(51):
        sel = theta[k] == 300.0 + i
        up |= sel & (rhow[k][sel].sum() > 0)
    np.testing.assert_allclose(ts['f_up']['q'][k], (rhow[k] * q[k])[up].sum() / rhow[k][up].sum())


def test_diabatic_tendency_of_cooling_updraft():
    # Steady flow with constant density: an updraft covering a fraction a of the domain, whose
    # theta_e decreases with height at rate gamma (diabatic cooling w * dtheta_e/dz = -W gamma),
    # and a compensating downdraft at constant theta_e (no diabatic tendency).
    nz, ncol = 41, 100
    z = np.linspace(0.0, 4000.0, nz)
    W, a, gamma = 2.0, 0.1, 5.0e-3
    nup = int(a * ncol)
    theta = np.full((nz, ncol), 305.0)
    theta[:, :nup] = 345.0 - gamma * z[:, None]
    rhow = np.full((nz, ncol), -W * a / (1 - a))
    rhow[:, :nup] = W
    out, area = iso_multi_integral(rhow[None], theta, False, 1.0, **TH)
    psi = streamfunction(out[0], area)
    tend = diabatic_tendency(psi, z)
    # Integrated over theta_e: domain-mean rho * dtheta_e/dt = -a W gamma
    np.testing.assert_allclose(tend.sum(-1)[1:-1] * DTH, -a * W * gamma, rtol=1e-6)
    # No diabatic tendency in the downdraft bin
    np.testing.assert_allclose(tend[:, 4], 0.0, atol=1e-12)
    # Isentropic-mean tendency is -W gamma where the updraft is, and finite only where there is air
    mean_tend = isentropic_mean_tendency(psi, z, area, np.ones(nz), DTH)
    k = 20
    i_up = slice(int(theta[k, 0] - 300.0) - 1, int(theta[k, 0] - 300.0) + 1)
    np.testing.assert_allclose(np.nansum(mean_tend[k, i_up] * area[k, i_up]) / area[k, i_up].sum(),
                               -W * gamma, rtol=0.05)
    assert np.isnan(mean_tend[k, 20])
