import numpy as np
import pytest

from isentropic import sam, thermo

BINS = {"theta_e": dict(th_min=300.0, th_max=370.0, nbin=71),
        "theta_ef": dict(th_min=300.0, th_max=390.0, nbin=91)}


@pytest.fixture
def snap():
    rng = np.random.default_rng(1)
    nz, ny, nx = 6, 8, 10
    z = np.linspace(0, 10e3, nz)[:, None, None]
    T = 300 - 6.5e-3 * z + rng.normal(0, 1, (nz, ny, nx))
    qv = 15 * np.exp(-z / 2500) * rng.uniform(0.8, 1.0, (nz, ny, nx))   # g/kg
    qn = np.where(rng.random((nz, ny, nx)) < 0.2, 0.5, 0.0)             # g/kg
    p = 1000 * np.exp(-z[:, 0, 0] / 8000)                                # mb
    pp = rng.normal(0, 10, (nz, ny, nx))                                 # Pa
    w = rng.normal(0, 1, (nz, ny, nx))
    return dict(w=w, T=T, qv=qv, qn=qn, p=p, pp=pp)


def test_condensate_ramp_and_qi():
    T = np.array([240.0, 263.16, 280.0])
    qn = np.ones(3)
    ql, qi = sam.condensate(T, qn)
    np.testing.assert_allclose(ql, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(ql + qi, qn)
    ql, qi = sam.condensate(T, qn, qi=np.array([0.2, 2.0, -1.0]))
    np.testing.assert_allclose(qi, [0.2, 1.0, 0.0])
    np.testing.assert_allclose(ql, [0.8, 0.0, 1.0])


def test_snapshot_units(snap):
    rhow, th, rho, cloudy = sam.snapshot(**snap)
    P = snap["p"][:, None, None] * 100 + snap["pp"]
    ql, qi = sam.condensate(snap["T"], snap["qn"] * 1e-3)
    the = thermo.equivalent_potential_temperature(P, snap["T"], snap["qv"] * 1e-3, ql, qi)
    np.testing.assert_allclose(th["theta_e"], the)
    np.testing.assert_allclose(rhow, rho[:, None, None] * snap["w"])
    np.testing.assert_array_equal(cloudy, snap["qn"] > sam.QN_CLOUD)
    # a given density profile is used as is
    _, _, rho2, _ = sam.snapshot(**snap, rho=np.full(6, 1.1))
    np.testing.assert_allclose(rho2, 1.1)


def test_conditional_sums_cloudy_part(snap):
    rhow, th, rho, cloudy = sam.snapshot(**snap)
    out = sam.conditional_sums(rhow, th, BINS, cloudy)
    npts = rhow[0].size
    for name in BINS:
        # every point is counted once, and the cloudy sums are a subset of the total
        np.testing.assert_allclose(out[f"AREA_{name}"].sum(-1), npts)
        np.testing.assert_allclose(out[f"AREA_CLD_{name}"].sum(-1), cloudy.sum((1, 2)))
        np.testing.assert_allclose(out[f"MF_{name}"].sum(-1), rhow.sum((1, 2)))
        np.testing.assert_allclose(out[f"MF_CLD_{name}"].sum(-1), (rhow * cloudy).sum((1, 2)))
    assert "MF_CLD_theta_e" not in sam.conditional_sums(rhow, th, BINS)
