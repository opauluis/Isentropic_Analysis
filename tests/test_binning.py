import numpy as np
import pytest

from isentropic import iso_integral, iso_multi_integral, iso_multicell_integral, iso_average
from reference import iso_analysis_legacy as legacy

TH = dict(th_min=300.0, th_max=350.0, nbin=51)


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    nz, nx, ny, nf = 5, 12, 9, 3
    theta = rng.uniform(290.0, 360.0, (nz, nx, ny))  # includes values outside the bins
    field = rng.normal(size=(nf, nz, nx, ny))
    mask = rng.random((nz, nx, ny)) < 0.2
    area = rng.uniform(0.5, 2.0, (nx, ny))
    return field, theta, mask, area


def test_iso_integral_matches_legacy(data):
    field, theta, mask, area = data
    out, ar = iso_integral(field[0], theta, mask, area, **TH)
    ref_out, ref_ar = legacy.iso_integral(field[0], theta, mask, area, **TH)
    np.testing.assert_allclose(out, ref_out, atol=1e-12)
    np.testing.assert_allclose(ar, ref_ar, atol=1e-12)


def test_iso_multi_integral_matches_legacy(data):
    field, theta, mask, area = data
    out, ar = iso_multi_integral(field, theta, mask, area, **TH)
    ref_out, ref_ar = legacy.iso_multi_integral(field, theta, mask, area, **TH)
    np.testing.assert_allclose(out, ref_out, atol=1e-12)
    np.testing.assert_allclose(ar, ref_ar, atol=1e-12)


def test_iso_multicell_integral_matches_legacy(data):
    field, theta, mask, area = data
    nf, nz = field.shape[:2]
    f = field.reshape(nf, nz, -1)
    th = theta.reshape(nz, -1)
    m = mask.reshape(nz, -1)
    a = area.reshape(-1)
    out, ar = iso_multicell_integral(f, th, m, a, **TH)
    ref_out, ref_ar = legacy.iso_multicell_integral(f, th, m, a, **TH)
    np.testing.assert_allclose(out, ref_out, atol=1e-12)
    np.testing.assert_allclose(ar, ref_ar, atol=1e-12)


def test_conservation(data):
    field, theta, mask, area = data
    out, ar = iso_multi_integral(field, theta, mask, area, **TH)
    a = np.where(mask, 0.0, area)
    np.testing.assert_allclose(ar.sum(-1), a.sum((1, 2)))
    np.testing.assert_allclose(out.sum(-1), (field * a).sum((2, 3)))


def test_nan_theta_is_excluded(data):
    field, theta, mask, area = data
    theta = theta.copy()
    field = field.copy()
    theta[mask] = np.nan
    field[:, mask] = np.nan
    out, ar = iso_multi_integral(field, theta, mask, area, **TH)
    assert np.all(np.isfinite(out)) and np.all(np.isfinite(ar))
    np.testing.assert_allclose(ar.sum(-1), np.where(mask, 0.0, area).sum((1, 2)))


def test_linear_split_between_bins():
    theta = np.array([[[301.25]]])  # a quarter of the way from 301 to 302
    out, ar = iso_integral(np.array([[[2.0]]]), theta, False, 1.0, th_min=300.0, th_max=310.0, nbin=11)
    np.testing.assert_allclose(ar[0, 1:3], [0.75, 0.25])
    np.testing.assert_allclose(out[0, 1:3], [1.5, 0.5])


def test_iso_average():
    nt, nz, nbin = 4, 2, 3
    area = np.ones((nt, nz, nbin))
    area[:, :, 2] = 0.0  # empty bin falls back to the profile
    field_sum = 3.0 * area
    profile = np.array([10.0, 20.0])
    av = iso_average(field_sum, area, profile, dt=2)
    assert av.shape == (2, nz, nbin)
    np.testing.assert_allclose(av[:, :, :2], 3.0)
    np.testing.assert_allclose(av[:, :, 2], [[10.0, 20.0]] * 2)
