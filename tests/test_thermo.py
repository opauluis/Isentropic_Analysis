import numpy as np

from isentropic import thermo


def test_saturation_vapor_pressure():
    np.testing.assert_allclose(thermo.es_water(373.15), 101325.0, rtol=5e-3)
    np.testing.assert_allclose(thermo.es_water(273.15), 611.0, rtol=5e-3)
    np.testing.assert_allclose(thermo.es_ice(273.15), thermo.es_water(273.15), rtol=5e-3)
    assert thermo.es_mix(250.0) == thermo.es_ice(250.0)


def test_mixing_ratio_roundtrip():
    rv = np.array([0.001, 0.01, 0.02])
    e = thermo.vapor_pressure(9.0e4, rv)
    np.testing.assert_allclose(thermo.mixing_ratio(9.0e4, e), rv)


def test_dry_theta_e_is_temperature_at_pref():
    # rv = 0 gives 0 * log(0); use a tiny mixing ratio as the drivers do
    T = np.array([280.0, 300.0])
    np.testing.assert_allclose(thermo.equivalent_potential_temperature(thermo.Pref, T, 1e-12, 0.0, 0.0), T, rtol=1e-6)


def test_theta_e_increases_with_moisture():
    the_dry = thermo.equivalent_potential_temperature(1.0e5, 300.0, 1e-8, 0.0, 0.0)
    the_moist = thermo.equivalent_potential_temperature(1.0e5, 300.0, 0.015, 0.0, 0.0)
    assert the_moist - the_dry > 35.0


def test_saturated_entropy_triple_accepts_arrays():
    s, rv, ri = thermo.saturated_moist_entropy_triple(1.0e5, np.array([0.001, 0.01]), np.array([0.0, 0.0]))
    assert s.shape == (2,) and np.all(ri >= 0.0)
