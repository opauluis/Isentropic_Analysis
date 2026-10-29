! Moist thermodynamics for the ERA5 isentropic analysis
! Olivier Pauluis
!
! Constants and formulas are those of isentropic/thermo.py, except for the saturation vapor pressure.
! Mixing ratios are per unit mass of dry air, computed from the ERA5 specific humidities.
! Entropy and enthalpy use liquid water at TREF as reference state; the equivalent potential
! temperature is theta_e = TREF * exp(s / (CPAIR + r_t CL)).
! Pressure in Pa, temperature in K, entropy in J/kg/K, enthalpy in J/kg (per kg of dry air).

module iso_thermo
  implicit none
  private
  public :: moist_thermo, esat
  public :: CPAIR, RD, CPV, CL, CI, LF0, LV0, TREF, RVAP, PREF

  integer, parameter :: dp = kind(1.0d0)

  real(dp), parameter :: CPAIR = 1007.0_dp
  real(dp), parameter :: RD = 287.04_dp
  real(dp), parameter :: CPV = 1880.0_dp
  real(dp), parameter :: CL = 4218.0_dp
  real(dp), parameter :: CI = 2106.0_dp
  real(dp), parameter :: LF0 = 333.7e3_dp
  real(dp), parameter :: LV0 = 2500.0e3_dp
  real(dp), parameter :: TREF = 273.15_dp
  real(dp), parameter :: RVAP = 461.5_dp
  real(dp), parameter :: PREF = 1.0e5_dp

contains

  ! theta_e, entropy and enthalpy from specific humidities q (vapor), ql (liquid), qi (ice)
  elemental subroutine moist_thermo(theta_e, s, h, temp, q, ql, qi, p)
    real, intent(out) :: theta_e, s, h
    real, intent(in) :: temp, q, ql, qi, p

    real(dp) :: t, qt, e, es, pd, r, rl, ri, rt, cp, cp0, lv

    t = temp
    qt = real(q, dp) + ql + qi
    r = q / (1.0_dp - qt)
    rl = ql / (1.0_dp - qt)
    ri = qi / (1.0_dp - qt)
    rt = r + rl + ri

    es = 100.0_dp * esat(temp)
    e = p * RVAP * r / (RD + RVAP * r)
    pd = p - min(e, es)
    lv = LV0 + (t - TREF) * (CPV - CL)
    cp = CPAIR + (r + rl) * CL + ri * CI
    cp0 = CPAIR + rt * CL

    s = real(cp * log(t / TREF) - RD * log(pd / PREF) + lv * r / t - LF0 * ri / TREF &
         - RVAP * r * log(e / es))
    theta_e = real(TREF * exp(s / cp0))
    h = real(cp * (t - TREF) + lv * r - LF0 * ri)
  end subroutine moist_thermo

  ! Saturation vapor pressure over liquid water (hPa), polynomial fit of Flatau et al. (1992),
  ! held constant below -80C
  elemental real(dp) function esat(t)
    real, intent(in) :: t
    real(dp), parameter :: a0 = 6.11239921_dp, a1 = 0.443987641_dp, a2 = 0.142986287e-1_dp, &
         a3 = 0.264847430e-3_dp, a4 = 0.302950461e-5_dp, a5 = 0.206739458e-7_dp, &
         a6 = 0.640689451e-10_dp, a7 = -0.952447341e-13_dp, a8 = -0.976195544e-15_dp
    real(dp) :: dt

    dt = max(-80.0_dp, t - 273.16_dp)
    esat = a0 + dt*(a1 + dt*(a2 + dt*(a3 + dt*(a4 + dt*(a5 + dt*(a6 + dt*(a7 + a8*dt)))))))
  end function esat

end module iso_thermo
