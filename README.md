# Isentropic_Analysis

Python tools for the isentropic analysis of convective motions (Pauluis and Mrowiec 2013)
and the Mean Air Flow As Lagrangian Dynamics Approximation, MAFALDA (Pauluis 2016;
Pauluis and Zhang 2017).

## Install

```
pip install -e .            # core (numpy only)
pip install -e ".[io,test]" # plus xarray/netCDF/intake for the scripts, and pytest
```

## Contents

- `isentropic/thermo.py`: thermodynamic constants and functions (saturation vapor pressure,
  moist entropy, equivalent potential temperature, Gibbs free energies, ...). Mixing ratios
  are per unit mass of dry air, pressure in Pa.
- `isentropic/binning.py`: conditional sums in (z, theta_e) coordinates
  (`iso_integral`, `iso_multi_integral`, `iso_multicell_integral`) and time-averaged
  conditional means (`iso_average`).
- `scripts/cloud_botany_iso.py`: isentropic binning of one member/time of the Cloud Botany
  DALES ensemble (EUREC4A intake catalog).

```
python scripts/cloud_botany_iso.py MEMBER TIME [SAVE_DIR]
```

## Tests

```
pytest
```

The binning tests compare against the original loop implementation kept in
`tests/reference/iso_analysis_legacy.py`.

## References

- Pauluis, O. M., and A. A. Mrowiec, 2013: Isentropic analysis of convective motions.
  *J. Atmos. Sci.*, 70, 3673–3688.
- Pauluis, O. M., 2016: The mean air flow as Lagrangian dynamics approximation and its
  application to moist convection. *J. Atmos. Sci.*, 73, 4407–4425.
- Pauluis, O. M., and F. Zhang, 2017: Reconstruction of thermodynamic cycles in a
  high-resolution simulation of a hurricane. *J. Atmos. Sci.*, 74, 3367–3381.
