# Isentropic_Analysis

Python tools for the isentropic analysis of convective motions (Pauluis and Mrowiec 2013)
and the Mean Air Flow As Lagrangian Dynamics Approximation, MAFALDA (Pauluis 2016;
Pauluis and Zhang 2017).

## Install

```
pip install -e .            # core (numpy only)
pip install -e ".[io,test]" # plus xarray/netCDF/intake for the scripts, and pytest
pip install -e ".[io,notebook]"  # plus JupyterLab, ipykernel and matplotlib
```

To use the environment from a shared Jupyter (e.g. Open OnDemand), register it as a kernel:

```
python -m ipykernel install --user --name iso --display-name "Python (iso)"
```

## Contents

- `isentropic/thermo.py`: thermodynamic constants and functions (saturation vapor pressure,
  moist entropy, equivalent potential temperature, Gibbs free energies, ...). Mixing ratios
  are per unit mass of dry air, pressure in Pa.
- `isentropic/binning.py`: conditional sums in (z, theta_e) coordinates
  (`iso_integral`, `iso_multi_integral`, `iso_multicell_integral`) and time-averaged
  conditional means (`iso_average`).
- `isentropic/analysis.py`: diagnostics from the conditional sums (Pauluis and Mrowiec 2013):
  isentropic distribution `<f>` (Eq. 1), streamfunction `Psi(z, theta_e)` (Eq. 3), conditional
  means, diabatic tendency `<rho dtheta_e/dt> = -dPsi/dz` and its isentropic mean (Eq. 10),
  and the two-stream mean ascent/descent `M_up`, `M_down`, `f_up`, `f_down` (Eqs. 14-17).
- `isentropic/sam.py`: SAM 3D output to mass flux, theta_e and frozen theta_e, cloud mask
  (QN > 0.01 g/kg), and the conditional sums in both coordinates.
- `scripts/sam_iso_integrals.py`: conditional sums of every snapshot of a SAM run, written to
  one netCDF file (`scripts/sam_iso_integrals.sbatch` runs it on Torch). The diagnostics are
  made from these files by `notebooks/isentropic_diagnostics.ipynb`.
- `scripts/cloud_botany_iso.py`: isentropic binning of one member/time of the Cloud Botany
  DALES ensemble (EUREC4A intake catalog).

```
python scripts/sam_iso_integrals.py RUN_DIR RUN_NAME [-o OUTPUT] [--times START:STOP:STEP]
python scripts/cloud_botany_iso.py MEMBER TIME [SAVE_DIR]
```

## Example

```python
import numpy as np
import xarray as xr
import isentropic as iso

ds = xr.open_mfdataset('CB_ISO_EXP_001_TIME_*.nc')   # output of scripts/cloud_botany_iso.py
area = ds.AREA.sum('time').values                    # sum over times = time average
mf = ds.MF.sum('time').values
z = ds.height.values

psi = iso.streamfunction(mf, area)                   # kg m-2 s-1
the_mean = iso.conditional_mean(ds.ATHE.sum('time').values, area)
ts = iso.two_stream(mf, area, {'qt': ds.RHOWQT.sum('time').values})
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
