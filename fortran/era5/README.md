# ERA5 isentropic analysis (Fortran)

Daily isentropic analysis of ERA5 hourly pressure-level data (RDA ds633.0, 0.25°) in
equivalent potential temperature coordinates. For one day, the program

1. reads T, Q, W (omega), Z and the four condensates (CLWC, CRWC, CIWC, CSWC) on the levels from
   150 to 1000 hPa, for all 24 hours;
2. computes theta_e, moist entropy and enthalpy with liquid water as reference (`iso_thermo.F90`,
   same constants and formulas as `isentropic/thermo.py`, Flatau et al. 1992 saturation vapor pressure);
3. splits omega into zonal mean, zonal wavenumbers 1–3, 4–10 and the remainder (`zonal_filter.F90`);
4. bins area, mass flux and tracers in theta_e (200–380 K, 1 K) on 3°×3° boxes (12 longitudes,
   13 latitudes with the edge rows shared), leaving out points below the surface (Z < Z_sfc);
5. writes the daily means and the streamfunctions STR*_THE(theta_e, p), the cumulative sums over
   theta_e of the mass flux.

## Build and run

```
module load intel netcdf     # NCAR Casper
make FC=ifort                # or: make  (gfortran)
./era5_iso.exe 2020 1 15 [OUT_DIR]
```

Output goes to `OUT_DIR/YYYY/ERA5_STR_THE.YYYY_MM_DD.nc` (netCDF-4, deflated), with
`OUT_DIR` defaulting to `/glade/derecho/scratch/pauluis/ERA5/ISO/THETAE/`. The input
locations default to the Casper RDA paths and can be changed with the environment variables
`ERA5_PL_DIR` (directory holding the `YYYYMM/e5.oper.an.pl.*` files) and `ERA5_ZS_FILE`
(surface geopotential). Grid sizes, levels and number of hours are read from the files.
The run needs about 4 GB of memory.

## Output

| variable | dims (netCDF order) | content |
|---|---|---|
| `AREA_THE` | day, press, lat, lon, theta_e | area (m²) |
| `MFC_THE` | same | mass flux omega · area / g (kg s⁻¹, positive downward), mean over each pressure level removed |
| `MFCNV_THE` | same | mass flux, mean over each column removed |
| `RHOT_THE`, `RHOQ_THE`, `RHOZ_THE`, `RHOS_THE`, `RHOH_THE`, `RHOQL_THE`, `RHOQI_THE` | same | area-weighted T, q, gZ, s, h, q_l, q_i |
| `RHOWQ_THE`, `RHOWQL_THE`, `RHOWQI_THE` | same | water fluxes omega·q·area/g, omega·q_l·area/g, omega·q_i·area/g (kg s⁻¹) |
| `MF0_THE`, `MF13_THE`, `MF410_THE`, `MFSW_THE` | day, press, lat, theta_e | mass flux of the omega components (kg s⁻¹); global mean (MF0) or latitude-band mean removed |
| `STR_THE`, `STR0_THE`, `STR13_THE`, `STR410_THE`, `STRSW_THE` | day, press, theta_e | cumulative sums over theta_e of the above |

Latitude follows the ERA5 order (north to south); `lat` and `lon` are the box centres.

## Changes from `Read_ERA_NCAR.F90` (2026)

These change the output relative to files written by the original code:

- output names follow the content: `MFC_THE` (level mean removed, was written as `MF_THE`) and
  `MFCNV_THE` (column mean removed, was written as `MFC_THE`);
- mass and water fluxes are divided by g (kg s⁻¹ instead of Pa m² s⁻¹);
- constants of `isentropic/thermo.py` (Cpd 1007, Rd 287.04, Rv 461.5, Lv0 2.5e6, Lf0 3.337e5;
  were 1004, 287, 461, 2.5104e6, 3.336e5);
- ice entropy `Cice ln(T/Tref) - Lf0/Tref` per unit ice, as in `thermo.py` (was `- Lf0/T`);
- enthalpy: the ice term is `-LF0 r_i` (was `-LF0 r_i / T`), affects `RHOH_THE`;
- `MFCNV_THE` removes only the column mean (it also removed the level mean before, so it did not sum
  to zero over theta_e in each column);
- `lat` and `lon` are the actual box centres (before: latitude reversed, longitude negative);
- grid-cell area uses the latitude spacing 180°/(NLAT−1) (was 180°/NLAT, 0.14% low at 0.25°);
- netCDF-4 output with deflation instead of classic format.

With the old constants and formulas, everything else agrees with the original to single precision
(checked on a synthetic day).
Also changed: the wavenumber filter projects onto the cos/sin coefficients, O(n) instead of the O(n²)
convolution; allocatable arrays (no `-mcmodel medium`); paths from arguments/environment;
`get_command_argument`.

`tests/test_fortran_era5.py` builds the program, runs it on a small synthetic day and compares
with a numpy version of the same algorithm (skipped without gfortran and netCDF-Fortran).
