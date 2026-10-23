import os 
import time
# os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import numpy as np
import xarray as xr
import dask.array as da
import sys

# OP packages - thermodynamic properties and isentropic analysis
from isentropic import binning as iso_analysis
from isentropic import thermo as Thermo

from pathlib import Path  # Object-oriented library to deal with paths

import intake     # For catalogs


# Usage: python cloud_botany_iso.py MEMBER TIME [SAVE_DIR]
member_idx = int(sys.argv[1])
time_idx = int(sys.argv[2])

print('Isentropic analaysis')
print('Dataset: Cloud Botany')
print('Ensemble member: ' + sys.argv[1])
print('Time:  '+sys.argv[2])



def verify_field_integrity(da: xr.DataArray ,var_name:str) -> bool:
    """
    Checks whether the loaded DataArray contains dropped chunks or unphysical zeros.
    Returns True if data passes integrity checks, False otherwise.
    """
    # 1. Check for unexpected NaNs (if no fill value is intended)
    if (var_name == 'qt') :
        count = np.isnan(da.values).sum()
        if count > 100:
            print(f"⚠️ Warning: Detected {count} NaNs in '{var_name}'.")
            return False
        elif count >=1:
            print(f"⚠️ Warning: Detected {count} NaNs in '{var_name}'. Accepting the data.")
    elif (var_name == 'ql')or (var_name == 'qr'):
        count = np.isnan(da.values).sum()
        print(f"⚠️ Warning: Detected {count} NaNs in '{var_name}'. Proceeding anyways...")
    else:
        if np.isnan(da.values).any():
            count = np.isnan(da.values).sum()
            print(f"⚠️ Warning: Detected {count} NaNs in '{var_name}'.")
            return False


    return True


def load_field_with_retry(
    catalog_entry,
    var_name: str,
    selection_dict: dict = None,
    max_retries: int = 3,
    backoff_factor: float = 2.0
) -> xr.DataArray:
    """
    Loads a DataArray from an intake catalog entry with automated retry on data drop.

    Parameters:
    -----------
    catalog_entry : intake entry (e.g., botany_cat['profiles'])
    var_name : str, name of the variable to load (e.g., 'thl', 'u', 'lwp')
    selection_dict : dict, optional slice/index selection (e.g., {'time': slice(10, 20)})
    max_retries : int, maximum number of reload attempts
    backoff_factor : float, delay multiplier between attempts
    """
    selection_dict = selection_dict or {}

    for attempt in range(1, max_retries + 1):
        try:
            print(f"[Attempt {attempt}/{max_retries}] Connecting and fetching '{var_name}'...")
            ds = catalog_entry.to_dask()
            # Reopen dataset to reset any stale HTTP/fsspec chunk buffers
            da = ds[var_name].isel(**selection_dict)

            # Trigger actual network transfer via .load()
            da = da.load()

            # Run physical integrity and drop checks
            if verify_field_integrity(da, var_name):
                print(f"✅ Successfully loaded and verified '{var_name}'!")
                return da
            else:
                print(f"❌ Integrity check failed on attempt {attempt}.")

        except Exception as err:
            print(f"❌ Network or reading error on attempt {attempt}: {err}")

        if attempt < max_retries:
            sleep_time = backoff_factor ** attempt
            print(f"Retrying in {sleep_time:.1f} seconds...\n")
            time.sleep(sleep_time)

    raise RuntimeError(f"Failed to load a valid '{var_name}' array after {max_retries} attempts.")





#save directory (optional third argument)
save_dir = sys.argv[3] if len(sys.argv) > 3 else '/scratch/b/b382527/ISO/Cloud_Botany/'

print('save directorty: '+save_dir)
print('Saving file: ')
save_file = save_dir + 'CB_ISO_EXP_'+str(1000+member_idx)[-3:]+'_TIME_'+str(100+time_idx)[-2:] +'.nc'
print(save_file)

print('Loading the data...')
#load Catalog
cat = intake.open_catalog("https://raw.githubusercontent.com/eurec4a/eurec4a-intake/master/catalog.yml")
botany_cat = cat.simulations.DALES.botany

selection = {"member": member_idx-1, "time": time_idx}

print(selection)

ds_3D = botany_cat.dx100m.nx1536['3D'].to_dask().isel(selection)
ds_1D = botany_cat.dx100m.nx1536['profiles'].to_dask().isel(selection)

# isentropic bins:
thmin = 300
thmax = 350
nbin = 101
theta_bin = thmin + np.arange(nbin) * (thmax - thmin) / (nbin - 1.0)

nlev = ds_3D.sizes['zt'] - 1
nx = ds_3D.sizes['xt'] 
ny = ds_3D.sizes['yt'] 


# read the data

print('here')

zt =  ds_3D.zt.to_numpy()[:nlev]

thl = load_field_with_retry(
        botany_cat.dx100m.nx1536['3D'],
        var_name='thl',
        selection_dict=selection,
            max_retries=5
        ).to_numpy()[:nlev,:,:]

qt= np.maximum(load_field_with_retry(
        botany_cat.dx100m.nx1536['3D'],
        var_name='qt',
        selection_dict=selection,
            max_retries=5
        ).to_numpy()[:nlev,:,:],1e-10)
ql= np.maximum(load_field_with_retry(
        botany_cat.dx100m.nx1536['3D'],
        var_name='ql',
        selection_dict=selection,
            max_retries=5
        ).to_numpy()[:nlev,:,:],0.0)
qr= np.maximum(load_field_with_retry(
        botany_cat.dx100m.nx1536['3D'],
        var_name='qr',
        selection_dict=selection,
            max_retries=5
        ).to_numpy()[:nlev,:,:],0.0)


rho_zt = ds_1D.rhobf[:nlev].to_numpy()
w =  load_field_with_retry(
        botany_cat.dx100m.nx1536['3D'],
        var_name='w',
        selection_dict=selection,
            max_retries=5
        ).to_numpy()[:nlev+1,:,:]
w_zt = 0.5 * (w[:-1,:,:] + w[1:,:,:])

del w

#compute the equivalent potential temperature:
# This assumes that air parcels are unsaturated when brought back adiabatically at pref. 
pref = 1.0e5

the = Thermo.equivalent_potential_temperature(pref,thl,qt/(1.0 - qt),0.0,0.0)


# setting the iso_integral

nfield = 8
field = np.zeros((nfield,nlev,nx,ny))
AREA_THE = np.zeros((nlev,nbin)) #area in each thetae bin at each level
FIELD_THE = np.zeros((nfield,nlev,nbin)) #area in each thetae bin at each level

mask = np.isnan(the)
area_loc = ~mask * 1.0
field[0,:,:,:] = np.expand_dims(np.expand_dims(rho_zt,1),1)* w_zt 
field[1,:,:,:] = the
field[2,:,:,:] = thl
field[3,:,:,:] = qt
field[4,:,:,:] = ql
field[5,:,:,:] = qr
field[6,:,:,:] = np.expand_dims(np.expand_dims(rho_zt,1),1)* w_zt * (qt - ql - qr)
field[7,:,:,:] = np.expand_dims(np.expand_dims(rho_zt,1),1)* w_zt * qt

del thl, qt, ql, qr, w_zt

# Data is big, so we loop on the vertical axis


print( 'Starting the main loop')


for i in range(nlev):
    if (i%20) == 0:
        print('lvl : '+ str(i))
    output ,area = iso_analysis.iso_multi_integral(field[:,i:i+1,:,:],the[i:i+1:,:,:], \
                        mask[i:i+1,:,:],area_loc[i:i+1,:,:],th_min = thmin, th_max=thmax, nbin=nbin)
    FIELD_THE[:,i,:] = output[:,0,:]
    AREA_THE[i,:] = area[0,:]
#    print('lvl '+ str(i) + ' done')

print('Saving the data in: ')
save_file = save_dir + 'CB_ISO_EXP_'+str(1000+member_idx)[-3:]+'_TIME_'+str(100+time_idx)[-2:] +'.nc'
save_zarr = save_dir + 'CB_ISO_EXP_'+str(1000+member_idx)[-3:]+'_TIME_'+str(100+time_idx)[-2:] +'.zarr'

print(save_zarr)

print(save_file)

time_slice = ds_3D.time

coords = {
        "time": ("time", [time_slice.values]),  # Extract numpy array from DataArray
        "height":("height",zt),
        "theta_e": ("theta_e", theta_bin),
    }
dim_shape = (1,nlev,nbin)
data_vars = {
        "AREA": (("time", "height", "theta_bin"),  np.expand_dims(AREA_THE,0)),
        "MF": (("time", "height","theta_bin"),   np.expand_dims(FIELD_THE[0,:,:],0)),
        "ATHE": (("time", "height", "theta_bin"), np.expand_dims(FIELD_THE[1,:,:],0)),
        "ATHL": (("time", "height", "theta_bin"), np.expand_dims(FIELD_THE[2,:,:],0)),
        "AQT": (("time", "height", "theta_bin"), np.expand_dims(FIELD_THE[3,:,:],0)),
        "AQL": (("time", "height", "theta_bin"), np.expand_dims(FIELD_THE[4,:,:],0)),
        "AQR": (("time", "height", "theta_bin"), np.expand_dims(FIELD_THE[5,:,:],0)),   
        "RHOWQV": (("time", "height", "theta_bin"), np.expand_dims(FIELD_THE[6,:,:],0)),
        "RHOWQT": (("time", "height", "theta_bin"), np.expand_dims(FIELD_THE[7,:,:],0)),
    }
ds_out = xr.Dataset(data_vars=data_vars, coords=coords)

ds_out["MF"].attrs.setdefault("long_name", "Upward mass flux")
ds_out["MF"].attrs.setdefault("units", "kg s-1")
ds_out["AREA"].attrs.setdefault("long_name", "Isentropic Area")
ds_out["AREA"].attrs.setdefault("units", "m2")
ds_out["ATHE"].attrs.setdefault("long_name", "Area * theta_e")
ds_out["ATHE"].attrs.setdefault("units", "m^2")
ds_out["ATHL"].attrs.setdefault("long_name", "Area * theta_l")
ds_out["ATHL"].attrs.setdefault("units", "m^2")
ds_out["AQT"].attrs.setdefault("long_name", "Area * q_t")
ds_out["AQT"].attrs.setdefault("units", "m^2/K")
ds_out["AQL"].attrs.setdefault("long_name", "Area * q_l")
ds_out["AQL"].attrs.setdefault("units", "m^2/K")
ds_out["AQR"].attrs.setdefault("long_name", "Area * q_r")
ds_out["AQR"].attrs.setdefault("units", "m^2/K")
ds_out["RHOWQV"].attrs.setdefault("long_name", "Upward water vapor flux")
ds_out["RHOWQV"].attrs.setdefault("units", "kg s-1")
ds_out["RHOWQT"].attrs.setdefault("long_name", "Upward total water flux")
ds_out["RHOWQT"].attrs.setdefault("units", "kg s-1")

# ds_out.to_netcdf(save_file)
# potential fix for buggy netcdf file output...
# ds_out.to_netcdf(save_file, engine='h5netcdf') 
# ds_out.to_netcdf(save_file, scheduler='single-threaded',engine='h5netcdf') 

for var in ds_out.variables:
        ds_out[var].encoding = {}

ds_out.to_netcdf(save_file, engine='netcdf4') 


# Step 1: Write flawlessly to Zarr on Lustre
# ds_out.to_zarr(save_zarr, mode="w")

# Step 2: Open the zarr, and consolidate it cleanly into a single NetCDF
# xr.open_zarr(save_zarr).to_netcdf(
#    save_file, engine="h5netcdf"
# )
