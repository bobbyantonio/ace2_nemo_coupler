# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3.11.10-01
#     language: python
#     name: python-3.11.10-01
# ---

# %%
import os
import numpy as np
import xarray as xr
from glob import glob

import matplotlib.pyplot as plt

# %%
nemo_results_dir = '/home/ecme4254/scratch/run_dir/n3.6_ace2_1951_spinupCMIP6_19710101-19810101_m0'

# %% [markdown]
# ## Investigate output files after a crash

# %%
for n in range(32):
    ds = xr.load_dataset(os.path.join(nemo_results_dir, f'output.abort_{n:04d}.nc'))
    if ds['sossheig'].max().item() > 1:
        print(n, ds['sossheig'].max().item())

# %%
ix = 9
xr.load_dataset(os.path.join(nemo_results_dir, 'output.abort_0026.nc')).isel(time_counter=0)['sossheig'].plot(x='nav_lon', y='nav_lat')

# %%
err_ds.isel(time_counter=0)['sinflx'].plot(x='nav_lon', y='nav_lat')

# %%
err_ds.isel(time_counter=0)['sithicat'].sel(ncatice=5).plot(x='nav_lon', y='nav_lat')

# %%
err_ds.isel(time_counter=0)['sossheig'].plot(x='nav_lon', y='nav_lat')

# %%
sst_da = xr.open_dataset('/home/ecme4254/scratch/run_dir/n3.6_ace2_19510101-19610101_interp_na_m0/router/oce2atm_18h_ace2_nemo.nc')

# %%
land_mask = np.isnan(sst_da['sea_surface_temperature'].isel(time=0))

# %%
from scipy.ndimage import uniform_filter

# %%
atm2oce_ds = xr.open_dataset("/home/ecme4254/scratch/run_dir/n3.6_ace2_19510101-19610101_crash_experiment_m0/router/atm2oce_19510222-06_ace2_nemo.nc")

# %%

# %%
land_mask.sel(longitude=np.arange(200,350,1), method='nearest').sel(latitude=np.arange(60,90,1), method='nearest').plot(x='longitude', y='latitude')

# %%
filtered_land_mask = land_mask.copy()
filtered_land_mask.values = uniform_filter(land_mask.values.astype(np.float32), size=5)
filtered_land_mask.sel(longitude=np.arange(200,350,1), method='nearest').sel(latitude=np.arange(60,90,1), method='nearest').plot(x='longitude', y='latitude')

# %%
all_fps = glob(os.path.join(nemo_results_dir, 'nemo_ocean_output_grid_T_*.nc'))

# %%
grid_ds = xr.open_dataset('/home/ecme4254/hpcperm/run_dir/NCPL/grids.nc')

# %%
maskutil = xr.open_dataset('/home/ecme4254/hpcperm/ece3data/nemo/domain/ORCA1/maskutil.nc')

# %%
# Regrid data and save it

dt = datetime.datetime(2010,1,1)

for var in ['mean_surface_latent_heat_flux']:
    for n in range(31):
        os.makedirs(os.path.join('/ec/res4/hpcperm/ecme4254/era5_ORCAT', 'surface', var), exist_ok=True)
        era5_ds = xr.load_dataset(os.path.join('/ec/res4/hpcperm/ecme4254/era5', 'surface', var, f"era5_{var}_{(dt+datetime.timedelta(days=n)).strftime('%Y%m%d')}.nc"))
        regridder(era5_ds).to_netcdf(os.path.join('/ec/res4/hpcperm/ecme4254/era5_ORCAT', 'surface', var, f"era5_{var}_{(dt+datetime.timedelta(days=n)).strftime('%Y%m%d')}.nc"))

# %%
regridder

# %%
rmp_ds = xr.open_dataset('/hpcperm/ecme4254/run_dir/NCPL/rmp_ORCA1-T_to_ORCA1-T_BILINEAR.nc')

# %%
for dv in rmp_ds.data_vars:
    print(dv)
    print(rmp_ds[dv].values)
    
    if dv.startswith('src'):
        print((rmp_ds[dv].values - rmp_ds[dv.replace('src', 'dst')].values).max())

# %%
rmp_ds['dst_grid_center_lat'].values

# %%
rmp_ds

# %%
oasis_ds = xr.open_dataset('/hpcperm/ecme4254/run_dir/NCPL/A_Qs_mix_GenCast_03.nc')

# %%
oasis_ds

# %%

# %%
import matplotlib.pyplot as plt
plt.imshow(oasis_ds['A_Qs_mix'].isel(time=3).values.reshape(362,292))

# %%
