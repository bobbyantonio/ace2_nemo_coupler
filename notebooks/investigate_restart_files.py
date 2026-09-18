# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import os
import xarray as xr
import numpy as np
from glob import glob

import xesmf as xe
import matplotlib.pyplot as plt

BASE_FOLDER = '/home/ecme4254/perm/ece3data/nemo'

# %% editable=true slideshow={"slide_type": ""}
ace2_grid_da = xr.load_dataarray("/hpcperm/ecme4254/ml_model_data/ace2/grid.nc")

# %%
era5_sst_1951 = xr.open_dataarray("/home/ecme4254/scratch/era5/surface/sea_surface_temperature/era5_sea_surface_temperature_19510101.nc").isel(time=0)

# %%
era5_regridder = xe.Regridder(era5_sst_1951,
                         ace2_grid_da, 
                         'bilinear',
                         ignore_degenerate=True, 
                         reuse_weights=False, 
                         periodic=True, 
                         filename='era5_weights.nc')

# %%
era5_sst_1951 = era5_regridder(era5_sst_1951)

# %%
sea_mask = ~np.isnan(era5_sst_1951)

# %%

years =[1951, 1995, 2000]

oce_restart_ds_dict = {}
ice_restart_ds_dict = {}
for y in years:

    ice_fp = glob(os.path.join(BASE_FOLDER, 'restart', 'ORCA1', f'{y}0101', '*_ice.nc'))[0]
    oce_fp = glob(os.path.join(BASE_FOLDER, 'restart', 'ORCA1', f'{y}0101', '*_oce.nc'))[0]

    oce_restart_ds_dict[y] = xr.open_dataset(oce_fp)[['tn', 'nav_lon', 'nav_lat']]
    ice_restart_ds_dict[y] = xr.open_dataset(ice_fp)

    oce_restart_ds_dict[y] = xr.where(oce_restart_ds_dict[y]  == 0., np.nan, oce_restart_ds_dict[y])

    oce_restart_ds_dict[y] = oce_restart_ds_dict[y].assign_coords({'longitude': oce_restart_ds_dict[y]['nav_lon'], 'latitude': oce_restart_ds_dict[y]['nav_lat']})
    ice_restart_ds_dict[y] = ice_restart_ds_dict[y].assign_coords({'longitude': oce_restart_ds_dict[y]['nav_lon'], 'latitude': oce_restart_ds_dict[y]['nav_lat']})

# %%
pablo_restart = xr.load_dataset("/home/ecme4254/perm/ece3data/nemo/restart/ORCA1/Pablo_test/a3jp_00011680_restart_oce.nc")['tn'].sel(z=0).isel(t=0)
restart_1951 = xr.load_dataset("/home/ecme4254/perm/ece3data/nemo/restart/ORCA1/19510101/restart_oce.nc")['tn'].sel(z=0).isel(t=0)

# %%
regridder = xe.Regridder(oce_restart_ds_dict[y]['tn'].isel(t=0, z=0),
                         ace2_grid_da, 
                         'bilinear',
                         ignore_degenerate=True, 
                         reuse_weights=False, 
                         periodic=True, 
                         filename='weights.nc')

# %%
oce_restart_ds_dict = {y: regridder(da) for y, da in oce_restart_ds_dict.items()}

# %%
# Load ACE2 forcing files

# %%
ace2_forcing = xr.load_dataset("/scratch/ecme4254/ace2_forcing_data/control_1951-2051/forcing_1951.nc")

# %%
clim_sst_da = xr.load_dataarray('/home/ecme4254/hpcperm/era5/climatology/mean_sea_surface_temperature_1979-01-01__2018-12-31.nc').sel(dayofyear=1).rename({'latitude': 'lat', 'longitude': 'lon'})
clim_sst_da = clim_sst_da.sel(lat=np.arange(-90,91,1), lon=np.arange(0,360))
regridder = xe.Regridder(oce_restart_ds_dict[y]['tn'].sel(z=0), 
                         clim_sst_da, 'bilinear',
                         ignore_degenerate=True, reuse_weights=False, 
                         periodic=True, filename='weights.nc')

# %%
from tqdm import tqdm

fig, axs = plt.subplots(1, len(years), figsize=(len(years) *8,len(years)))

for n, y in tqdm(enumerate(years)):

    temp_diff = (oce_restart_ds_dict[y]['tn'].sel(z=0).isel(t=0) + 273 - era5_sst_1951)
    xr.where(sea_mask, temp_diff, np.nan).sel(latitude=slice(-60,60)).plot(ax=axs[n], x='longitude', y='latitude', vmin=-10, vmax=10, cmap='RdBu_r')
    axs[n].set_title(f"Restart SST ({y}) - ERA5 SST (1951)")

# %%
