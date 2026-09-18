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
import xarray as xr
# import xarray_regrid
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs

# %%
ace2_ic = xr.load_dataset('/home/ecme4254/hpcperm/ml_model_data/ace2/initial_conditions/ic_1950.nc').isel(time=0)
ace2_forcing = xr.load_dataset('/home/ecme4254/scratch/ace2_forcing_data/control_1951-2051/forcing_1951.nc')

# %%
ace2_forcing['global_mean_co2'].plot()

# %%
era5_sst = xr.open_dataset("/home/ecme4254/scratch/era5/surface/sea_surface_temperature/era5_sea_surface_temperature_19500101.nc").isel(time=0)
era5_skin_temp = xr.open_dataset("/home/ecme4254/scratch/era5/surface/skin_temperature/era5_skin_temperature_19500101.nc").isel(time=0)

era5_ds = xr.merge([era5_sst, era5_skin_temp])

# %%
era5_ds = era5_ds.regrid.linear(ace2_ic, time_dim=None)

# %%
fig, ax = plt.subplots(1,3, figsize=(3*9,6), subplot_kw = {'projection': ccrs.Robinson(central_longitude=180.0)})
(era5_ds['sst'] - ace2_ic['surface_temperature']).plot(ax=ax[0], transform=ccrs.PlateCarree())

# %%

# %%

# %%
(ace2_forcing_mean['ocean_fraction'] > 0.1).plot()

# %%
ace2_sea_mask = np.logical_and((ace2_forcing_mean['land_fraction'] <0.5), np.logical_or((ace2_forcing_mean['ocean_fraction'] > 0.0),(ace2_forcing_mean['sea_ice_fraction'] > 0.0)))

ace2_sea_mask.to_netcdf('/home/ecme4254/hpcperm/ece3data/era5/era5_sea_mask_ace2.nc')
