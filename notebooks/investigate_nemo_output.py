# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3.12.9-01
#     language: python
#     name: python-3.12.9-01
# ---

# %%
import os
import numpy as np
import xarray as xr
from glob import glob

import matplotlib.pyplot as plt

# %%
nemo_results_dir = '/home/ecme4254/scratch/run_dir/n3.6_ace2_spinupCMIP6_laplacian_20060101-20110101_m0'

# %%
ace2_land_mask = xr.load_dataarray("/hpcperm/ecme4254/ml_model_data/ace2/era5_sea_mask_ACE2.nc")


# %%
def get_location_of_maxima(da):

    max_loc_da = da.where(da==da.max(), drop=True).squeeze()

    return [(v.name, v.item()) for v in max_loc_da.coords.values()]


# %%
max_loc_da = ds['sossheig'].where(ds['sossheig']==ds['sossheig'].max(), drop=True).squeeze()

# %%
[(v.name, v.item()) for v in max_loc_da.coords.values()]

# %% [markdown]
# ## Investigate output files after a crash

# %%
ds = xr.load_dataset(os.path.join(nemo_results_dir, f'output.abort_0001.nc'))

# %%
[{dv: ds[dv].attrs['standard_name']} for dv in list(ds.data_vars)]

# %%
error_variable = 'vozocrtx'

for n in range(32):
    ds = xr.load_dataset(os.path.join(nemo_results_dir, f'output.abort_{n:04d}.nc'))

    # if 'deptht' in ds.dims:
    #     ds = ds.isel(deptht=0)

    if ds[error_variable].max().item() > 1:
        print(n, ds[error_variable].max().item())
        print(get_location_of_maxima(ds[error_variable]))

# %%
ds = xr.load_dataset(os.path.join(nemo_results_dir, 'output.abort_0005.nc')).isel(time_counter=0)

# %%
ds[error_variable].plot(x='nav_lon', y='nav_lat')

# %%
ds['sinflx'].plot(x='nav_lon', y='nav_lat')

# %%
max_loc = ds['sinflx'].where(ds['sinflx']==ds['sinflx'].max(), drop=True).squeeze()
lat_val = max_loc['nav_lat'].values
lon_val = max_loc['nav_lon'].values

if lon_val < 0:
    lon_val = lon_val + 360


# %%
ds['sossheig'].where(ds['sossheig']==ds['sossheig'].max(), drop=True).squeeze()

# %%
ds['sinflx'].where(ds['sinflx']==ds['sossheig'].max(), drop=True).squeeze()

# %%
min_lat, max_lat = ds['nav_lat'].min().item(),ds['nav_lat'].max().item()
min_lon, max_lon = ds['nav_lon'].min().item(),ds['nav_lon'].max().item()
if min_lon < 0:
    min_lon = min_lon + 360

if max_lon < 0:
    max_lon = max_lon + 360

# %%
final_atm2oce_file = sorted(glob(os.path.join(nemo_results_dir,"router/atm2oce_*_ace2_nemo.nc")))[-1]
final_oce2atm_file = sorted(glob(os.path.join(nemo_results_dir,"router/oce2atm_*_ace2_nemo.nc")))[-1]

atm2oce_ds = xr.load_dataset(final_atm2oce_file)
oce2atm_ds = xr.load_dataset(final_oce2atm_file)

atm2oce_ds['A_Qns_oce_smoothed'] = atm2oce_ds['A_Qns_oce'].copy()
atm2oce_ds['A_Qns_oce_smoothed'].values = uniform_filter(atm2oce_ds['A_Qns_oce_smoothed'], 3)
    
atm2oce_ds['A_Qns_ice_smoothed'] = atm2oce_ds['A_Qns_ice'].copy()
atm2oce_ds['A_Qns_ice_smoothed'].values = uniform_filter(atm2oce_ds['A_Qns_ice_smoothed'], 3)
    

# %%
A_vars =[v for v in atm2oce_ds.data_vars if v.startswith('A_')]

# %%
# Value of fluxes at problem point
for v in A_vars:
    print(v, atm2oce_ds[v].sel(latitude=lat_val, longitude=lon_val, method='nearest').values)

# %%
(atm2oce_ds['A_Qns_oce'] > 1000).plot(x='longitude', y='latitude')

# %%
atm2oce_ds = atm2oce_ds.sel(latitude=slice(min_lat, max_lat), longitude=slice(min_lon, max_lon))
oce2atm_ds = oce2atm_ds.sel(latitude=slice(min_lat, max_lat), longitude=slice(min_lon, max_lon))

# %%
oce2atm_ds['sea_ice_fraction'].sel(latitude=slice(lat_val-2, lat_val+2), longitude=slice(lon_val-2, lon_val+2))

# %%
from scipy.ndimage import uniform_filter

# %%
atm2oce_ds['A_Qns_oce'].values = uniform_filter(atm2oce_ds['A_Qns_ice'], 3)

# %%
atm2oce_ds['A_Qns_ice'].plot()

# %%
(oce2atm_ds['sea_ice_fraction']>0.1).plot()

# %%
(atm2oce_ds['A_Qns_oce']).sel(latitude=slice(lat_val-2, lat_val+2), longitude=slice(lon_val-2, lon_val+2))

# %%
(atm2oce_ds['A_Qns_ice']).sel(latitude=slice(lat_val-2, lat_val+2), longitude=slice(lon_val-2, lon_val+2)).plot(x='longitude', y='latitude') 

# %%
# Plot the non-solar fluxes

# %%
# Plot the fluxes
for v in A_vars:
    fig, ax = plt.subplots(1,1)
    atm2oce_ds.isel(time=0)[v].plot(x='longitude', y='latitude', ax=ax)

# %%
xr.where(ds['sinflx']<-300,ds['sinflx'],np.nan)

# %%
xr.where(
         <-300,ds['sinflx'],np.nan).plot(x='nav_lon', y='nav_lat')

# %%
da = ds['sinflx']
da.where(da==da.min(), drop=True).squeeze()

# %%
atm2oce_ds = xr.load_dataset("/home/ecme4254/scratch/run_dir/n3.6_ace2_spinupCMIP6_laplacian_19610101-19710101_m0/router/atm2oce_19690323-12_ace2_nemo.nc")
oce2atm_ds = xr.load_dataset("/home/ecme4254/scratch/run_dir/n3.6_ace2_spinupCMIP6_laplacian_19610101-19710101_m0/router/oce2atm_72090h_ace2_nemo.nc")

# %%
oce2atm_ds.sel(longitude=slice(320,330), latitude=slice(-80, -75))

# %%
360-37.26

# %%
(oce2atm_ds.sel(longitude=slice(320,330), latitude=slice(-80, -75))['sea_ice_fraction'] > 0.01).plot(x='longitude', y='latitude')

# %%

# %%
(atm2oce_ds.isel(time=0)['A_Qns_oce']<-250).plot()

# %%
atm2oce_ds.isel(time=0)['A_Qns_oce'].sel(longitude=slice(300,340), latitude=slice(-90, -60)).min()

# %%
(atm2oce_ds.isel(time=0)['A_Qns_oce']<-250).plot()

# %%
atm2oce_ds.isel(time=0)['A_Qns_oce'].sel(longitude=slice(300,340), latitude=slice(-90, -60)).plot(x='longitude', y='latitude')

# %%
for v in A_vars:
    fig, ax = plt.subplots(1,1)
    atm2oce_ds.isel(time=0)[v].plot(x='longitude', y='latitude', ax=ax)

# %%
xr.where(np.abs(nonsolar_flux_ice)>800, np.abs(nonsolar_flux_ice), np.nan).plot()

# %%
xr.where(np.abs(nonsolar_flux_ice)>400, np.abs(nonsolar_flux_oce), np.nan).plot()

# %%
ds['sithicat'].sel(ncatice=5).plot(x='nav_lon', y='nav_lat')

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
