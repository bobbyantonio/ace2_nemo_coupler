# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: ece4
#     language: python
#     name: python3
# ---

# %%
import os
import datetime
from tqdm import tqdm
import numpy as np
import pandas as pd
import xarray as xr
import xarray_regrid
import xesmf as xe

# %% [markdown]
# ## Create forcing files from an EC-Earth3 run

# %%
ece_tos_dir = "/network/group/aopp/predict/HMC005_ANTONIO_EERIE/CMIP6_data/EC-Earth3P_control-1950-3hr/tos"
ece_sice_dir = "/network/group/aopp/predict/HMC005_ANTONIO_EERIE/CMIP6_data/EC-Earth3P_control-1950-daily"
ace2_data_dir = "/network/group/aopp/predict/HMC005_ANTONIO_EERIE/ace2_data"

ace2grid = xr.load_dataset(os.path.join(ace2_data_dir, "grid.nc"))
ace2_sea_mask = xr.load_dataset(os.path.join(ace2_data_dir, "era5_sea_mask_ACE2.nc"))

# %%


def create_ace2_forcing_file(year, ece_tos_dir=ece_tos_dir, ece_sice_dir=ece_sice_dir, ace2grid=ace2grid, ace2_sea_mask=ace2_sea_mask):
    tos_ds = xr.open_dataset(os.path.join(ece_tos_dir, f'tos_3hr_EC-Earth3P_control-1950_r1i1p2f1_gn_{year}01010130-{year}12312230.nc'))['tos']

    ace2_forcing_ds = xr.open_dataset(f"/network/group/aopp/predict/HMC005_ANTONIO_EERIE/ace2_data/forcing_data/control_1951-2051/forcing_{year}.nc")

    new_date_range = ace2_forcing_ds['time'].values
    tos_ds = tos_ds.sel(time=new_date_range, method='nearest', tolerance=pd.Timedelta('1.5h'))
    tos_ds = tos_ds.assign_coords(time=new_date_range)

    tos_ds = tos_ds + 273.15  # Convert from C to K


    ace2_forcing_ds = xr.open_dataset(f"/network/group/aopp/predict/HMC005_ANTONIO_EERIE/ace2_data/forcing_data/control_1951-2051/forcing_{year}.nc")

    regridder = xe.Regridder(tos_ds.isel(time=0), 
                                    ace2grid, 
                                    'bilinear',
                                    ignore_degenerate=True, 
                                    reuse_weights=False, 
                                    periodic=True, 
                                    filename=f'weights_ece3_oce_tos.nc')
    tos_ds = regridder(tos_ds)


    # Load daily sea ice data
    siconc_da = xr.open_dataset(os.path.join(ece_sice_dir, 'siconc', f'siconc_SIday_EC-Earth3P_control-1950_r1i1p2f1_gn_{year}0101-{year}1231.nc'))['siconc']

    # Load daily sea ice data
    sitemptop_da = xr.open_dataset(os.path.join(ece_sice_dir, 'sitemptop', f'sitemptop_SIday_EC-Earth3P_control-1950_r1i1p2f1_gn_{year}0101-{year}1231.nc'))['sitemptop']

    # Change from % to fraction
    siconc_da = siconc_da / 100.0

    # sitemptop_da = sitemptop_da + 273.15  # Convert from C to K

    siconc_da['time'] = pd.date_range(f'{year}0101-00:00', f'{year}1231-00:00', freq='24h')
    sitemptop_da['time'] = pd.date_range(f'{year}0101-00:00', f'{year}1231-00:00', freq='24h')


    regridder_seaice = xe.Regridder(siconc_da.isel(time=0), 
                                    ace2grid, 
                                    'bilinear',
                                    ignore_degenerate=True, 
                                    reuse_weights=False, 
                                    periodic=True, 
                                    filename=f'weights_ece3_oce_tos.nc')
    siconc_da = regridder(siconc_da)
    sitemptop_da = regridder(sitemptop_da)


    # Fill out the daily data to match the 6-hourly time steps of the forcing data
    siconc_da = siconc_da.sel(time=new_date_range, method='ffill')
    sitemptop_da = sitemptop_da.sel(time=new_date_range, method='ffill')

    siconc_da = siconc_da.assign_coords(time=new_date_range)
    sitemptop_da = sitemptop_da.assign_coords(time=new_date_range)


    new_surf_temp_da = xr.where( np.logical_and(ace2_sea_mask['sst'], ~np.isnan(tos_ds)), (1-siconc_da)* tos_ds + siconc_da* sitemptop_da, ace2_forcing_ds['surface_temperature'])


    ocean_fraction = xr.where(np.logical_and(ace2_sea_mask['sst'], ~np.isnan(siconc_da)), 1-siconc_da, ace2_forcing_ds['ocean_fraction'])
    ocean_fraction.name = 'ocean_fraction'

    siconc_da = xr.where(~np.isnan(siconc_da), siconc_da, ace2_forcing_ds['sea_ice_fraction'])
    siconc_da.name = 'sea_ice_fraction'

    new_ace2_forcing_ds = ace2_forcing_ds.assign({'surface_temperature': new_surf_temp_da, 
                            'ocean_fraction': ocean_fraction,
                            'sea_ice_fraction': siconc_da
                            })

    return new_ace2_forcing_ds



# %%
output_dir = "/network/group/aopp/predict/HMC005_ANTONIO_EERIE/ace2_data/forcing_data/ECE3P-control"
os.makedirs(output_dir, exist_ok=True)
for year in tqdm(range(1951, 2022)):
    ds = create_ace2_forcing_file(year)
    
    ds.to_netcdf(f"{output_dir}/forcing_{year}.nc")
