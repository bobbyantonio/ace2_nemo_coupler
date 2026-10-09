# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: ece4
#     language: python
#     name: ece4
# ---

# %%
import os
import datetime
import calendar
from tqdm import tqdm
import numpy as np
import pandas as pd
import xarray as xr
import xarray_regrid

# %%
import xesmf as xe

# %%
forcing_dir = "/network/group/aopp/predict/HMC005_ANTONIO_EERIE/ace2_data/forcing_data"
historical_dir = os.path.join(forcing_dir, "historical_1951-2021")

# %% [markdown]
# ## Create forcing with constant CO2 forcing but varying surface temperature

# %%
# output_folder="/home/ecme4254/scratch/ace2_forcing_data/fixedCO2_1951"
# os.makedirs(output_folder, exist_ok=True)

# historical_dir = "/home/ecme4254/scratch/ace2_forcing_data/historical_1951-2021"

# ds_1951 = xr.open_dataset(os.path.join(historical_dir, "forcing_1951.nc"))
# mean_1951_co2 = ds_1951['global_mean_co2'].mean().item()

# for y in tqdm(range(1951, 2052)):
    
#     tmp_ds = xr.open_dataset(os.path.join(historical_dir, f'forcing_{y}.nc'))
    
#     # Set co2 forcing to average over 1951
#     tmp_ds['global_mean_co2'] = tmp_ds['global_mean_co2'] * 0 + mean_1951_co2
        
#     tmp_ds.to_netcdf(os.path.join(output_folder, f'forcing_{y}.nc'))

# %% [markdown]
# ## Create constant 1951 forcing for 100 years

# %%
output_folder="/home/ecme4254/scratch/ace2_forcing_data/control-1951"
os.makedirs(output_folder, exist_ok=True)

# %%
end_year = 2101

# The historical data is downloaded from the ACE2-ERA5 Hugging Face record
ds_1951 = xr.open_dataset(f"/home/ecme4254/scratch/ace2_forcing_data/historical_1951-2021/forcing_1951.nc")
mean_1951_co2 = ds_1951['global_mean_co2'].mean().item()

# %%
for y in tqdm(range(2099, end_year + 1)):
    dts = [np.datetime64(dt, 'ns') for dt in pd.date_range(f'{y}0101-00:00', f'{y}1231-18:00', freq='6h')]
    dts_without_leap_day = [dt for dt in dts if not ((pd.Timestamp(dt).month == 2) and (pd.Timestamp(dt).day == 29))]

    assert len(dts_without_leap_day) == 1460
    
    tmp_ds = ds_1951.copy().assign_coords(time=dts_without_leap_day)

    time_independent_vars = [v for v in tmp_ds.data_vars if ('time' not in tmp_ds[v].dims)]
    time_dependent_vars = [v for v in tmp_ds.data_vars if ('time' in tmp_ds[v].dims)]
    
    time_dependent_ds = tmp_ds[time_dependent_vars]
    time_independent_ds = tmp_ds[time_independent_vars]

    if calendar.isleap(y):
        dts_with_leap_day = [dt for dt in dts if ((pd.Timestamp(dt).month == 2) and (pd.Timestamp(dt).day == 29))]
        leap_day_ds = time_dependent_ds.sel(time=pd.date_range(f'{y}0228-00:00', f'{y}0228-18:00', freq='6h')).assign_coords(time=dts_with_leap_day)
        time_dependent_ds = xr.concat([time_dependent_ds, leap_day_ds], dim='time')

        assert len(set(time_dependent_ds['time'].values)) == 1464
    else:
        assert len(set(time_dependent_ds['time'].values)) == 1460
        
    time_dependent_ds = time_dependent_ds.sortby('time', ascending=True)

    output_ds = xr.merge([time_dependent_ds, time_independent_ds])[list(ds_1951.data_vars)]
    
    # Set co2 forcing to average over 1951
    output_ds['global_mean_co2'] = output_ds['global_mean_co2'] * 0 + mean_1951_co2
        
    output_ds.to_netcdf(os.path.join(output_folder, f'forcing_{y}.nc'))

# %% [markdown]
# ## Create 70-year Samudrace forcing

# %% vscode={"languageId": "plaintext"}
output_folder="/home/ecme4254/scratch/ace2_forcing_data/samudrace_70year"
os.makedirs(output_folder, exist_ok=True)

# %%
ds_311 = xr.open_dataset("/home/ecme4254/scratch/ace2_forcing_data/historical_1951-2021/forcing_1951.nc")
mean_1951_co2 = ds_1951['global_mean_co2'].mean().item()

# %%


for y in tqdm(range(311, 311+70)):
    
    dts = [np.datetime64(dt, 'ns') for dt in pd.date_range(f'{y}0101-00:00', f'{y}1231-18:00', freq='6h')]
    dts_without_leap_day = [dt for dt in dts if not ((pd.Timestamp(dt).month == 2) and (pd.Timestamp(dt).day == 29))]

    assert len(dts_without_leap_day) == 1460
    
    tmp_ds = ds_1951.copy().assign_coords(time=dts_without_leap_day)

    time_independent_vars = [v for v in tmp_ds.data_vars if ('time' not in tmp_ds[v].dims)]
    time_dependent_vars = [v for v in tmp_ds.data_vars if ('time' in tmp_ds[v].dims)]
    
    time_dependent_ds = tmp_ds[time_dependent_vars]
    time_independent_ds = tmp_ds[time_independent_vars]

    if y%4 == 0:
        dts_with_leap_day = [dt for dt in dts if ((pd.Timestamp(dt).month == 2) and (pd.Timestamp(dt).day == 29))]
        leap_day_ds = time_dependent_ds.sel(time=pd.date_range(f'{y}0228-00:00', f'{y}0228-18:00', freq='6h')).assign_coords(time=dts_with_leap_day)
        time_dependent_ds = xr.concat([time_dependent_ds, leap_day_ds], dim='time')

        assert len(set(time_dependent_ds['time'].values)) == 1464
    else:
        assert len(set(time_dependent_ds['time'].values)) == 1460
        
    time_dependent_ds = time_dependent_ds.sortby('time', ascending=True)

    output_ds = xr.merge([time_dependent_ds, time_independent_ds])[list(ds_1951.data_vars)]
    
    # Set co2 forcing to average over 1951
    output_ds['global_mean_co2'] = output_ds['global_mean_co2'] * 0 + mean_1951_co2
        
    output_ds.to_netcdf(os.path.join(output_folder, f'forcing_{y}.nc'))

# %% [markdown]
# ## Create 70-year Samudrace forcing

# %%
output_folder="/network/group/aopp/predict/HMC005_ANTONIO_EERIE/ace2_data/forcing_data/samudrace_70year"
os.makedirs(output_folder, exist_ok=True)

# %%
ds_311 = xr.open_dataset("/network/group/aopp/predict/HMC005_ANTONIO_EERIE/ace2_data/forcing_data/samudrace/forcing_0311.nc")
time_vals = ds_311['time'].values

# %%
for y in tqdm(range(311, 311+70)):
    
    dts = [tv + datetime.timedelta(days=365) for tv in time_vals]
    
    tmp_ds = ds_311.copy().assign_coords(time=dts)

    time_independent_vars = [v for v in tmp_ds.data_vars if ('time' not in tmp_ds[v].dims)]
    time_dependent_vars = [v for v in tmp_ds.data_vars if ('time' in tmp_ds[v].dims)]
    
    time_dependent_ds = tmp_ds[time_dependent_vars]
    time_independent_ds = tmp_ds[time_independent_vars]
        
    time_dependent_ds = time_dependent_ds.sortby('time', ascending=True)

    output_ds = xr.merge([time_dependent_ds, time_independent_ds])[list(ds_311.data_vars)]
        
    output_ds.to_netcdf(os.path.join(output_folder, f'forcing_{y}.nc'))

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
ds = create_ace2_forcing_file(1951)

# %%
ds

# %%
