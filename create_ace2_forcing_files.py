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
from tqdm import tqdm
import numpy as np
import pandas as pd
import xarray as xr

# %% [markdown]
# ## Create constant 1951 forcing for 100 years

# %%
output_folder="/home/ecme4254/scratch/ace2_forcing_data/control_1951-2051"
os.makedirs(output_folder, exist_ok=True)

# %%
ds_1951 = xr.open_dataset("/home/ecme4254/scratch/ace2_forcing_data/historical_1951-2021/forcing_1951.nc")

# %%
mean_1951_co2 = ds_1951['global_mean_co2'].mean().item()

# %%
for y in tqdm(range(1951, 2052)):
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

# %%
