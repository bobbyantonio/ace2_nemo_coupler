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
import os, sys
import gc
import subprocess
import string
import pickle
import datetime
from tqdm import tqdm
import numpy as np
import xarray as xr
from argparse import ArgumentParser

# %%
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from pathlib import Path
from matplotlib import gridspec
import cartopy.crs as ccrs
import calendar
from itertools import chain
import cartopy.mpl.ticker as cticker
from scipy import signal

# %%
# python_path = sys.executable
# esmkfile_path = python_path.replace('bin/python', 'lib/esmf.mk')
# os.environ['ESMFMKFILE'] = esmkfile_path
import xarray_regrid
import xesmf as xe

sys.path.append('/perm/ecme4254/repos/nwp_notebooks')
from notebook_utils.misc import is_notebook
from eerie.coupled_experiments.coupling_processing_utils import detrend_dataarray, \
    convert_dts_to_first_of_month, calculate_en34 ,calculate_linear_relationship, \
    mean_areas, calculate_en34_spectra, bjerknes_feedback_analysis, calculate_nino_index, calculate_anomalies, \
    load_era5_monthly, calculate_lagged_correlations

BASE_OUTPUT_DIR = '/perm/ecme4254/repos/nwp_notebooks/eerie/coupled_experiments/processed_data'
era5_dir = "/home/ecme4254/scratch/era5_monthly"

# %%
if is_notebook():
    years=range(2015,2021)
    debug=True
    
else:
    parser = ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--years', type=str, default='1951-2021')
    args = parser.parse_args()

    debug = args.debug

    years_split = args.years.split('-')
    years = range(int(years_split[0]), int(years_split[1])+1)

    if debug:
        years = years[:10]


OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, 'ERA5')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ACE2 grid / sea mask
ace2grid = xr.load_dataset("/home/ecme4254/hpcperm/ml_model_data/ace2/grid.nc")
sea_mask = xr.load_dataarray("/home/ecme4254/perm/ece3data/era5/era5_sea_mask_ACE2.nc")

# %%
era5_vars = ['2m_temperature', 
             'total_precipitation', 
             'sea_ice_cover', 
             'mean_surface_sensible_heat_flux', 
             'mean_surface_latent_heat_flux', 
             'sea_surface_temperature',
             'mean_sea_level_pressure',
            'ice_temperature_layer_1', 
             'ice_temperature_layer_2', 
             'ice_temperature_layer_3',
             'ice_temperature_layer_4',
            '10m_u_component_of_wind',
            'mean_surface_downward_short_wave_radiation_flux'] 
era5_ds = []
for era5_var in era5_vars:
    era5_ds.append(load_era5_monthly(era5_var, era5_dir, years).regrid.linear(ace2grid))

era5_ds = xr.merge(era5_ds)
era5_time_vals = [pd.Timestamp(dt) for dt in sorted(era5_ds['time'].values)]
era5_ds['2m_temperature_sea_points'] = xr.where(sea_mask, era5_ds['2m_temperature'], np.nan)

era5_ds['sea_ice_temperature'] = (era5_ds['ice_temperature_layer_1'] + 
                                  era5_ds['ice_temperature_layer_2'] + 
                                  era5_ds['ice_temperature_layer_3'] + 
                                  era5_ds['ice_temperature_layer_4'])/4

era5_ds = era5_ds.rename({'total_precipitation': 'total_precipitation_daily',
                          'mean_sea_level_pressure': 'surface_pressure'})

era5_ds['total_precipitation_daily'] = era5_ds['total_precipitation_daily'] * 1000 # convert from m to mm
era5_ds['mean_surface_heat_flux'] = era5_ds['mean_surface_sensible_heat_flux'] + era5_ds['mean_surface_latent_heat_flux']

# %%
# oras5_vars = ['sea_surface_temperature',  'sea_ice_concentration'] 
# oras5_ds = []
# for oras5_var in oras5_vars:
#     tmp_da = load_oras5_single_level(oras5_var, oras5_dir, years)
#     oras5_ds.append(tmp_da)

# oras5_ds = xr.merge(oras5_ds, compat='override') # Use compat='override' as there are some very small differences in latitude
# regridder_oras5 = xe.Regridder(oras5_ds['sea_surface_temperature'].isel(time=0), 
#                          atmosphere_monthly_ds.isel(member=0,time=0)['2m_temperature'], 
#                          'bilinear',
#                          ignore_degenerate=True, 
#                          reuse_weights=False, 
#                          periodic=True, 
#                          filename='weights_oras5.nc')
# oras5_ds = regridder_oras5(oras5_ds)
# oras5_ds = convert_dts_to_first_of_month(oras5_ds)

# %%
time_vals = [pd.Timestamp(dt) for dt in sorted(era5_ds['time'].values)]

# %%
# Weights for calculating global averages
weights = np.cos(np.deg2rad(era5_ds.latitude))
weights = weights / weights.sum().item()

# %% [markdown]
# ## Bjerknes analysis

# %%
results_dict, anomaly_ds = bjerknes_feedback_analysis(era5_ds.copy())
        
if not debug:
    anomaly_ds[[v for v in anomaly_ds  if (v.endswith('gradient') or v.endswith('area_avg'))]].to_netcdf(os.path.join(OUTPUT_DIR, f'zonal_pacific_gradients.nc'))

if not debug:
    with open(os.path.join(OUTPUT_DIR, f'bjerknes_correlations.pkl'), 'wb+') as ofh:
        pickle.dump(results_dict, ofh)

# %% [markdown]
# ## Climate mean state

# %%
print('Calculating time means', flush=True)


time_range_dict = {'Pre-1980': [dt for dt in time_vals if dt.year <=1980],
                   'Post-1980': [dt for dt in time_vals if dt.year> 1980],
                   'All January': [dt for dt in time_vals if dt.month == 1],
                   '1st month': time_vals[:1],
                   '1st year': time_vals[:12],
                   '5th year': time_vals[48:60],
                   '1st decade': time_vals[:120],
                   'All': time_vals}

time_mean_state_dict = {}

for name, tvals in time_range_dict.items():

    time_mean_state_dict[name] = era5_ds.sel(time=tvals).mean('time')

if not debug:
    with open(os.path.join(OUTPUT_DIR, f'time_mean_state_dict.pkl'), 'wb+') as ofh:
        pickle.dump(time_mean_state_dict, ofh)

# %% [markdown]
# ## Spatial aggregations

# %%
print('Calculating spatial means', flush=True)

mean_dict = {}


for area_name, lat_dict in mean_areas.items():
    
    era5_mean_ds = era5_ds.sel(latitude=slice(lat_dict['min_lat'],lat_dict['max_lat'])).weighted(weights.sel(latitude=slice(lat_dict['min_lat'],lat_dict['max_lat']))).mean(['latitude', 'longitude']).sortby('time')
  
    # Unweighted sum, for variables that are already expressed in weighted units (e.g. ice area)
    era5_unweighted_sum_ds = era5_ds.sel(latitude=slice(lat_dict['min_lat'],lat_dict['max_lat'])).sum(['latitude', 'longitude']).sortby('time')

    mean_dict[area_name] = {'mean': era5_mean_ds,
                            'UnweightedSum': era5_unweighted_sum_ds
                           }

if not debug:
    with open(os.path.join(OUTPUT_DIR, f'mean_dict.pkl'), 'wb+') as ofh:
        pickle.dump(mean_dict, ofh)

# %% [markdown]
# ## Lagged correlations

# %%
for lag_vars in  [('mean_surface_downward_short_wave_radiation_flux', 'sea_surface_temperature'),
                    ('mean_surface_heat_flux', 'sea_surface_temperature')]:
    
    print('Calculating lagged correlations', flush=True)
    lag_var1 = lag_vars[0]
    lag_var2 = lag_vars[1]
    
    ace2_nemo_results_dict = calculate_lagged_correlations(
                                                            era5_ds.copy(), 
                                                            lag_var1, 
                                                            lag_var2, 
                                                            month_lag_max=1)
        
    
    # if not debug:
    with open(os.path.join(OUTPUT_DIR, f'lagged_correlations_max1_{lag_var1}_{lag_var2}.pkl'), 'wb+') as ofh:
        pickle.dump(ace2_nemo_results_dict, ofh)

# %% [markdown]
# ## Map of trends

# %%
# Trends for different periods

drift_vars = ['sea_surface_temperature', 
             'mean_surface_sensible_heat_flux', 'mean_surface_latent_heat_flux']

trends_time_range_dict = {'Pre-1980': [dt for dt in time_vals if dt.year <=1980],
                           'Post-1980': [dt for dt in time_vals if dt.year> 1980],
                           'All': time_vals}

trends_dict = {}

for name, tvals in time_range_dict.items():
    trends_dict[name] = {}
    if len(tvals) > 0:
        for n, varname in enumerate(drift_vars):
            _, polyfit = detrend_dataarray(era5_ds[varname].sel(time=tvals).groupby('time.year').mean(), 'year')
            # trends_dict[name][varname] = polyfit

if not debug:
    with open(os.path.join(OUTPUT_DIR, f'trends_dict.pkl'), 'wb+') as ofh:
        pickle.dump(trends_dict, ofh)

# %% [markdown]
# ## ENSO analysis

# %%
# ## Calculate ENSO index

# Niño 3.4: Average SST anomalies over (5N-5S, 170W-120W)

# %%
print('Performing ENSO analysis', flush=True)
var = 'sea_surface_temperature'

years = sorted(set(era5_ds['time.year'].values))

# %%
en34_da_era5 = calculate_en34(era5_ds['sea_surface_temperature'])
en34_da_era5_seasonal = calculate_en34(era5_ds['sea_surface_temperature'], remove_seasonal_cycle=False)

# %%
if not is_notebook():
    print(f'Saving Nino data to {OUTPUT_DIR}')
    en34_da_era5.to_netcdf(os.path.join(OUTPUT_DIR, 'nino3_4_era5.nc'))
    en34_da_era5.to_netcdf(os.path.join(OUTPUT_DIR, 'nino3_4_era5_seasonal.nc'))


# %%
from scipy.stats import t
# Currently just doing this for one enesmble member

x = en34_da_era5
y = era5_ds['total_precipitation_daily'] # Already converted to mm/day
era5_nino_stats_ds = calculate_linear_relationship(x,y)

# %%
if not is_notebook():
    print(f'Saving Nino stats data to {OUTPUT_DIR}')
    era5_nino_stats_ds.to_netcdf(os.path.join(OUTPUT_DIR, 'era5_nino3_4_stats.nc'))

# %% [markdown]
# ## Experimental area

# %%
