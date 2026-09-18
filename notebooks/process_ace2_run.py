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
import datetime
from glob import glob
import pickle
import pandas as pd
import xarray as xr
import numpy as np
from pathlib import Path
from tqdm import tqdm
sys.path.append("/home/ecme4254/perm/repos/ace2_nemo_coupler")
from notebooks.coupling_processing_utils import calculate_linear_relationship, calculate_anomalies, ace2_var_lookup, is_notebook, mean_areas

# %%
BASE_OUTPUT_DIR = '/home/ecme4254/perm/repos/ace2_nemo_coupler/notebooks/processed_data'
experiment_id = 'ace2_forced_ece3P_control_70years'

if is_notebook():
    debug=True
else:
    debug=False

OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, experiment_id)
os.makedirs(OUTPUT_DIR, exist_ok=True)
sea_mask = xr.load_dataarray("/hpcperm/ecme4254/ml_model_data/ace2/era5_sea_mask_ACE2.nc")

# %%
experiment_ds = xr.open_dataset(os.path.join(f"/ec/res4/hpcperm/ecme4254/model_runs/ace2/{experiment_id}", "monthly_mean_predictions.nc"))

if debug:
    experiment_ds = experiment_ds.isel(time=slice(0, 12*5))
        
time_vals = pd.date_range(start="1951-01-01", end="2021-12-31", freq="MS")[: len(experiment_ds['time'])]
years = sorted(set(time_vals.year))

experiment_ds = experiment_ds.assign_coords(time=time_vals)
experiment_ds = experiment_ds.rename({varname: v for varname, v in ace2_var_lookup.items() 
                                      if varname in experiment_ds.variables}).rename({'lat': 'latitude', 
                                                                                      'lon': 'longitude'}).isel(sample=0).drop_vars(['init_time', 'valid_time', 'counts'])



# %%
experiment_ds['total_precipitation_daily'] = experiment_ds['total_precipitation']*86400
experiment_ds['mean_surface_heat_flux'] = experiment_ds['mean_surface_latent_heat_flux'] + experiment_ds['mean_surface_sensible_heat_flux']
experiment_ds['mean_surface_latent_heat_flux'] = -1 * experiment_ds['mean_surface_latent_heat_flux']
experiment_ds['mean_surface_sensible_heat_flux'] = -1 * experiment_ds['mean_surface_sensible_heat_flux']

# %%
# Weights for calculating global averages
weights = np.cos(np.deg2rad(experiment_ds.latitude))
weights = weights / weights.sum().item()

# %%
for var in ['mean_surface_sensible_heat_flux', 
                'mean_surface_latent_heat_flux', 
                'mean_surface_downward_short_wave_radiation_flux',
                'mean_surface_upward_short_wave_radiation_flux', 
                'mean_surface_downward_long_wave_radiation_flux',
                'mean_surface_upward_long_wave_radiation_flux'
               ]:
    experiment_ds[var] = xr.where(sea_mask, experiment_ds[var], np.nan)
    experiment_ds[f'{var}_oce'] = xr.where(ice_mask, np.nan, experiment_ds[var])

# %% [markdown]
# ## Climate mean state

# %%
time_range_dict = {'Pre-1980': [dt for dt in time_vals if dt.year <=1980],
                   'Post-1980': [dt for dt in time_vals if dt.year> 1980],
                   'All January': [dt for dt in time_vals if dt.month == 1],
                   'JJA': [dt for dt in time_vals if dt.month in [6,7,8]],
                   'DJF': [dt for dt in time_vals if dt.month in [12,1,2]],
                   '1st month': time_vals[:1],
                   '1st year': time_vals[:12],
                   '5th year': time_vals[48:60],
                   '1st decade': time_vals[:120],
                   'All': time_vals}

time_mean_state_dict = {}

for name, tvals in time_range_dict.items():

    time_mean_state_dict[name] = experiment_ds.sel(time=tvals).mean('time')

if not debug:
    with open(os.path.join(OUTPUT_DIR, f'time_mean_state_dict.pkl'), 'wb+') as ofh:
        pickle.dump(time_mean_state_dict, ofh)

# %%
## Spatial Aggregations

# %%
mean_dict = {}


for area_name, lat_dict in mean_areas.items():

    mean_ds = experiment_ds.sel(latitude=slice(lat_dict['min_lat'],lat_dict['max_lat'])).weighted(weights.sel(latitude=slice(lat_dict['min_lat'],lat_dict['max_lat']))).mean(['latitude', 'longitude']).sortby('time')

    unweighted_sum_ds = experiment_ds.sel(latitude=slice(lat_dict['min_lat'],lat_dict['max_lat'])).sum(['latitude', 'longitude']).sortby('time')

    mean_dict[area_name] = {'mean': mean_ds,
                            'UnweightedSum': unweighted_sum_ds
                           }

if not debug:
    with open(os.path.join(OUTPUT_DIR, f'mean_dict.pkl'), 'wb+') as ofh:
        pickle.dump(mean_dict, ofh)

# %%

# %%

# sst_da_dict = {}
# for k in experiment_path_dict:
#     print(f"Processing {k} run...")
    
#     sst_da_dict[k] = []
#     fps = [os.path.join(input_folder_dict[k], f"forcing_{year}.nc") for year in years]
    
#     for fp in tqdm(fps, desc="Processing ACE2 forced runs", total=len(fps)):
#         y = Path(fp).stem.split("_")[1]
        
#         ds = xr.load_dataset(fp).resample(time='MS').mean()
        
#         output_dir = os.path.join(experiment_path_dict[k], 'forcing_data')
#         os.makedirs(output_dir, exist_ok=True)
        
#         # Save surface temperature data for later use in plotting and analysis
#         sst_da_dict[k].append(ds['surface_temperature'].assign_coords(latitude=experiment_ds_dict[k].latitude, longitude=experiment_ds_dict[k].longitude))

#     sst_da_dict[k] = xr.concat(sst_da_dict[k], dim='time')
#     experiment_ds_dict[k]['sea_surface_temperature'] = sst_da_dict[k]

# %%
# def bjerknes_feedback_analysis(ds):
    
#     enso_vars_ds = ds[['sea_surface_temperature', 
#                        '10m_u_component_of_wind']].sel(longitude=slice(130, 250), latitude=slice(-15,15)).copy()
    
#     anomaly_ds = calculate_anomalies(enso_vars_ds).transpose('time', 'latitude', 'longitude')

#     for var in ['sea_surface_temperature']:
    
#         anomaly_ds[f'{var}_gradient'] = anomaly_ds[var].sel(longitude=slice(220, 250), 
#                                                             latitude=slice(-5,5)).mean(['longitude', 'latitude']) - anomaly_ds[var].sel(longitude=slice(130, 160), latitude=slice(-5,5)).mean(['longitude', 'latitude'])
#         anomaly_ds[f'{var}_gradient'] = anomaly_ds[f'{var}_gradient'] / ( ( 235 - 145) * 111.32 * 1000) # Result is in K/m

#     anomaly_ds['10m_u_component_of_wind_area_avg'] = anomaly_ds['10m_u_component_of_wind'].sel(latitude=slice(-5,5)).mean(['longitude', 'latitude'])
    
#     ###########
#     results_dict = {}
#     for comparison_vars in [
#                             ['sea_surface_temperature_gradient', '10m_u_component_of_wind'],
#                            ]:
    
#         cvar1 = comparison_vars[0]
#         cvar2 = comparison_vars[1]
    
#         results_dict[f'{cvar1}__{cvar2}'] = calculate_linear_relationship(anomaly_ds[cvar1], anomaly_ds[cvar2])
#         results_dict[f'{cvar2}__{cvar1}'] = calculate_linear_relationship(anomaly_ds[cvar2], anomaly_ds[cvar1])
        
#     return results_dict, anomaly_ds

# %%
# for k, ds in experiment_ds_dict.items():
#     ds.attrs['experiment_id'] = experiment_id
#     results_dict, anomaly_ds = bjerknes_feedback_analysis(ds.copy())
            
#     if not debug:
#         print(f"Saving zonal gradient and area average variables for {k} run...")   
#         anomaly_ds[[v for v in anomaly_ds  if (v.endswith('gradient') or v.endswith('area_avg'))]].to_netcdf(os.path.join(OUTPUT_DIR, f'zonal_pacific_gradients.nc'))

#     if not debug:
#         print(f"Saving Bjerknes feedback results for {k} run...")
#         with open(os.path.join(OUTPUT_DIR, f'bjerknes_correlations_{k}.pkl'), 'wb+') as ofh:
#             pickle.dump(results_dict, ofh)
