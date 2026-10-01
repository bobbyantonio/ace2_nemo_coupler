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
import xarray_regrid
import numpy as np
from pathlib import Path
from tqdm import tqdm
import xesmf as xe
from itertools import chain

sys.path.append("/home/ecme4254/perm/repos/ace2_nemo_coupler")
from notebooks.coupling_processing_utils import calculate_linear_relationship, calculate_anomalies, ace2_var_lookup, ece3_var_lookup, \
convert_dts_to_first_of_month, is_notebook, mean_areas, OLEVEL_VALUES, load_ece3_data, calculate_nino_index

# %%
BASE_OUTPUT_DIR = '/home/ecme4254/perm/repos/ace2_nemo_coupler/notebooks/processed_data'
experiment_id = 'ace2_forced_ece3P_control_70years'

if is_notebook():
    debug=True
else:
    debug=False

OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, experiment_id)
os.makedirs(OUTPUT_DIR, exist_ok=True)
ace2_data_dir = "/hpcperm/ecme4254/ml_model_data/ace2"
sea_mask =  xr.load_dataset(os.path.join(ace2_data_dir, "era5_sea_mask_ACE2.nc"))
ace2grid = xr.load_dataset(os.path.join(ace2_data_dir, "grid.nc"))

# %%
atmosphere_ds = xr.open_dataset(os.path.join(f"/ec/res4/hpcperm/ecme4254/model_runs/ace2/{experiment_id}", "monthly_mean_predictions.nc"))

if debug:
    atmosphere_ds = atmosphere_ds.isel(time=slice(0, 12*5))
        
time_vals = pd.date_range(start="1951-01-01", end="2021-12-31", freq="MS")[: len(atmosphere_ds['time'])]
years = sorted(set(time_vals.year))

atmosphere_ds = atmosphere_ds.assign_coords(time=time_vals)
atmosphere_ds = atmosphere_ds.rename({varname: v for varname, v in ace2_var_lookup.items() 
                                      if varname in atmosphere_ds.variables}).rename({'lat': 'latitude', 
                                                                                      'lon': 'longitude'}).isel(sample=0).drop_vars(['init_time', 'valid_time', 'counts'])



# %%
atmosphere_ds['total_precipitation_daily'] = atmosphere_ds['total_precipitation']*86400
atmosphere_ds['mean_surface_heat_flux'] = atmosphere_ds['mean_surface_latent_heat_flux'] + atmosphere_ds['mean_surface_sensible_heat_flux']
atmosphere_ds['mean_surface_latent_heat_flux'] = -1 * atmosphere_ds['mean_surface_latent_heat_flux']
atmosphere_ds['mean_surface_sensible_heat_flux'] = -1 * atmosphere_ds['mean_surface_sensible_heat_flux']

# %%
# Weights for calculating global averages
weights = np.cos(np.deg2rad(atmosphere_ds.latitude))
weights = weights / weights.sum().item()

# %%
ocean_vars = {'t': ['tos', 'siconc', 'sithick']}
ece3_experiment_id ="EC-Earth3P_control-1950"
ece3_data_dir = f"/scratch/ecme4254/ece3_cmip6_data_download/{ece3_experiment_id}"
var_glob_string = '{var}'

# %%
ocean_ds_dict = {}
for ocean_grid_type, var_list in ocean_vars.items():
    if len(var_list) > 0:
        ocean_ds_dict[ocean_grid_type] = xr.merge([load_ece3_data(var, 
                                                 ece3_data_dir = os.path.join(ece3_data_dir, var_glob_string.format(var=var)),
                                                 years=range(1951,2022), 
                                                 level_values=OLEVEL_VALUES, 
                                                 ece3_experiment_id=ece3_experiment_id) 
                                  for var in var_list], compat='no_conflicts')

        if 'siconc' in ocean_ds_dict[ocean_grid_type].data_vars:
            ocean_ds_dict[ocean_grid_type]['siconc'] = ocean_ds_dict[ocean_grid_type]['siconc']/100.0
        
        regridder = xe.Regridder(ocean_ds_dict[ocean_grid_type][var_list[0]].isel(time=0), 
                                 ace2grid, 
                                 'bilinear',
                                 ignore_degenerate=True, 
                                 reuse_weights=False, 
                                 periodic=True, 
                                 filename=f'weights_ece3_oce_{ocean_grid_type}.nc')
        ocean_ds_dict[ocean_grid_type] = regridder(ocean_ds_dict[ocean_grid_type])

# %%
ocean_ds = xr.merge(list(ocean_ds_dict.values()))

# %%
if debug:
    ocean_ds = ocean_ds.isel(time=slice(0, 12*5))

# %%
ece3_var_lookup = {k: v for k, v in ece3_var_lookup.items() if k in list(chain.from_iterable(list(ocean_vars.values())))}
all_renamed_vars = list(ece3_var_lookup.values())
ocean_ds = convert_dts_to_first_of_month(ocean_ds)

ocean_ds = ocean_ds.rename(ece3_var_lookup)

if 'sea_surface_temperature' in all_renamed_vars:
    ocean_ds['sea_surface_temperature'] = ocean_ds['sea_surface_temperature'] + 273

# %%
ocean_ds = ocean_ds.sel(time=atmosphere_ds['time'].values)

# %%
atmosphere_ds = atmosphere_ds.regrid.linear(ace2grid)
experiment_ds = xr.merge([atmosphere_ds, ocean_ds], join='exact')

# %%
# Load ice data, in order to get ice mask
ice_mask = experiment_ds['sea_ice_fraction'].mean('time') > 0.1

# %%
# Have to slightly regrid the sea mask due to very small difference in lat/lon
# sea_mask = sea_mask.astype(np.int8).regrid.linear(experiment_ds) >0

# %%
for var in ['mean_surface_sensible_heat_flux', 
                'mean_surface_latent_heat_flux', 
                'mean_surface_downward_short_wave_radiation_flux',
                'mean_surface_upward_short_wave_radiation_flux', 
                'mean_surface_downward_long_wave_radiation_flux',
                'mean_surface_upward_long_wave_radiation_flux'
               ]:
    experiment_ds[var] = xr.where(sea_mask['sst'], experiment_ds[var], np.nan)
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

# %% [markdown]
# # Investigate ENSO correlations

# %%
en34_da = calculate_nino_index(experiment_ds['sea_surface_temperature'], nino_region=3.4)
en34_da_smoothed = calculate_nino_index(experiment_ds['sea_surface_temperature'], rolling_window=5, nino_region=3.4)
en34_da_seasonal = calculate_nino_index(experiment_ds['sea_surface_temperature'], remove_seasonal_cycle=False, nino_region=3.4)

en3_da = calculate_nino_index(experiment_ds['sea_surface_temperature'], nino_region=3)

m=0
if not debug:
    print(f'Saving Nino timeseries data to {OUTPUT_DIR}')
    en34_da.to_netcdf(os.path.join(OUTPUT_DIR, f'nino3_4.nc'))
    en34_da_smoothed.to_netcdf(os.path.join(OUTPUT_DIR, f'nino3_4_smoothed.nc'))
    en34_da_seasonal.to_netcdf(os.path.join(OUTPUT_DIR, f'nino3_4_seasonal.nc'))

for var in ['total_precipitation_daily']:
    y = experiment_ds[var]
    
    nino_stats_ds = calculate_linear_relationship(en34_da,y)
    nino_stats_smoothed_ds = calculate_linear_relationship(en34_da_smoothed,y)
    nino_stats_seasonal_ds = calculate_linear_relationship(en34_da_seasonal,y)

    nino_3_stats_ds = calculate_linear_relationship(en3_da,y)
    
    if not debug:
        print(f'Saving Nino stats data to {OUTPUT_DIR}')
        nino_stats_ds.to_netcdf(os.path.join(OUTPUT_DIR, f'nino3_4_stats_{var}_m{m}.nc'))
        nino_stats_smoothed_ds.to_netcdf(os.path.join(OUTPUT_DIR, f'nino3_4_stats_{var}_smoothed_m{m}.nc'))
        nino_stats_seasonal_ds.to_netcdf(os.path.join(OUTPUT_DIR, f'nino3_4_stats_{var}_seasonal_m{m}.nc'))

        nino_stats_ds.to_netcdf(os.path.join(OUTPUT_DIR, f'nino3_stats_{var}_m{m}.nc'))

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
