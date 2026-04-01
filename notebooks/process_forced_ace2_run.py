# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3.12.9-01
#     language: python
#     name: python-3.12.9-01
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
from notebooks.coupling_processing_utils import calculate_linear_relationship, calculate_anomalies, ace2_var_lookup, is_notebook

# %%
BASE_OUTPUT_DIR = '/home/ecme4254/perm/repos/ace2_nemo_coupler/notebooks/processed_data'
experiment_id = 'ace2_forced'

if is_notebook():
    debug=True
else:
    debug=False
debug=False
OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, experiment_id)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# %%
input_folder_dict = {
    'fixedCO2': "/home/ecme4254/scratch/ace2_forcing_data/fixedCO2_1951-2051",
    'historical': "/home/ecme4254/scratch/ace2_forcing_data/historical_1951-2021/"
}

experiment_path_dict = {
    'fixedCO2': "/home/ecme4254/hpcperm/model_runs/ace2/ace2_forced_fixedCO2_70years/",
    'historical': "/home/ecme4254/hpcperm/model_runs/ace2/ace2_forced_hist_70years/"
}

# %%
experiment_ds_dict = {}
for k in input_folder_dict:
    
    experiment_ds_dict[k] = xr.open_dataset(os.path.join(experiment_path_dict[k], "monthly_mean_predictions.nc"))

    if debug:
        experiment_ds_dict[k] = experiment_ds_dict[k].isel(time=slice(0, 12*5))
        
time_vals = pd.date_range(start="1951-01-01", end="2021-12-31", freq="MS")[: len(experiment_ds_dict['fixedCO2']['time'])]
years = sorted(set(time_vals.year))

for k in input_folder_dict:
    experiment_ds_dict[k] = experiment_ds_dict[k].assign_coords(time=time_vals)
    experiment_ds_dict[k] = experiment_ds_dict[k].rename({varname: v for varname, v in ace2_var_lookup.items() if varname in experiment_ds_dict[k].variables}).rename({'lat': 'latitude', 'lon': 'longitude'}).isel(sample=0).drop_vars(['init_time', 'valid_time', 'counts'])


# %%

sst_da_dict = {}
for k in experiment_path_dict:
    print(f"Processing {k} run...")
    
    sst_da_dict[k] = []
    fps = [os.path.join(input_folder_dict[k], f"forcing_{year}.nc") for year in years]
    
    for fp in tqdm(fps, desc="Processing ACE2 forced runs", total=len(fps)):
        y = Path(fp).stem.split("_")[1]
        
        ds = xr.load_dataset(fp).resample(time='MS').mean()
        
        output_dir = os.path.join(experiment_path_dict[k], 'forcing_data')
        os.makedirs(output_dir, exist_ok=True)
        
        # Save surface temperature data for later use in plotting and analysis
        sst_da_dict[k].append(ds['surface_temperature'].assign_coords(latitude=experiment_ds_dict[k].latitude, longitude=experiment_ds_dict[k].longitude))

    sst_da_dict[k] = xr.concat(sst_da_dict[k], dim='time')
    experiment_ds_dict[k]['sea_surface_temperature'] = sst_da_dict[k]


# %%
def bjerknes_feedback_analysis(ds):
    
    enso_vars_ds = ds[['sea_surface_temperature', 
                       '10m_u_component_of_wind']].sel(longitude=slice(130, 250), latitude=slice(-15,15)).copy()
    
    anomaly_ds = calculate_anomalies(enso_vars_ds).transpose('time', 'latitude', 'longitude')

    for var in ['sea_surface_temperature']:
    
        anomaly_ds[f'{var}_gradient'] = anomaly_ds[var].sel(longitude=slice(220, 250), 
                                                            latitude=slice(-5,5)).mean(['longitude', 'latitude']) - anomaly_ds[var].sel(longitude=slice(130, 160), latitude=slice(-5,5)).mean(['longitude', 'latitude'])
        anomaly_ds[f'{var}_gradient'] = anomaly_ds[f'{var}_gradient'] / ( ( 235 - 145) * 111.32 * 1000) # Result is in K/m

    anomaly_ds['10m_u_component_of_wind_area_avg'] = anomaly_ds['10m_u_component_of_wind'].sel(latitude=slice(-5,5)).mean(['longitude', 'latitude'])
    
    ###########
    results_dict = {}
    for comparison_vars in [
                            ['sea_surface_temperature_gradient', '10m_u_component_of_wind'],
                           ]:
    
        cvar1 = comparison_vars[0]
        cvar2 = comparison_vars[1]
    
        results_dict[f'{cvar1}__{cvar2}'] = calculate_linear_relationship(anomaly_ds[cvar1], anomaly_ds[cvar2])
        results_dict[f'{cvar2}__{cvar1}'] = calculate_linear_relationship(anomaly_ds[cvar2], anomaly_ds[cvar1])
        
    return results_dict, anomaly_ds


# %%
for k, ds in experiment_ds_dict.items():
    ds.attrs['experiment_id'] = experiment_id
    results_dict, anomaly_ds = bjerknes_feedback_analysis(ds.copy())
            
    if not debug:
        print(f"Saving zonal gradient and area average variables for {k} run...")   
        anomaly_ds[[v for v in anomaly_ds  if (v.endswith('gradient') or v.endswith('area_avg'))]].to_netcdf(os.path.join(OUTPUT_DIR, f'zonal_pacific_gradients.nc'))

    if not debug:
        print(f"Saving Bjerknes feedback results for {k} run...")
        with open(os.path.join(OUTPUT_DIR, f'bjerknes_correlations_{k}.pkl'), 'wb+') as ofh:
            pickle.dump(results_dict, ofh)
