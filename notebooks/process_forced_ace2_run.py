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
import os, sys
import datetime
import pickle
import pandas as pd
import xarray as xr
from pathlib import Path

sys.path.append("/home/ecme4254/perm/repos/ace2_nemo_coupler")
from notebooks.coupling_processing_utils import calculate_linear_relationship, calculate_anomalies, ace2_var_lookup, is_notebook

# %%
BASE_OUTPUT_DIR = '/perm/ecme4254/repos/nwp_notebooks/eerie/coupled_experiments/processed_data'
experiment_id = 'ace2_forced'
debug=False
OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, experiment_id)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# %%
control_ds = xr.open_dataset("/home/ecme4254/hpcperm/model_runs/ace2/ace2_forced_control_70years/monthly_mean_predictions.nc")
hist_ds = xr.open_dataset("/home/ecme4254/hpcperm/model_runs/ace2/ace2_forced_hist_70years/monthly_mean_predictions.nc")
time_vals = pd.date_range(start="1951-01-01", end="2021-12-31", freq="MS")[: len(control_ds['time'])]

control_ds = control_ds.assign_coords(time=time_vals)
hist_ds = hist_ds.assign_coords(time=time_vals)

# %%
control_ds = control_ds.rename({k: v for k, v in ace2_var_lookup.items() if k in control_ds.variables}).rename({'lat': 'latitude', 'lon': 'longitude'})
hist_ds = hist_ds.rename({k: v for k, v in ace2_var_lookup.items() if k in hist_ds.variables}).rename({'lat': 'latitude', 'lon': 'longitude'})

control_sst_da = xr.open_mfdataset("/home/ecme4254/scratch/ace2_forcing_data/control_1951-2051/forcing_*.nc", combine="by_coords", preprocess=lambda x:x['surface_temperature'])['surface_temperature']
hist_sst_da = xr.open_mfdataset("/home/ecme4254/scratch/ace2_forcing_data/historical_1951-2021/forcing_*.nc", combine="by_coords", preprocess=lambda x:x['surface_temperature'])['surface_temperature']

control_sst_da = control_sst_da.isel(time=range(len(time_vals))).assign_coords(time=time_vals)
hist_sst_da = hist_sst_da.isel(time=range(len(time_vals))).assign_coords(time=time_vals)

control_ds['sea_surface_temperature'] = control_sst_da
hist_ds['sea_surface_temperature'] = hist_sst_da

# %%
control_ds = control_ds.isel(sample=0)
hist_ds = hist_ds.isel(sample=0)


# %%
def bjerknes_feedback_analysis(ds):
    
    enso_vars_ds = ds[['sea_surface_temperature', 
                       '10m_u_component_of_wind']].copy()
    
    anomaly_ds = calculate_anomalies(enso_vars_ds).sel(longitude=slice(130, 250), latitude=slice(-15,15)).transpose('time', 'latitude', 'longitude')

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
ds_dict = {'control': control_ds, 'hist': hist_ds}
for k, ds in ds_dict.items():
    ds.attrs['experiment_id'] = experiment_id
    results_dict, anomaly_ds = bjerknes_feedback_analysis(ds.copy())
            
    if not debug:
        anomaly_ds[[v for v in anomaly_ds  if (v.endswith('gradient') or v.endswith('area_avg'))]].to_netcdf(os.path.join(OUTPUT_DIR, f'zonal_pacific_gradients.nc'))

    if not debug:
        with open(os.path.join(OUTPUT_DIR, f'bjerknes_correlations_{k}.pkl'), 'wb+') as ofh:
            pickle.dump(results_dict, ofh)

# %%
