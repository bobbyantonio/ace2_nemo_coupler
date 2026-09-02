# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: graphcast
#     language: python
#     name: python3
# ---

# %%
import os, sys
import datetime
from glob import glob
import pickle
import xarray as xr
import xarray_regrid
from pathlib import Path
from argparse import ArgumentParser

base_dir = Path(os.getcwd())

if not str(base_dir).endswith('ace2_nemo_coupler'):
    base_dir = base_dir.parent
    
sys.path.append(str(base_dir))
from notebooks.coupling_processing_utils import calculate_linear_relationship, calculate_anomalies, ace2_var_lookup, is_notebook, bjerknes_feedback_analysis, calculate_lagged_correlations

# %%
experiment_id='samudrace'
if is_notebook():
    years=range(1951,1955)
    raw_data_dir = "/network/group/aopp/predict/HMC005_ANTONIO_EERIE/predictions/samudrace_70yr"
    debug=True
    base_output_dir =os.path.join(base_dir, 'notebooks', 'processed_data')
    ace2_data_dir = '/network/group/aopp/predict/HMC005_ANTONIO_EERIE/ace2_data'
else:
    parser = ArgumentParser()
    parser.add_argument('--raw-data-dir', type=str, required=True)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--years', type=str, default='1951-2021')
    parser.add_argument('--ace2-data-dir', type=str, required=True)
    parser.add_argument('--base-output-dir', type=str, required=True)
    
    args = parser.parse_args()

    debug = args.debug
    raw_data_dir = args.raw_data_dir
    experiment_id = args.experiment_id
    ace2_data_dir = args.ace2_data_dir
    years_split = args.years.split('-')
    base_output_dir = args.base_output_dir
    years = range(int(years_split[0]), int(years_split[1])+1)

    if debug:
        years = years[:10]


OUTPUT_DIR = os.path.join(base_output_dir, experiment_id)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# %%
# ACE2 grid / sea mask
ace2grid = xr.load_dataset(os.path.join(ace2_data_dir, "grid.nc"))
sea_mask = xr.load_dataarray(os.path.join(ace2_data_dir, "era5_sea_mask_ACE2.nc"))
ace2_grid_area = xr.load_dataset(os.path.join(ace2_data_dir, "gridarea.nc"))['cell_area']

# %%
atmosphere_ds = xr.load_dataset(os.path.join(raw_data_dir, "atmosphere", "monthly_mean_predictions.nc"))[['PRATEsfc', 'LHTFLsfc', 'SHTFLsfc', 'UGRD10m','TMP2m','PRESsfc', 'DSWRFsfc']]


# %%
atmosphere_ds = atmosphere_ds.rename({'PRATEsfc': 'total_precipitation', 
                                'LHTFLsfc': 'mean_surface_latent_heat_flux', 
                                'SHTFLsfc': 'mean_surface_sensible_heat_flux',
                                'DSWRFsfc': 'mean_surface_downward_short_wave_radiation_flux',
                                'UGRD10m': '10m_u_component_of_wind',
                                'TMP2m': '2m_temperature',
                                'PRESsfc': 'surface_pressure',
                                'lat': 'latitude', 
                                'lon': 'longitude'})

# %%
atmosphere_ds['total_precipitation_daily'] = atmosphere_ds['total_precipitation']*86400
atmosphere_ds = atmosphere_ds.isel(sample=0).drop_vars(['counts', 'time'])

# ACE2 sign convention for these fluxes is opposite to ECMWF convention of positive downward 
atmosphere_ds['mean_surface_latent_heat_flux'] = -1 * atmosphere_ds['mean_surface_latent_heat_flux']
atmosphere_ds['mean_surface_sensible_heat_flux'] = -1 * atmosphere_ds['mean_surface_sensible_heat_flux']

atmosphere_ds['mean_surface_heat_flux'] = atmosphere_ds['mean_surface_latent_heat_flux'] + atmosphere_ds['mean_surface_sensible_heat_flux']


# %%
ocean_ds = xr.load_dataset(os.path.join(raw_data_dir, "ocean","monthly_mean_predictions.nc"))[['sst']]
ocean_ds = ocean_ds.isel(sample=0).drop_vars(['counts'])

# %%
atmosphere_ds = atmosphere_ds.assign_coords(time=ocean_ds['valid_time'])


# %%
ocean_ds = ocean_ds.assign_coords(time=ocean_ds['valid_time'])
ocean_ds = ocean_ds.rename({'sst': 'sea_surface_temperature',
                            'lat': 'latitude', 
                            'lon': 'longitude'})

# %%
experiment_ds = xr.merge([atmosphere_ds, ocean_ds], compat='no_conflicts')

# %%
adjusted_time = [datetime.datetime(v.year - 311 + 1951, v.month, 15) for v in experiment_ds['time'].values]

# %%
experiment_ds = experiment_ds.assign_coords(time=adjusted_time)

# %%
experiment_ds = experiment_ds.regrid.linear(sea_mask)

# %%
experiment_ds

# %% [markdown]
# ## Bjerknes feedback

# %%
results_dict, anomaly_ds = bjerknes_feedback_analysis(experiment_ds)

# %%
anomaly_ds[[v for v in anomaly_ds  if (v.endswith('gradient') or v.endswith('area_avg'))]].to_netcdf(os.path.join(OUTPUT_DIR, f'zonal_pacific_gradients.nc'))

with open(os.path.join(OUTPUT_DIR, f'bjerknes_correlations.pkl'), 'wb+') as ofh:
        pickle.dump(results_dict, ofh)

# %% [markdown]
# ## Lagged correlations

# %%
month_lag_max=0
for lag_vars in [['mean_surface_heat_flux', 'sea_surface_temperature'],
                    ['mean_surface_downward_short_wave_radiation_flux','sea_surface_temperature']
                ]:
    
    print('Calculating lagged correlations for ', lag_vars, flush=True)
    lag_var1 = lag_vars[0]
    lag_var2 = lag_vars[1]
    
    ace2_nemo_results_dict = calculate_lagged_correlations(
                                                            experiment_ds, 
                                                            lag_var1, 
                                                            lag_var2, 
                                                            month_lag_max=month_lag_max)
        
    
    with open(os.path.join(OUTPUT_DIR, f'lagged_correlations_max{month_lag_max}_{lag_var1}_{lag_var2}.pkl'), 'wb+') as ofh:
        pickle.dump(ace2_nemo_results_dict, ofh)
