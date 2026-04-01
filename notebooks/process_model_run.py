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
import pickle
import datetime
import numpy as np
import xarray as xr
from argparse import ArgumentParser

# %%
import pandas as pd
import calendar
from itertools import chain

# %%
# python_path = sys.executable
# esmkfile_path = python_path.replace('bin/python', 'lib/esmf.mk')
# os.environ['ESMFMKFILE'] = esmkfile_path
import xarray_regrid
import xesmf as xe
import xarray_regrid

sys.path.append("/home/ecme4254/perm/repos/ace2_nemo_coupler")
from notebooks.coupling_processing_utils import detrend_dataarray, \
    convert_dts_to_first_of_month, calculate_en34 ,calculate_linear_relationship, \
    mean_areas, calculate_en34_spectra, vertical_integral, load_ds_subset, load_nemo_ds_subset,\
    calculate_correlation, OLEVEL_VALUES, OLEVEL_BIN_EDGES, calculate_lagged_correlations, calculate_anomalies, \
    bjerknes_feedback_analysis, calculate_nino_index, ace2_var_lookup, is_notebook

BASE_OUTPUT_DIR = '/home/ecme4254/perm/repos/ace2_nemo_coupler/notebooks/processed_data'

# %%
if is_notebook():
    experiment_id = 'n3.6_ace2_1951_spinupCMIP6_19510101-20210101'
    ensemble_members = [0]
    glob_str = '199*'
    # model_run_dir='/home/ecme4254/perm/old_model_runs'
    model_run_dir ='/home/ecme4254/hpcperm/model_runs'

    components = ''
    month_lag_max = 1
    debug=True
else:
    parser = ArgumentParser()
    parser.add_argument('--experiment-id', type=str, required=True,
                        help="Experiment ID")
    parser.add_argument('--ensemble-members', nargs='+', default=[0], type=int)
    parser.add_argument('--model-run-dir', type=str, default='/home/ecme4254/hpcperm/model_runs')
    parser.add_argument('--month-lag-max', type=int, default=None)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--components', default='all')
    
    args = parser.parse_args()

    print(args, flush=True)

    experiment_id = args.experiment_id
    ensemble_members = args.ensemble_members
    model_run_dir = args.model_run_dir
    month_lag_max = args.month_lag_max
    components = args.components

    if args.debug:
        glob_str = '201*'
        debug=True
    else:
        glob_str = '??????'
        debug=False

# %%
atm2oce_vars = ['mean_surface_sensible_heat_flux', 
           'mean_surface_latent_heat_flux', 
           'sensible_heat_flux_ice', 
           'latent_heat_flux_ice', 
           'solar_flux_over_ice', 
           'total_non_solar_flux_ice', 
           'net_long_wave_radiation_flux_ice', 
           'mean_surface_upward_long_wave_radiation_flux', 
           'mean_surface_downward_long_wave_radiation_flux', 
           'mean_surface_upward_short_wave_radiation_flux', 
           'mean_surface_downward_short_wave_radiation_flux',
           'evaporation',
           'evaporation_ice',
           'solid_precipitation',
           'liquid_precipitation',
           'instantaneous_eastward_turbulent_surface_stress',
           'instantaneous_northward_turbulent_surface_stress']

oce2atm_vars = ['sea_ice_fraction', 'sea_ice_thickness', 'sea_ice_temperature']

atmosphere_vars = list(ace2_var_lookup.keys())

nemo_vars_dict = {'T': ['mldr10_1', 'ssh', 'heatc', 'toce_pot'],
                  'U': ['ssu'],
                  'V': ['ssv']
                 }

rename_dict = {'ssh': 'sea_surface_height', 
                'mldr10_1': 'mixed_layer_depth', 
                'heatc': 'heat_content', 
               'toce_pot':'sea_water_potential_temperature'}


# %%
base_dir = os.path.join(model_run_dir, experiment_id)
model_name = 'ace2'
era5_dir = '/scratch/ecme4254/era5_monthly'
oras5_dir = '/scratch/ecme4254/oras5'

OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, experiment_id)
os.makedirs(OUTPUT_DIR, exist_ok=True)

ace2_grid_area = xr.load_dataset("/home/ecme4254/hpcperm/ml_model_data/ace2/gridarea.nc")['cell_area']
sea_mask = xr.load_dataarray("/hpcperm/ecme4254/ml_model_data/ace2/era5_sea_mask_ACE2.nc")

# %%
print('Loading atm2oce data', flush=True)
atm2oce_ds = xr.concat([load_ds_subset(f'{base_dir}_m{n}', f'atm2oce_MS_{model_name}_nemo_{glob_str}.nc', atm2oce_vars).expand_dims({'member': [n]}) for n in ensemble_members], dim='member')

# %%
print('Loading oce2atm data', flush=True)

oce2atm_ds = xr.concat([load_ds_subset(f'{base_dir}_m{n}', f'oce2atm_MS_{model_name}_nemo_{glob_str}.nc', oce2atm_vars).expand_dims({'member': [n]}) for n in ensemble_members], dim='member')



# %%
print('Loading atmosphere data', flush=True)


atmosphere_monthly_ds = xr.concat([load_ds_subset(f'{base_dir}_m{n}', f'{model_name}_MS_{model_name}_nemo_{glob_str}.nc', atmosphere_vars).expand_dims({'member': [n]}) for n in ensemble_members], dim='member')
atmosphere_monthly_ds = atmosphere_monthly_ds.rename(ace2_var_lookup)

# %%
# Calcaulate integral of specific total water over all the levels
atmosphere_monthly_ds['specific_total_water'] = xr.concat([atmosphere_monthly_ds[f'specific_total_water_{n}'] for n in range(8)], dim='level')
atmosphere_monthly_ds = atmosphere_monthly_ds.drop_vars([f'specific_total_water_{n}' for n in range(8)])
atmosphere_monthly_ds = atmosphere_monthly_ds.transpose('member', 'time', 'latitude', 'longitude', 'level')

atmosphere_monthly_ds['total_water_path'] = vertical_integral(atmosphere_monthly_ds['specific_total_water'], 
                                                              atmosphere_monthly_ds['surface_pressure'])

# %%
years = sorted(set(atmosphere_monthly_ds['time.year'].values))
time_vals = [pd.Timestamp(dt) for dt in sorted(atmosphere_monthly_ds['time'].values)]
month_limits = [ (dt, datetime.datetime(dt.year, dt.month, calendar.monthrange(dt.year, dt.month)[1])) for dt in time_vals]
expanded_time_vals = sorted(chain.from_iterable([ list(pd.date_range(ml[0], ml[1])) for ml in month_limits]))

# %%
# Samudra levels: 2.5m, 10m, 22.5m, 40m, 65m, 105m, 165m, 250m, 375m, 550m, 775m, 1050m, 1400m, 1850m, 2400m, 3100m, 4000m, 5000m, 6000m

# %%
# from glob import glob
# import gc
# fps = glob(os.path.join('/home/ecme4254/hpcperm/model_runs/n3.6_ace2_1951_spinupCMIP6_19510101-20210101_m0/nemo_output_ace2/nemo_ocean_output_grid_T_20*'))
# ds = []
# tmp_base_dir = f'{base_dir}_m0'
# glob_filename=f'nemo_output_{model_name}/nemo_ocean_output_grid_T_{glob_str}-{glob_str}.nc'
# vars_to_select= ['toce_pot', 'sss']
# level_values=OLEVEL_VALUES
# bin_edge_values=OLEVEL_BIN_EDGES
# decode_times=False

# %%
# fp = fps[0]
# date_str = fp.split('/')[-1].split('_')[-1][:6]
# dt = pd.Timestamp(date_str + '01')

# tmp_ds = xr.open_dataset(fp, decode_times=decode_times)[vars_to_select]
# tmp_ds = tmp_ds.assign_coords({'time_counter': [dt]})
# tmp_ds = tmp_ds.rename({'time_counter': 'time'}).drop_vars('time_centered')

# if 'olevel' in tmp_ds.dims:
#     vars_with_level = [v for v in tmp_ds.data_vars if 'olevel' in tmp_ds[v].coords]

#     if bin_edge_values is not None:
#         binned_das = []
#         for v in vars_with_level:
    
        
#             bin_edges = sorted(bin_edge_values)
#             bin_labels = [f'{bin_edges[n]}-{bin_edges[n+1]}' for n in range(len(bin_edges)-1)]
#             binned_da = tmp_ds[v].groupby_bins(group='olevel', bins=bin_edges, right=True, labels=bin_labels).mean()

#             binned_da.name = f'{v}_binned'
#             binned_das.append(binned_da)
#         tmp_ds = xr.merge([tmp_ds] + binned_das)
        
#     if level_values is not None:
            
#         tmp_ds = tmp_ds.sel(olevel=level_values)


# %%
print('Loading NEMO output vars', flush=True)
nemo_vars_dict = {'T': ['mldr10_1', 'ssh', 'heatc', 'toce_pot', 'sss', 'sst'],
                  'U': ['ssu'],
                  'V': ['ssv']
                 }

rename_dict = {'ssh': 'sea_surface_height', 
               'sss': 'sea_surface_salinity',
               'sst': 'sea_surface_temperature',
                'mldr10_1': 'mixed_layer_depth', 
                'heatc': 'heat_content', 
               'toce_pot':'sea_water_potential_temperature'}

nemo_ds_dict = {}

for grid_string, grid_vars in nemo_vars_dict.items():
    print(grid_string)
    # Note: there seems to be a problem with time_counter in 1990-2000, when using join='exact'. This isprobably the reason there
    # are gaps in the heat content time series

    nemo_ds_dict[grid_string] = []
    for n in ensemble_members:
        tmp_ds = load_nemo_ds_subset(f'{base_dir}_m{n}', f'nemo_output_{model_name}/nemo_ocean_output_grid_{grid_string}_{glob_str}-{glob_str}.nc', 
                                                               vars_to_select=grid_vars, 
                                                               level_values=OLEVEL_VALUES,
                                                               bin_edge_values=OLEVEL_BIN_EDGES,
                                                               decode_times=False, 
                                                               concat_dim='time').expand_dims({'member': [n]})
        nemo_ds_dict[grid_string].append(tmp_ds)


    nemo_ds_dict[grid_string] = xr.concat(nemo_ds_dict[grid_string], dim='member', join='outer', coords='minimal')
    
    nemo_ds_dict[grid_string] = nemo_ds_dict[grid_string].rename({k: v for k,v in rename_dict.items() if k in grid_vars})
    nemo_ds_dict[grid_string] = nemo_ds_dict[grid_string].sortby('time')
    
    tmp_regridder = xe.Regridder(nemo_ds_dict[grid_string].isel(time=0, member=0), 
                         atmosphere_monthly_ds.isel(member=0,time=0)['2m_temperature'], 
                         'bilinear',
                         ignore_degenerate=True, 
                         reuse_weights=False, 
                         periodic=True, 
                         filename=f'nemo_{grid_string}_weights.nc')


    nemo_ds_dict[grid_string] = tmp_regridder(nemo_ds_dict[grid_string])


# %%
nemo_ds = xr.merge(list(nemo_ds_dict.values()))

# %%
print('Joining datasets together', flush=True)
experiment_ds = xr.merge([atm2oce_ds, oce2atm_ds, atmosphere_monthly_ds, nemo_ds], compat='no_conflicts')

# %%
# Calculate sea ice volume, and sea ice extent in m**2

experiment_ds['sea_ice_volume'] = experiment_ds['sea_ice_thickness'] * experiment_ds['sea_ice_fraction'] * ace2_grid_area
experiment_ds['sea_ice_volume'].attrs['long_name'] = 'Sea ice volume'
experiment_ds['sea_ice_volume'].attrs['standard_name'] = 'sea_ice_volume'

experiment_ds['sea_ice_extent'] = (experiment_ds['sea_ice_fraction'] > 0.15).astype(np.int8)* ace2_grid_area
experiment_ds['sea_ice_extent'] = experiment_ds['sea_ice_extent']/1e6 # Convert to km^2
experiment_ds['sea_ice_extent'].attrs['long_name'] = 'Sea ice extent'
experiment_ds['sea_ice_extent'].attrs['standard_name'] = 'sea_ice_extent'

# %%
# Convert precip to mm/day
experiment_ds['total_precipitation_daily'] = experiment_ds['total_precipitation']*86400

# %%
experiment_ds['mean_surface_net_short_wave_radiation_flux'] = experiment_ds['mean_surface_downward_short_wave_radiation_flux'] - experiment_ds['mean_surface_upward_short_wave_radiation_flux']
experiment_ds['mean_surface_net_long_wave_radiation_flux'] = experiment_ds['mean_surface_downward_long_wave_radiation_flux'] - experiment_ds['mean_surface_upward_long_wave_radiation_flux']

# %%
experiment_ds['mean_surface_sensible_heat_flux_raw'] = experiment_ds['mean_surface_sensible_heat_flux'].copy()
experiment_ds['mean_surface_latent_heat_flux_raw'] = experiment_ds['mean_surface_latent_heat_flux'].copy()
experiment_ds['mean_surface_net_short_wave_radiation_flux_raw'] = experiment_ds['mean_surface_net_short_wave_radiation_flux'].copy()
experiment_ds['mean_surface_net_long_wave_radiation_flux_raw'] = experiment_ds['mean_surface_net_long_wave_radiation_flux'].copy()



# %%
# Create ice mask, and account for missing dates

ice_mask = oce2atm_ds['sea_ice_fraction'].mean('time') > 0.15
ice_mask_t = xr.merge([oce2atm_ds['sea_ice_fraction'], experiment_ds['sensible_heat_flux_ice']], join='outer')['sea_ice_fraction']
missing_ice_mask_t_values = list(set(experiment_ds['time'].values).difference(set(oce2atm_ds['time'].values)))

for t_val in missing_ice_mask_t_values:
    missing_dt = pd.Timestamp(t_val)
    if missing_dt == experiment_ds['time'].values.max():
        ice_mask_t.loc[{'time': missing_dt}] = ice_mask_t.sel(time=missing_dt - pd.DateOffset(months=1))
    else:
        ice_mask_t.loc[{'time': missing_dt}] = ice_mask_t.sel(time=missing_dt + pd.DateOffset(months=1))

# %%
try:
    experiment_ds['mean_surface_sensible_heat_flux'] = xr.where(ice_mask_t, experiment_ds['sensible_heat_flux_ice'], experiment_ds['mean_surface_sensible_heat_flux'])
    experiment_ds['mean_surface_latent_heat_flux'] = xr.where(ice_mask_t, experiment_ds['latent_heat_flux_ice'], experiment_ds['mean_surface_latent_heat_flux'])
    experiment_ds['mean_surface_net_short_wave_radiation_flux'] = xr.where(ice_mask_t,  experiment_ds['solar_flux_over_ice'], experiment_ds['mean_surface_net_short_wave_radiation_flux'])
    experiment_ds['mean_surface_net_long_wave_radiation_flux'] = xr.where(ice_mask_t,  experiment_ds['net_long_wave_radiation_flux_ice'],  experiment_ds['mean_surface_net_long_wave_radiation_flux'])

    # Fluxes over ocean points only
    for var in ['mean_surface_sensible_heat_flux', 
                'mean_surface_latent_heat_flux', 
                'mean_surface_net_short_wave_radiation_flux', 
                'mean_surface_downward_short_wave_radiation_flux',
                'mean_surface_upward_short_wave_radiation_flux', 
                'mean_surface_net_long_wave_radiation_flux',
                'mean_surface_downward_long_wave_radiation_flux',
                'mean_surface_upward_long_wave_radiation_flux'
               ]:
        experiment_ds[var] = xr.where(sea_mask, experiment_ds[var], np.nan)
        experiment_ds[f'{var}_oce'] = xr.where(ice_mask, np.nan, experiment_ds[var])
    
        
        experiment_ds[f'{var}_ice'] = xr.where(ice_mask, experiment_ds[var], np.nan)
        
    for suffix in ['', '_oce', '_ice']:
        experiment_ds[f'mean_surface_heat_flux{suffix}'] = experiment_ds[f'mean_surface_sensible_heat_flux{suffix}'] + experiment_ds[f'mean_surface_latent_heat_flux{suffix}']
        experiment_ds[f'albedo{suffix}'] = experiment_ds[f'mean_surface_upward_short_wave_radiation_flux{suffix}'] / experiment_ds[f'mean_surface_downward_short_wave_radiation_flux{suffix}']
    
        experiment_ds[f'albedo{suffix}'] = xr.where(experiment_ds[f'mean_surface_downward_short_wave_radiation_flux{suffix}'] == 0.0, 1.0, experiment_ds[f'albedo{suffix}'])
    
        experiment_ds[f'non_solar_heat_flux{suffix}'] = experiment_ds[f'mean_surface_sensible_heat_flux{suffix}']  + experiment_ds[f'mean_surface_latent_heat_flux{suffix}'] + experiment_ds[f'mean_surface_net_long_wave_radiation_flux{suffix}']
        experiment_ds[f'total_heat_flux{suffix}'] = experiment_ds[f'non_solar_heat_flux{suffix}'] + experiment_ds[f'mean_surface_net_short_wave_radiation_flux{suffix}']
except Exception as e:
    print('********* Problem with flux processing')
    print(e)


# %%
# Add 2mt over sea points only
experiment_ds['2m_temperature_sea_points'] = xr.where(sea_mask, experiment_ds['2m_temperature'], np.nan)
experiment_ds['sea_ice_temperature_gradient'] = experiment_ds['2m_temperature'] - experiment_ds['sea_ice_temperature']

experiment_ds['surface_temperature_difference'] = experiment_ds['2m_temperature'] - xr.where(ice_mask_t, experiment_ds['sea_ice_temperature'], experiment_ds['sea_surface_temperature'])

# %%
# Weights for calculating global averages
weights = np.cos(np.deg2rad(atmosphere_monthly_ds.latitude))
weights = weights / weights.sum().item()

# %% [markdown]
# ## Bjerknes feedback

# %%
# Li et al: The zonal gradients in this study are defined as the differences between the area‐averaged anomalies over 
# the regions in the eastern tropical Pacific (140°–110°W, 5°S–5°N) and western tropical Pacific (130°-160°E, 5°S–5°N).

results_dict, anomaly_ds = bjerknes_feedback_analysis(experiment_ds.sel(member=0).copy())
        
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
                   'JJA': [dt for dt in time_vals if dt.month in [6,7,8]],
                   'DJF': [dt for dt in time_vals if dt.month in [12,1,2]],
                   '1st month': time_vals[:1],
                   '1st year': time_vals[:12],
                   '5th year': time_vals[48:60],
                   '1st decade': time_vals[:120],
                   'last 10 years': time_vals[-120:],
                   'last 20 years': time_vals[-240:],
                   '1980-1990': [dt for dt in time_vals if (1980 <= dt.year <=1990) ],
                   '1970-1990': [dt for dt in time_vals if (1970 <= dt.year <=1990) ],
                   '1980-2000': [dt for dt in time_vals if (1980 <= dt.year <=2000) ],
                   '1990-2000': [dt for dt in time_vals if (1990 <= dt.year <=2000) ],
                   '2000-2020': [dt for dt in time_vals if (2000 <= dt.year <=2020) ],
                   '2010-2020': [dt for dt in time_vals if (2010 <= dt.year <=2020) ],
                   'All': time_vals}

time_mean_state_dict = {}

for name, tvals in time_range_dict.items():

    time_mean_state_dict[name] = experiment_ds.sel(member=0).sel(time=tvals).mean('time')

if not debug:
    with open(os.path.join(OUTPUT_DIR, f'time_mean_state_dict.pkl'), 'wb+') as ofh:
        pickle.dump(time_mean_state_dict, ofh)

# %% [markdown]
# ## Lagged correlations

# %%
if month_lag_max is not None:

    for lag_vars in [['mean_surface_heat_flux', 'sea_surface_temperature'],
                     ['10m_u_component_of_wind', '10m_u_component_of_wind'],
                     ['total_heat_flux', 'sea_surface_temperature'],
                     ['mean_surface_latent_heat_flux', 'sea_surface_temperature'],
                     ['mean_surface_downward_short_wave_radiation_flux','sea_surface_temperature']
                    ]:
        
        print('Calculating lagged correlations', flush=True)
        lag_var1 = lag_vars[0]
        lag_var2 = lag_vars[1]
        
        ace2_nemo_results_dict = calculate_lagged_correlations(
                                                              experiment_ds.sel(member=0), 
                                                              lag_var1, 
                                                              lag_var2, 
                                                              month_lag_max=month_lag_max)
            
        
        # if not debug:
        with open(os.path.join(OUTPUT_DIR, f'lagged_correlations_max{month_lag_max}_{lag_var1}_{lag_var2}.pkl'), 'wb+') as ofh:
            pickle.dump(ace2_nemo_results_dict, ofh)

# %% [markdown]
# ## Spatial aggregations

# %%
print('Calculating spatial means', flush=True)

mean_dict = {}


for area_name, latlon_dict in mean_areas.items():
    
    experiment_mean_ds = experiment_ds.sel(latitude=slice(latlon_dict['min_lat'],latlon_dict['max_lat']), longitude=slice(latlon_dict.get('min_lon', 0), latlon_dict.get('max_lon', 360))).weighted(weights.sel(latitude=slice(latlon_dict['min_lat'],latlon_dict['max_lat']))).mean(['latitude', 'longitude']).sortby('time')
  
    # Unweighted sum, for variables that are already expressed in weighted units (e.g. ice area)
    experiment_unweighted_sum_ds = experiment_ds.sel(latitude=slice(latlon_dict['min_lat'],latlon_dict['max_lat']), longitude=slice(latlon_dict.get('min_lon', 0), latlon_dict.get('max_lon', 360))).sum(['latitude', 'longitude']).sortby('time')

    mean_dict[area_name] = {'mean': experiment_mean_ds,
                            'UnweightedSum': experiment_unweighted_sum_ds
                           }

if not debug:
    with open(os.path.join(OUTPUT_DIR, f'mean_dict.pkl'), 'wb+') as ofh:
        pickle.dump(mean_dict, ofh)

# %%

for m in ensemble_members:
    ace2_nemo_time_series = mean_dict['Global']['mean']['sea_surface_temperature'].sel(member=m).groupby('time.year').mean()

    percentile_25th = ace2_nemo_time_series.quantile(0.25).item()
    percentile_75th = ace2_nemo_time_series.quantile(0.75).item()
    
    upper_quartile_data = xr.where(ace2_nemo_time_series > percentile_75th, ace2_nemo_time_series, np.nan).dropna('year')
    lower_quartile_data = xr.where(ace2_nemo_time_series < percentile_25th, ace2_nemo_time_series, np.nan).dropna('year')
    
    upper_quartile_mean = experiment_ds['sea_surface_temperature'].sel(member=m).sel(time=[dt for dt in time_vals if dt.year in upper_quartile_data['year'].values]).mean('time')
    lower_quartile_mean = experiment_ds['sea_surface_temperature'].sel(member=m).sel(time=[dt for dt in time_vals if dt.year in lower_quartile_data['year'].values]).mean('time')
    
    if not debug:
        upper_quartile_mean.to_netcdf(os.path.join(OUTPUT_DIR, f'upper_quartile_mean_sst_m{m}.nc'))
        lower_quartile_mean.to_netcdf(os.path.join(OUTPUT_DIR, f'lower_quartile_mean_sst_m{m}.nc'))



# %% [markdown]
# ## Map of trends

# %%
print('Calculating map of trend; toce', flush=True)

ocean_drift_var= 'sea_water_potential_temperature'

_, polyfit = detrend_dataarray(experiment_ds[ocean_drift_var].isel(member=0).groupby('time.year').mean().sel(latitude=0, method='nearest').sel(longitude=slice(130,260)), 'year')

if not debug:
    print(f'Saving Ocean drift data to {OUTPUT_DIR}')
    polyfit.to_netcdf(os.path.join(OUTPUT_DIR, 'polyfit_toce_pacific.nc' ))


_, polyfit = detrend_dataarray(experiment_ds['sea_water_potential_temperature'].isel(member=0).groupby(['time.year']).mean().mean('longitude'), 'year')

if not debug:
    print(f'Saving Ocean drift data to {OUTPUT_DIR}')
    polyfit.to_netcdf(os.path.join(OUTPUT_DIR, 'polyfit_toce_latitude.nc' ))


# %%
# Trends for different periods

drift_vars = ['sea_surface_temperature', 'sea_ice_thickness', 'sea_ice_fraction', 'sea_surface_height',
             'mean_surface_sensible_heat_flux', 'mean_surface_latent_heat_flux', 'mean_surface_net_long_wave_radiation_flux', 
              'mean_surface_net_short_wave_radiation_flux', 
              'mean_surface_upward_short_wave_radiation_flux',
              'mean_surface_downward_short_wave_radiation_flux',
              'heat_content',
              'total_water_path', 'total_precipitation_daily', '2m_specific_humidity']

trends_time_range_dict = {'Pre-1980': [dt for dt in time_vals if dt.year <=1980],
                           'Post-1980': [dt for dt in time_vals if dt.year> 1980],
                           'Last 50 Years': time_vals[-600:],
                           'First 50 Years': time_vals[:600],
                           'Last 20 Years': time_vals[-240:],
                           'Last 10 Years': time_vals[-120:],
                           '1980-1990': [dt for dt in time_vals if (1980 <= dt.year <=1990) ],
                           '1970-1990': [dt for dt in time_vals if (1970 <= dt.year <=1990) ],
                           '1980-2000': [dt for dt in time_vals if (1980 <= dt.year <=2000) ],
                           '1990-2000': [dt for dt in time_vals if (1990 <= dt.year <=2000) ],
                           '2000-2020': [dt for dt in time_vals if (2000 <= dt.year <=2020) ],
                           '2010-2020': [dt for dt in time_vals if (2010 <= dt.year <=2020) ],
                           'All': time_vals}

trends_dict = {}

for name, tvals in time_range_dict.items():
    trends_dict[name] = {}
    if len(tvals)>0:
        for n, varname in enumerate(drift_vars):
            _, polyfit = detrend_dataarray(experiment_ds[varname].isel(member=0).sel(time=tvals).groupby('time.year').mean(), 'year')
            trends_dict[name][varname] = polyfit

if not debug:
    with open(os.path.join(OUTPUT_DIR, f'trends_dict.pkl'), 'wb+') as ofh:
        pickle.dump(trends_dict, ofh)

# %%
ocean_drift_var= 'sea_water_potential_temperature'

if ocean_drift_var in experiment_ds.data_vars:
    toce_trends_dict = {}
    toce_latitude_trends_dict = {}

    for name, tvals in time_range_dict.items():
        toce_trends_dict[name] = {}
        toce_latitude_trends_dict[name] = {}
        
        if len(tvals)>0:
            _, polyfit_ece3 = detrend_dataarray(experiment_ds[ocean_drift_var].isel(member=0).sel(time=tvals).groupby('time.year').mean().sel(latitude=0, method='nearest').sel(longitude=slice(130,260)), 'year')
            toce_trends_dict[name] = polyfit_ece3
            # Aggregation by latitude
            _, polyfit = detrend_dataarray(experiment_ds[ocean_drift_var].isel(member=0).sel(time=tvals).groupby(['time.year']).mean().mean('longitude'), 'year')
            toce_latitude_trends_dict[name] = polyfit
# if not debug:
#     print(f'Saving Ocean drift data to {OUTPUT_DIR}')
#     polyfit_ece3.to_netcdf(os.path.join(OUTPUT_DIR, 'polyfit_toce_ece3.nc'))

# if not debug:
#     print(f'Saving Ocean drift data to {OUTPUT_DIR}')
#     polyfit.to_netcdf(os.path.join(OUTPUT_DIR, 'polyfit_toce_latitude.nc' ))
if not debug:
    print(f'Saving Ocean drift data to {OUTPUT_DIR}')
    with open(os.path.join(OUTPUT_DIR, f'polyfit_toce_ece3_dict.pkl'), 'wb+') as ofh:
        pickle.dump(toce_trends_dict, ofh)
if not debug:
    print(f'Saving Ocean drift data to {OUTPUT_DIR}')
    with open(os.path.join(OUTPUT_DIR, f'polyfit_toce_latitude_dict.pkl'), 'wb+') as ofh:
        pickle.dump(toce_latitude_trends_dict, ofh)

# %% [markdown]
# ## ENSO analysis

# %%
# ## Calculate ENSO index

# Niño 3.4: Average SST anomalies over (5N-5S, 170W-120W)

# %%
print('Performing ENSO analysis', flush=True)
var = 'sea_surface_temperature'

years = sorted(set(experiment_ds['time.year'].values))


# %%
for m in ensemble_members:

    en34_da = calculate_nino_index(experiment_ds['sea_surface_temperature'].sel(member=m), nino_region=3.4)
    en34_da_smoothed = calculate_nino_index(experiment_ds['sea_surface_temperature'].sel(member=m), rolling_window=5, nino_region=3.4)
    en34_da_seasonal = calculate_nino_index(experiment_ds['sea_surface_temperature'].sel(member=m), remove_seasonal_cycle=False, nino_region=3.4)

    en3_da = calculate_nino_index(experiment_ds['sea_surface_temperature'].sel(member=m), nino_region=3)

    if not debug:
        print(f'Saving Nino timeseries data to {OUTPUT_DIR}')
        en34_da.to_netcdf(os.path.join(OUTPUT_DIR, f'nino3_4_m{m}.nc'))
        en34_da_smoothed.to_netcdf(os.path.join(OUTPUT_DIR, f'nino3_4_smoothed_m{m}.nc'))
        en34_da_seasonal.to_netcdf(os.path.join(OUTPUT_DIR, f'nino3_4_seasonal_m{m}.nc'))

    for var in ['total_precipitation_daily', 'surface_pressure', '10m_u_component_of_wind', 'instantaneous_eastward_turbulent_surface_stress']:
        y = experiment_ds[var].sel(member=m)
        
        nino_stats_ds = calculate_linear_relationship(en34_da,y)
        nino_stats_smoothed_ds = calculate_linear_relationship(en34_da_smoothed,y)
        nino_stats_seasonal_ds = calculate_linear_relationship(en34_da_seasonal,y)

        # if not debug:
        print(f'Saving Nino stats data to {OUTPUT_DIR}')
        nino_stats_ds.to_netcdf(os.path.join(OUTPUT_DIR, f'nino3_4_stats_{var}_m{m}.nc'))
        nino_stats_smoothed_ds.to_netcdf(os.path.join(OUTPUT_DIR, f'nino3_4_stats_{var}_smoothed_m{m}.nc'))
        nino_stats_seasonal_ds.to_netcdf(os.path.join(OUTPUT_DIR, f'nino3_4_stats_{var}_seasonal_m{m}.nc'))

        nino_3_stats_ds = calculate_linear_relationship(en3_da,y)
        nino_stats_ds.to_netcdf(os.path.join(OUTPUT_DIR, f'nino3_stats_{var}_m{m}.nc'))

# %%
# nperseg = np.min([40*12, len(en34_da['time'].values)])
# fs = 12

# enso_spectra_dict = {}
# for m in ensemble_members:
#     f, Pxx = calculate_en34_spectra(en34_da)

#     enso_spectra_dict[f'ACE2-NEMO m{m}'] = {'period': 1 / f[1:], 'power': Pxx[1:]*f[1:], 'Pxx': Pxx, 'f': f}

# # Calcualte spectra for ERA5
# era5_nino34_series =  en34_da_era5.sortby('time')
# era5_nino34_detrended = signal.detrend(era5_nino34_series.values)
# f, Pxx = signal.welch(era5_nino34_detrended, fs=fs, nperseg=nperseg, detrend=False)

# enso_spectra_dict['ERA5'] = {'period': 1 / f[1:], 'power': Pxx[1:]*f[1:], 'Pxx': Pxx, 'f': f}

# with open(os.path.join(OUTPUT_DIR, f'enso_spectra_dict.pkl'), 'wb+') as ofh:
#     pickle.dump(enso_spectra_dict, ofh)
