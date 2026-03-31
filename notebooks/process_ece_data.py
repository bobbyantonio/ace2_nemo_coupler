# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
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
import traceback
from tqdm import tqdm
import numpy as np
import xarray as xr
from argparse import ArgumentParser

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
import xesmf as xe

HOME = Path(os.getcwd()).parents[1]
# python_path = sys.executable
# esmkfile_path = python_path.replace('bin/python', 'lib/esmf.mk')
# os.environ['ESMFMKFILE'] = esmkfile_path

# %%
# Add this repo to the path, to enable using all of the helper functions
sys.path.append(str(HOME / 'ace2_nemo_coupler'))

from notebooks.coupling_processing_utils import detrend_dataarray, \
    load_ece3_data, convert_dts_to_first_of_month, calculate_en34, calculate_linear_relationship, \
    calculate_en34_spectra, calculate_correlation, OLEVEL_VALUES, OLEVEL_BIN_EDGES, calculate_lagged_correlations, calculate_anomalies, \
    bjerknes_feedback_analysis, calculate_nino_index, is_notebook
# from notebook_utils.plotting import plot_grid_shared_axes

ece3_var_lookup = {"tas": "2m_temperature", 
                   "tos": "sea_surface_temperature",
                   "siconc": "sea_ice_fraction",
                   'thetao': 'sea_water_potential_temperature',
                   "sithick": "sea_ice_thickness",
                   "pr": "total_precipitation",
                   "zos": "sea_surface_height", # sea surface height above geoid
                   "hfls": 'mean_surface_latent_heat_flux',
                    "hfss": 'mean_surface_sensible_heat_flux',
                    "rlds": "mean_surface_downward_long_wave_radiation_flux",
                    "rlus": "mean_surface_upward_long_wave_radiation_flux",
                    "rsds": "mean_surface_downward_short_wave_radiation_flux",
                    "rsus": "mean_surface_upward_short_wave_radiation_flux",
                   "mlotst": "mixed_layer_depth",
                   'tauu': 'instantaneous_eastward_turbulent_surface_stress',
                   'tauv': 'instantaneous_northward_turbulent_surface_stress',
                   "uo": "ssu",
                   "vo": "ssv",
                   "uas": "10m_u_component_of_wind",
                   "vas": "10m_v_component_of_wind",
                   'psl': 'surface_pressure',
                   'so': 'salinity',
                   'sos': 'sea_surface_salinity'
                  }

all_ocean_t_vars = ['tos', 'siconc', 'thetao', 'sithick', 'zos', 'mlotst', 'thetao', 'thkcello', 'so', 'sos']
all_ocean_u_vars = ['uo', 'uas']
all_ocean_v_vars = ['vo', 'vas']
all_ocean_vars = all_ocean_t_vars + all_ocean_u_vars + all_ocean_v_vars


all_atmosphere_vars = [k for k in ece3_var_lookup.keys() if k not in all_ocean_vars]

DEFAULT_ANALYSIS_VARS = ['hfls', 'hfss', 'mlotst', 'pr', 'psl', 'rlds', 'rlus', 'rsds', 'rsus', 'siconc', 'sithick', 'tas', 'tauu', 'tauv', 'thetao', 'tos', 'uas', 'uo', 'vas', 'vo', 'zos']

# %%
if is_notebook():
    years=range(1951,1952)
    ece3_experiment_id = 'EC-Earth3_piControl'
    debug=True
    month_lag_max = 1
    # ece3_data_dir = '/gws/nopw/j04/eerie/cache/portegam/EC-Earth3.3/aa3x-exp1-climatology_start'
    # base_output_dir = '/gws/nopw/j04/eerie/cache/bantonio/processed_spinup_data'
    # analysis_vars = ['thkcello', 'hfls', 'hfss', 'mlotst', 'pr', 'psl', 
                     # 'rlds', 'rlus', 'rsds', 'rsus', 'siconc', 
                     # 'sithick', 'tas', 'tauu', 'tauv', 'thetao', 'tos', 
                     # 'uas', 'uo', 'vas', 'vo']
    var_glob_string = '*/{var}/*/*'
    # ace2_data_dir = '/gws/nopw/j04/eerie/cache/bantonio/ace2_data'
    ace2_data_dir = '/home/users/bantonio'
    analysis_vars = ['thetao', 'sithick', 'tas', 'tos','sos', 'so', 'siconc']
    ece3_data_dir = '/work/scratch-pw4/portega'
    base_output_dir = '/home/users/bantonio/repos/ace2_nemo_coupler/notebooks/processed_data'
    ocean_level_bin_edges=OLEVEL_BIN_EDGES
    bin_ocean_levels=True
else:
    parser = ArgumentParser()
    parser.add_argument('--ece3-experiment-id', type=str, required=True,
                       help="Experiment ID of the EC-Earth 3 run, e.g. EC-Earth3_piControl")
    parser.add_argument('--month-lag-max', type=int, default=None, help="The maximum lag to use when calculating lagged correlations")
    parser.add_argument('--debug', action='store_true', help="Activate debug mode")
    parser.add_argument('--years', type=str, default='1951-2021',
                       help="Range of years to process; only the top 10 years will be used if --debug is provided as an arg")
    parser.add_argument('--ece3-data-dir', type=str,
                       help="Path to the EC-Earth3 dataset")
    parser.add_argument('--ace2-data-dir', type=str,
                       help="Path to the ACE2 data, for grid files in order to regrid to a common regular 1 degree grid, sea mask, and grid area files.")
    parser.add_argument('--base-output-dir', type=str,
                       help="Root folder for outputs")
    parser.add_argument('--analysis-vars', nargs='+', default=DEFAULT_ANALYSIS_VARS,
                       help="Variables to include in the analysis")
    parser.add_argument('--var-glob-string', type=str, default='{var}',
                       help="Glob string to use when selecting data, depending on the file structure. On JASMIN, this is '*/{var}/*/*'")
    parser.add_argument('--bin-ocean-levels', action='store_true', 
                        help='If specified, then ocean levels are binned rather than selected'
                       )
    args = parser.parse_args()

    ece3_experiment_id = args.ece3_experiment_id
    debug = args.debug
    month_lag_max = args.month_lag_max
    ece3_data_dir = args.ece3_data_dir
    ace2_data_dir = args.ace2_data_dir
    var_glob_string = args.var_glob_string
    years_split = args.years.split('-')
    base_output_dir = args.base_output_dir
    analysis_vars = args.analysis_vars
    years = range(int(years_split[0]), int(years_split[1])+1)
    bin_ocean_levels = args.bin_ocean_levels
    if debug:
        years = years[:10]

if 'hist-1950' in ece3_experiment_id or 'historical' in ece3_experiment_id:
    ece3_years = sorted(set(range(1951,2014)).intersection(set(years)))
else:
    ece3_years = years

OUTPUT_DIR = os.path.join(base_output_dir, ece3_experiment_id)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ACE2 grid / sea mask
ace2grid = xr.load_dataset(os.path.join(ace2_data_dir, "grid.nc"))
sea_mask = xr.load_dataarray(os.path.join(ace2_data_dir, "era5_sea_mask_ACE2.nc"))
ace2_grid_area = xr.load_dataset(os.path.join(ace2_data_dir, "gridarea.nc"))['cell_area']

# %%
# Make sure atmosphere has at least one variable
atmosphere_vars = list(set(all_atmosphere_vars).intersection(analysis_vars).union({'tas'}))
ocean_vars = {'t': list(set(all_ocean_t_vars).intersection(analysis_vars).union({'tos'})),
              'u': list(set(all_ocean_u_vars).intersection(analysis_vars)),
              'v': list(set(all_ocean_v_vars).intersection(analysis_vars))}

ece3_var_lookup = {k: v for k, v in ece3_var_lookup.items() if k in atmosphere_vars + list(chain.from_iterable(list(ocean_vars.values())))}
all_renamed_vars = list(ece3_var_lookup.values())

# %%
print('Loading atmospere data', flush=True)
# ECE3 data
ds_list = [load_ece3_data(var=var,
                          ece3_data_dir = os.path.join(ece3_data_dir, var_glob_string.format(var=var)), 
                          years=ece3_years, ece3_experiment_id=ece3_experiment_id) 
                        for var in atmosphere_vars]

# reset coords required because of mismatch in height for 2-metre temperature and 10m winds
ece3_atm_ds = xr.merge([item.reset_coords(drop=True) for item in ds_list], compat='no_conflicts')

regridder_ece3_atm = xe.Regridder(ece3_atm_ds['tas'].isel(time=0), 
                         ace2grid, 
                         'bilinear',
                         ignore_degenerate=True, 
                         reuse_weights=False, 
                         periodic=True, 
                         filename='weights_ece3_atm.nc')
ece3_atm_ds = regridder_ece3_atm(ece3_atm_ds)


# %%

def load_ece3_data(var,
                   ece3_data_dir,
                   years,
                   ece3_experiment_id,
                   level_values=None,
                   bin_edge_values=None):
    ece3_da = []
    for y in years:
        glob_str = os.path.join(ece3_data_dir, f'{var}_*mon_{ece3_experiment_id}_*_{y}01-{y}12.nc')
        fp = glob(glob_str)
        if len(fp) > 1:
            raise IOError(f'More than one file found for var={var}, year={y}')
        elif len(fp) == 0:
            raise IOError(f'No file found for var={var}, year={y}, glob_str={glob_str}')   
            
        fp = fp[0]
        tmp_da = xr.load_dataset(fp)[var]
        if 'lat' in tmp_da.coords:
            tmp_da = tmp_da.rename({'lat': 'latitude', 'lon': 'longitude'})

        if 'lev' in tmp_da.dims:
   
            if bin_edge_values is not None:
                bin_edges = sorted(bin_edge_values)
                bin_labels = [f'{bin_edges[n]}-{bin_edges[n+1]}' for n in range(len(bin_edges)-1)]
                binned_da = tmp_da.groupby_bins(group='lev', 
                                                bins=bin_edges, 
                                                right=True, 
                                                labels=bin_labels).mean()
                binned_da.name = f'{var}_binned'
                ece3_da.append(binned_da)
                
            if level_values is not None:
                tmp_da = tmp_da.sel(lev=level_values, method='nearest')
            
    
        ece3_da.append(tmp_da)
    ece3_da = xr.concat(ece3_da, dim='time', coords='minimal')

    return ece3_da 

# %%
var='thetao'
tmp_ece3_data_dir = os.path.join(ece3_data_dir, var_glob_string.format(var=var))
years=ece3_years
level_values=OLEVEL_VALUES
bin_edge_values=OLEVEL_BIN_EDGES
ece3_experiment_id=ece3_experiment_id

# %%
ece3_da = []
y=years[0]
glob_str = os.path.join(tmp_ece3_data_dir, f'{var}_*mon_{ece3_experiment_id}_*_{y}01-{y}12.nc')
fp = glob(glob_str)
if len(fp) > 1:
    raise IOError(f'More than one file found for var={var}, year={y}')
elif len(fp) == 0:
    raise IOError(f'No file found for var={var}, year={y}, glob_str={glob_str}')   
    
fp = fp[0]
tmp_da = xr.load_dataset(fp)[var]
if 'lat' in tmp_da.coords:
    tmp_da = tmp_da.rename({'lat': 'latitude', 'lon': 'longitude'})

if 'lev' in tmp_da.dims:

    if bin_edge_values is not None:
        bin_edges = sorted(bin_edge_values)
        bin_labels = [f'{bin_edges[n]}-{bin_edges[n+1]}' for n in range(len(bin_edges)-1)]
        binned_da = tmp_da.groupby_bins(group='lev', 
                                        bins=bin_edges, 
                                        right=True, 
                                        labels=bin_labels).mean()
        binned_da.name = f'{var}_binned'
        ece3_da.append(binned_da)
        
    if level_values is not None:
        tmp_da = tmp_da.sel(lev=level_values, method='nearest')
    

ece3_da.append(tmp_da)
ece3_da = xr.concat(ece3_da, dim='time', coords='minimal')


# %%

def load_ece3_data(var,
                   ece3_data_dir,
                   years,
                   ece3_experiment_id,
                   level_values=None,
                   bin_edge_values=None):
    ece3_da = []
    for y in years:
        glob_str = os.path.join(ece3_data_dir, f'{var}_*mon_{ece3_experiment_id}_*_{y}01-{y}12.nc')
        fp = glob(glob_str)
        if len(fp) > 1:
            raise IOError(f'More than one file found for var={var}, year={y}')
        elif len(fp) == 0:
            raise IOError(f'No file found for var={var}, year={y}, glob_str={glob_str}')   
            
        fp = fp[0]
        tmp_da = xr.load_dataset(fp)[var]
        if 'lat' in tmp_da.coords:
            tmp_da = tmp_da.rename({'lat': 'latitude', 'lon': 'longitude'})

        if 'lev' in tmp_da.dims:
   
            if bin_edge_values is not None:
                bin_edges = sorted(bin_edge_values)
                bin_labels = [f'{bin_edges[n]}-{bin_edges[n+1]}' for n in range(len(bin_edges)-1)]
                binned_da = tmp_da.groupby_bins(group='lev', 
                                                bins=bin_edges, 
                                                right=True, 
                                                labels=bin_labels).mean()
                binned_da.name = f'{var}_binned'
                tmp_da = xr.merge([tmp_da, binned_da], compat='no_conflicts')
                
            if level_values is not None:
                tmp_da = tmp_da.sel(lev=level_values, method='nearest')
                
        ece3_da.append(tmp_da)
    ece3_da = xr.concat(ece3_da, dim='time', coords='minimal')

    return ece3_da


# %%
print('Loading ocean data', flush=True)

ocean_ds_dict = {}
for ocean_grid_type, var_list in ocean_vars.items():
    
    if len(var_list) > 0:
        ocean_ds_dict[ocean_grid_type] = xr.merge([load_ece3_data(var, 
                                                 ece3_data_dir = os.path.join(ece3_data_dir, var_glob_string.format(var=var)),
                                                 years=ece3_years, 
                                                 level_values=OLEVEL_VALUES,
                                                 bin_edge_values=OLEVEL_BIN_EDGES,
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
# regridding ECE3 data
print('Regridding data', flush=True)

ece3_ds = xr.merge([ece3_atm_ds] + list(ocean_ds_dict.values()))
ece3_ds = convert_dts_to_first_of_month(ece3_ds)

ece3_ds = ece3_ds.rename(ece3_var_lookup)

if 'sea_surface_temperature' in all_renamed_vars:
    ece3_ds['sea_surface_temperature'] = ece3_ds['sea_surface_temperature'] + 273

if 'lev' in ece3_ds.dims:
    ece3_ds = ece3_ds.rename({'lev': 'olevel'})

# Sign convention for these fluxes is opposite
if 'mean_surface_sensible_heat_flux' in all_renamed_vars:
    ece3_ds['mean_surface_sensible_heat_flux'] = -1*ece3_ds['mean_surface_sensible_heat_flux']

if 'mean_surface_latent_heat_flux' in all_renamed_vars:
    ece3_ds['mean_surface_latent_heat_flux'] = -1*ece3_ds['mean_surface_latent_heat_flux']

if ece3_experiment_id=='EC-Earth3P_control-1950':
    # Primavera control run seems to have opposite sign convention for upward fluxes
    ece3_ds['mean_surface_upward_short_wave_radiation_flux'] = -1*ece3_ds['mean_surface_upward_short_wave_radiation_flux']
    ece3_ds['mean_surface_upward_long_wave_radiation_flux'] = -1*ece3_ds['mean_surface_upward_long_wave_radiation_flux']


# %%
# Calculate sea ice volume, and sea ice extent in m**2

if 'sea_ice_thickness' in all_renamed_vars and 'sea_ice_fraction' in all_renamed_vars:
    ece3_ds['sea_ice_volume'] = ece3_ds['sea_ice_thickness'] * ece3_ds['sea_ice_fraction'] * ace2_grid_area
    ece3_ds['sea_ice_volume'].attrs['long_name'] = 'Sea ice volume'
    ece3_ds['sea_ice_volume'].attrs['standard_name'] = 'sea_ice_volume'

    ece3_ds['sea_ice_extent'] = (ece3_ds['sea_ice_fraction'] > 0.15).astype(np.int8)* ace2_grid_area
    ece3_ds['sea_ice_extent'] = ece3_ds['sea_ice_extent']/1e6 # Convert to km^2
    ece3_ds['sea_ice_extent'].attrs['long_name'] = 'Sea ice extent'
    ece3_ds['sea_ice_extent'].attrs['standard_name'] = 'sea_ice_extent'

if 'total_precipitation' in all_renamed_vars:
    # Convert precip to mm/day
    ece3_ds['total_precipitation_daily'] = ece3_ds['total_precipitation']*86400

# %%
if 'mean_surface_downward_short_wave_radiation_flux' in all_renamed_vars and 'mean_surface_upward_short_wave_radiation_flux' in all_renamed_vars:
    ece3_ds['mean_surface_net_short_wave_radiation_flux'] = ece3_ds['mean_surface_downward_short_wave_radiation_flux'] - ece3_ds['mean_surface_upward_short_wave_radiation_flux']

if 'mean_surface_downward_long_wave_radiation_flux' in all_renamed_vars and 'mean_surface_upward_long_wave_radiation_flux' in all_renamed_vars:
    ece3_ds['mean_surface_net_long_wave_radiation_flux'] = ece3_ds['mean_surface_downward_long_wave_radiation_flux'] - ece3_ds['mean_surface_upward_long_wave_radiation_flux']

# %%
# Fluxes over ocean points only
all_flux_vars = ['mean_surface_sensible_heat_flux', 
            'mean_surface_latent_heat_flux', 
            'mean_surface_net_short_wave_radiation_flux', 
            'mean_surface_downward_short_wave_radiation_flux',
            'mean_surface_upward_short_wave_radiation_flux', 
            'mean_surface_net_long_wave_radiation_flux',
            'mean_surface_downward_long_wave_radiation_flux',
            'mean_surface_upward_long_wave_radiation_flux'
           ]
flux_vars = list(set(all_flux_vars).intersection(ece3_ds.data_vars))

if len(flux_vars)>0:
    ice_mask = ece3_ds['sea_ice_fraction'].mean('time') > 0.15
    ice_mask_t = ece3_ds['sea_ice_fraction'] > 0.15
    
for var in flux_vars:
    ece3_ds[var] = xr.where(sea_mask,ece3_ds[var], np.nan)
    ece3_ds[f'{var}_oce'] = xr.where(ice_mask_t, np.nan, ece3_ds[var])
    ece3_ds[f'{var}_ice'] = xr.where(ice_mask_t, ece3_ds[var], np.nan)

for suffix in ['', '_oce', '_ice']:
    if 'mean_surface_sensible_heat_flux' in ece3_ds.data_vars and 'mean_surface_latent_heat_flux' in ece3_ds.data_vars:
        ece3_ds[f'mean_surface_heat_flux{suffix}'] = ece3_ds[f'mean_surface_sensible_heat_flux{suffix}'] + ece3_ds[f'mean_surface_latent_heat_flux{suffix}']

        if 'mean_surface_net_long_wave_radiation_flux' in ece3_ds.data_vars:
            ece3_ds[f'non_solar_heat_flux{suffix}'] = ece3_ds[f'mean_surface_sensible_heat_flux{suffix}']  + ece3_ds[f'mean_surface_latent_heat_flux{suffix}'] + ece3_ds[f'mean_surface_net_long_wave_radiation_flux{suffix}']

            if 'mean_surface_net_short_wave_radiation_flux' in ece3_ds.data_vars:
                ece3_ds[f'total_heat_flux{suffix}'] = ece3_ds[f'non_solar_heat_flux{suffix}'] + ece3_ds[f'mean_surface_net_short_wave_radiation_flux{suffix}']
                
    if 'mean_surface_upward_short_wave_radiation_flux' in ece3_ds.data_vars and 'mean_surface_downward_short_wave_radiation_flux' in ece3_ds.data_vars:
        ece3_ds[f'albedo{suffix}'] = ece3_ds[f'mean_surface_upward_short_wave_radiation_flux{suffix}'] / ece3_ds[f'mean_surface_downward_short_wave_radiation_flux{suffix}']


# %%
# Add 2mt over sea points only
ece3_ds['2m_temperature_sea_points'] = xr.where(sea_mask, ece3_ds['2m_temperature'], np.nan)

# %%
# Weights for calculating global averages
weights = np.cos(np.deg2rad(ece3_ds.latitude))
weights = weights / weights.sum().item()

# %%
time_vals = [pd.Timestamp(dt) for dt in ece3_ds['time'].values]

# %% [markdown]
# ## Climate mean state

# %%
print('Calculating mean state', flush=True)

time_range_dict = {'Pre-1980': [dt for dt in time_vals if dt.year <=1980],
                   'Post-1980': [dt for dt in time_vals if dt.year> 1980],
                   'All January': [dt for dt in time_vals if dt.month == 1],
                   'JJA': [dt for dt in time_vals if dt.month in [6,7,8]],
                   'DJF': [dt for dt in time_vals if dt.month in [12,1,2]],
                   '1st month': time_vals[:1],
                   '1st year': time_vals[:12],
                   '5th year': time_vals[48:60],
                   '1st decade': time_vals[:120],
                   'last 50 years': time_vals[-600:],
                   'All': time_vals}

time_mean_state_dict = {}

for name, tvals in time_range_dict.items():
    if len(tvals) > 0:
        time_mean_state_dict[name] = ece3_ds.sel(time=time_vals).mean('time')

if not debug:
    with open(os.path.join(OUTPUT_DIR, f'time_mean_state_dict.pkl'), 'wb+') as ofh:
        pickle.dump(time_mean_state_dict, ofh)

# %%

# %% [markdown]
# ## Spatial aggregations

# %%
print('Calculating spatial aggregations', flush=True)

mean_areas = {'Global': {'min_lat': -90, 'max_lat': 90},
              'Northern Hemisphere': {'min_lat': 0, 'max_lat': 90},
                    'Southern Hemisphere': {'min_lat': -90, 'max_lat': 0},
                    'Tropics': {'min_lat': -20, 'max_lat': 20},
                    'Northern Extratropics': {'min_lat': 30, 'max_lat': 70},
                    'Southern Extratropics': {'min_lat': -70, 'max_lat': -30},
                    'North Atlantic': {'min_lat': 20, 'max_lat': 60, 'min_lon': 280, 'max_lon': 360},
                    'North Pacific': {'min_lat': 20, 'max_lat': 60, 'min_lon': 160, 'max_lon': 260},
                    'South Atlantic': {'min_lat': -60, 'max_lat': -20, 'min_lon': 300, 'max_lon': 360},
                    'South Pacific': {'min_lat': -60, 'max_lat': -20, 'min_lon': 160, 'max_lon': 260},
                    'Tropical Atlantic': {'min_lat': -20, 'max_lat': 20, 'min_lon': 300, 'max_lon': 360},
                    'Tropical Pacific': {'min_lat': -20, 'max_lat': 20, 'min_lon': 160, 'max_lon': 260},
                     'NorthAtlantic26.5N': {'min_lat': 25.9, 'max_lat': 26.9, 'min_lon': 280, 'max_lon': 340}}


mean_dict = {}


for area_name, lat_dict in mean_areas.items():
    
    ece3_mean_ds = ece3_ds.sel(latitude=slice(lat_dict['min_lat'],lat_dict['max_lat'])).weighted(weights.sel(latitude=slice(lat_dict['min_lat'],lat_dict['max_lat']))).mean(['latitude', 'longitude']).sortby('time')

    ece3_unweighted_sum_ds = ece3_ds.sel(latitude=slice(lat_dict['min_lat'],lat_dict['max_lat'])).sum(['latitude', 'longitude']).sortby('time')

    mean_dict[area_name] = {'mean': ece3_mean_ds,
                            'UnweightedSum': ece3_unweighted_sum_ds
                           }

if not debug:
    with open(os.path.join(OUTPUT_DIR, f'mean_dict.pkl'), 'wb+') as ofh:
        pickle.dump(mean_dict, ofh)

# %% [markdown]
# ## Lagged correlations

# %%
lag_vars_list = [['mean_surface_heat_flux', 'sea_surface_temperature'],
                     ['10m_u_component_of_wind', '10m_u_component_of_wind'],
                     ['total_heat_flux', 'sea_surface_temperature'],
                     ['mean_surface_latent_heat_flux', 'sea_surface_temperature'],
                     ['mean_surface_downward_short_wave_radiation_flux','sea_surface_temperature']
                    ]

# %%
print('Calculating spatial aggregations', flush=True)

if month_lag_max is not None:
    for lag_vars in lag_vars_list:

        lag_var1 = lag_vars[0]
        lag_var2 = lag_vars[1]
        if lag_var1 in ece3_ds.data_vars and lag_var2 in ece3_ds.data_vars:
            print(f'Calculating lagged correlations for {lag_var1}, {lag_var2}', flush=True)
            
            
            
            ece_results_dict = calculate_lagged_correlations(
                                                                  ece3_ds, 
                                                                  lag_var1, 
                                                                  lag_var2, 
                                                                  month_lag_max=month_lag_max)
            
            if not debug:
                with open(os.path.join(OUTPUT_DIR, f'lagged_correlations_max{month_lag_max}_{lag_var1}_{lag_var2}.pkl'), 'wb+') as ofh:
                    pickle.dump(ece_results_dict, ofh)

# %% [markdown]
# ## Map of trends

# %%
# Trends for different periods

all_drift_vars = ['sea_surface_temperature', 'sea_ice_thickness', 'sea_ice_fraction', 'sea_surface_height',
             'mean_surface_sensible_heat_flux', 'mean_surface_latent_heat_flux', 'mean_surface_net_long_wave_radiation_flux', 
              'mean_surface_net_short_wave_radiation_flux', 
              'mean_surface_upward_short_wave_radiation_flux',
              'mean_surface_downward_short_wave_radiation_flux',
              'total_precipitation_daily']

drift_vars = set(all_drift_vars).intersection(all_renamed_vars)

trends_time_range_dict = {'Pre-1980': [dt for dt in time_vals if dt.year <=1980],
                           'Post-1980': [dt for dt in time_vals if dt.year> 1980],
                           'First 50 Years': time_vals[:600],
                           'Last 50 Years': time_vals[-600:],
                           'Last 20 Years': time_vals[-240:],
                           'First 20 Years': time_vals[:240],
                           'All': time_vals}

trends_dict = {}

for name, tvals in time_range_dict.items():
    trends_dict[name] = {}
    if len(tvals) > 0:
        for n, varname in enumerate(drift_vars):
            _, polyfit = detrend_dataarray(ece3_ds[varname].sel(time=tvals).groupby('time.year').mean(), 'year')
            trends_dict[name][varname] = polyfit

if not debug:
    print(f'Saving Ocean drift data to {OUTPUT_DIR}')
    with open(os.path.join(OUTPUT_DIR, 'trends_dict.pkl'), 'wb+') as ofh:
        pickle.dump(trends_dict, ofh)




# %%
ocean_drift_var= 'sea_water_potential_temperature'

if ocean_drift_var in ece3_ds.data_vars:
    toce_trends_dict = {}
    toce_latitude_trends_dict = {}

    for name, tvals in time_range_dict.items():
        toce_trends_dict[name] = {}
        toce_latitude_trends_dict[name] = {}
        
        if len(tvals)>0:
            _, polyfit_ece3 = detrend_dataarray(ece3_ds[ocean_drift_var].sel(time=tvals).groupby('time.year').mean().sel(latitude=0, method='nearest').sel(longitude=slice(130,260)), 'year')
            toce_trends_dict[name] = polyfit_ece3
            # Aggregation by latitude
            _, polyfit = detrend_dataarray(ece3_ds[ocean_drift_var].sel(time=tvals).groupby(['time.year']).mean().mean('longitude'), 'year')
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
# ## Bjerknes feedback

# %%
try:
    
    results_dict, anomaly_ds = bjerknes_feedback_analysis(ece3_ds)
            
    if not debug:
        anomaly_ds[[v for v in anomaly_ds  if v.endswith('gradient')]].to_netcdf(os.path.join(OUTPUT_DIR, f'zonal_pacific_gradients.nc'))
        
    if not debug:
        with open(os.path.join(OUTPUT_DIR, f'bjerknes_correlations.pkl'), 'wb+') as ofh:
            pickle.dump(results_dict, ofh)

except Exception as e:
    print('Failed to calculated Bjerknes feedbacks')
    traceback.print_exc()

# %% [markdown]
# ## ENSO analysis

# %%
# ## Calculate ENSO index

# Niño 3.4: Average SST anomalies over (5N-5S, 170W-120W)

# %%
en34_da_ece3 = calculate_nino_index(ece3_ds['sea_surface_temperature'], nino_region=3.4)
en34_da_ece3_seasonal = calculate_nino_index(ece3_ds['sea_surface_temperature'], remove_seasonal_cycle=False, nino_region=3.4)

en3_da_ece3 = calculate_nino_index(ece3_ds['sea_surface_temperature'], nino_region=3)

# %%
if not debug:
    print(f'Saving Nino data to {OUTPUT_DIR}')
    en34_da_ece3.to_netcdf(os.path.join(OUTPUT_DIR, 'nino3_4_ece3.nc'))
    en34_da_ece3_seasonal.to_netcdf(os.path.join(OUTPUT_DIR, 'nino3_4_ece3_seasonal.nc'))

# %%
all_enso_correlation_vars = ['total_precipitation_daily', 'surface_pressure', '10m_u_component_of_wind', 'instantaneous_eastward_turbulent_surface_stress']
enso_correlation_vars = list(set(all_enso_correlation_vars).intersection(all_renamed_vars))

for var in enso_correlation_vars:
    x = en34_da_ece3
    y = ece3_ds[var]
    ece3_nino_stats_ds = calculate_linear_relationship(x,y)
    
    if not debug:
        print(f'Saving Nino stats data to {OUTPUT_DIR}')
        ece3_nino_stats_ds.to_netcdf(os.path.join(OUTPUT_DIR, f'ece3_nino3_4_stats_{var}.nc'))

    x = en3_da_ece3
    y = ece3_ds[var]
    ece3_nino_3_stats_ds = calculate_linear_relationship(x,y)

    if not debug:
        print(f'Saving Nino stats data to {OUTPUT_DIR}')
        ece3_nino_3_stats_ds.to_netcdf(os.path.join(OUTPUT_DIR, f'ece3_nino3_stats_{var}.nc'))

# %%
# def calculate_en34_spectra(da, fs = 12):

#     nperseg = np.min([40*12, len(da['time'].values)])
    
#     nino34_series =  da.sortby('time')
#     nino34_detrended = signal.detrend(nino34_series.values)
#     f, Pxx = signal.welch(nino34_detrended, fs=fs, nperseg=nperseg, detrend=False)

#     return f, Pxx

# %%
# Calcualte spectra for ECE3
f, Pxx = calculate_en34_spectra(en34_da_ece3)

enso_spectra_dict = {'period': 1 / f[1:], 'power': Pxx[1:]*f[1:], 'Pxx': Pxx, 'f': f}
if not debug:
    with open(os.path.join(OUTPUT_DIR, f'enso_spectra_dict.pkl'), 'wb+') as ofh:
        pickle.dump(enso_spectra_dict, ofh)
