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
import cartopy.feature as cfeature
import calendar
from itertools import chain
import cartopy.mpl.ticker as cticker
import matplotlib.style
import matplotlib as mpl

mpl.style.use('default')

# %%
# Directories for the different experiments
# BASE_DATA_DIR = '/gws/nopw/j04/eerie/cache/bantonio/processed_spinup_data'
BASE_DATA_DIR = '/home/users/bantonio/repos/ace2_nemo_coupler/notebooks/processed_data/'
ece3_spinup_dir = os.path.join(BASE_DATA_DIR, 'EC-Earth3_piControl')

# ace2_data_dir = '/gws/nopw/j04/eerie/cache/bantonio/ace2_data'
ace2_data_dir = '/home/users/bantonio'
sea_mask = xr.load_dataarray(os.path.join(ace2_data_dir, "era5_sea_mask_ACE2.nc"))

# %%
name_lookup = {
           'surface_temperature': {'name':'Surface Temperature', 'units': 'K'},
            'sea_surface_temperature': {'name': 'Sea surface temperature', 'units': 'K', 'abbrev': 'SST'},
               'sea_surface_height': {'name': 'Sea Surface Height', 'units': 'm', 'abbrev': 'SSH'},
               'mixed_layer_depth': {'name': 'Mixed Layer Depth', 'units': 'm', 'abbrev': 'MLD'},
              'sea_ice_fraction': {'name':'Sea Ice Fraction', 'units': 'Fraction', 'abbrev': 'siconc'},
              'sea_ice_thickness': {'name':'Sea Ice Thickness', 'units': 'm', 'abbrev': 'SIthick'},
               'sea_ice_extent': {'name':'Sea Ice Extent', 'units': '$km^2$', 'abbrev': 'SIext'},
              'sea_ice_volume': {'name':'Sea Ice Volume', 'units': '$m^3$', 'abbrev': 'SIvol'},
              'LHTFLsfc': {'name':'Latent heat flux', 'units': '$W/m^2$'}, 
               'SHTFLsfc': {'name':'Sensible heat flux', 'units': '$W/m^2$'}, 
               'DLWRFsfc': {'name':'LW flux down', 'units': '$W/m^2$'}, 
               'ULWRFsfc': {'name':'LW flux up', 'units': '$W/m^2$'},
               'DSWRFsfc': {'name':'SW flux down', 'units': '$W/m^2$'},
               'USWRFsfc': {'name':'SW flux up', 'units': '$W/m^2$'},
               'PRATEsfc': {'name':'Precipitation rate', 'units': '$kg/m^2/s$', 'abbrev': 'TP'},
                'total_precipitation': {'name':'Precipitation rate', 'units': '$kg/m^2/s$', 'abbrev': 'TP'},
               'total_precipitation_daily': {'name':'Precipitation', 'units': 'mm/day', 'abbrev': 'P'},
                'TMP2m': {'name': '2-metre temperature', 'units': '$K$', 'abbrev': 'T2m'},
              '2m_temperature': {'name': '2-metre temperature', 'units': '$K$', 'abbrev': 'T2m'},
               '2m_temperature_sea_points': {'name': '2-metre temperature (sea points)', 'units': '$K$', 'abbrev': 'T2m sea'},
              'mean_surface_sensible_heat_flux': {'name': 'Sensible heat flux', 'units': '$W/m^2$', 'abbrev': 'SHF'},
              'mean_surface_latent_heat_flux': {'name': 'Latent heat flux', 'units': '$W/m^2$', 'abbrev': 'LHF'},
               'sea_water_potential_temperature': {'name': 'Sea water potential temperature', 'units': '$K$', 'abbrev': r"$\theta_o$$"},
               'mean_surface_upward_long_wave_radiation_flux': {'name': 'Upward LW Radiation Flux',  'units': '$W/m^2$'},
               'mean_surface_downward_long_wave_radiation_flux': {'name': 'Downward LW Radiation Flux', 'units': '$W/m^2$'},
               'mean_surface_upward_short_wave_radiation_flux': {'name': 'Upward SW Radiation Flux', 'abbrev':r'$R_{sw\uparrow}$','units': '$W/m^2$'},
               'mean_surface_downward_short_wave_radiation_flux': {'name': 'Downward SW Radiation Flux', 'abbrev':r'$R_{sw\downarrow}$','units': '$W/m^2$'},
            'mean_surface_net_short_wave_radiation_flux': {'name': 'Net SW Radiation Flux','abbrev': r'$R_{sw,net}$', 'units': '$W/m^2$'},
               'mean_surface_upward_short_wave_radiation_flux_oce': {'name': 'Upward SW Radiation Flux (ocean)', 'abbrev':r'$R_{sw\uparrow}$','units': '$W/m^2$'},
               'mean_surface_downward_short_wave_radiation_flux_oce': {'name': 'Downward SW Radiation Flux (ocean)', 'abbrev':r'$R_{sw\downarrow}$','units': '$W/m^2$'},
            'mean_surface_net_short_wave_radiation_flux_oce': {'name': 'Net SW Radiation Flux (ocean)','abbrev': r'$R_{sw,net}$', 'units': '$W/m^2$'},
                'mean_surface_upward_short_wave_radiation_flux_ice': {'name': 'Upward SW Radiation Flux (ice)', 'abbrev':r'$R_{sw\uparrow}$','units': '$W/m^2$'},
               'mean_surface_downward_short_wave_radiation_flux_ice': {'name': 'Downward SW Radiation Flux (ice)', 'abbrev':r'$R_{sw\downarrow}$','units': '$W/m^2$'},
            'mean_surface_net_short_wave_radiation_flux_ice': {'name': 'Net SW Radiation Flux (ice)','abbrev': r'$R_{sw,net}$', 'units': '$W/m^2$'},
               'mean_surface_net_long_wave_radiation_flux': {'name': 'Net LW Radiation Flux', 'abbrev':r'$R_{lw,net}$', 'units': '$W/m^2$'},
               'heat_content': {'name': 'Ocean heat content', 'abbrev': 'OHC', 'units': '$J/m^2$'},
               'surface_temperature_difference': {'name': 'T2m - Sea Ice Temperature', 'units': '$K$'},
               'total_water_path': {'name': 'Total water path', 'abbrev': 'TWP', 'units': '$mm$'},
               'albedo_oce': {'name': 'Albedo (ocean)', 'abbrev': r'$\alpha_{oce}$', 'units': 'Fraction'},
               'albedo_ice': {'name': 'Albedo (ice)', 'abbrev': r'$\alpha_{ice}$', 'units': 'Fraction'},
               '10m_u_component_of_wind': {'name': '10m eastward wind', 'units': '$m s^{-1}$'},
               'instantaneous_eastward_turbulent_surface_stress': {'name': 'Eastward wind stress', 'units': '$Nm^{-2}$'},
                'instantaneous_northward_turbulent_surface_stress': {'name': 'Northward wind stress', 'units': '$Nm^{-2}$'},
                'total_heat_flux': {'name': 'Total heat flux', 'units': '$Wm^{-2}$', 'abbrev': 'Total HF'},
               'total_heat_flux_oce': {'name': 'Total heat flux (ocean)', 'units': '$Wm^{-2}$', 'abbrev': 'Total HF (oce)'},
               'mean_surface_heat_flux': {'name': 'Latent + sensible heat flux', 'units': '$Wm^{-2}$', 'abbrev': 'LHF + SHF'}
}

# %% [markdown]
# ## Plots of aggregated evolution in time

# %%
with open(os.path.join(ece3_spinup_dir, f'mean_dict.pkl'), 'rb') as ifh:
    ece_spinup_mean_dict = pickle.load(ifh)

# %%
vars_to_plot = [
                'sea_surface_temperature',  'sea_ice_volume', '2m_temperature_sea_points']
area_name = 'Global'
ncols=3
nrows = int(np.ceil(len(vars_to_plot)/ncols))

fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*6, 4*nrows))
fig.tight_layout(pad=5)
handles = []
labels = []

time_vals = ece_spinup_mean_dict[area_name]['mean']['time']
for n, var in enumerate(vars_to_plot):

    row = int(n/ncols)
    col = n%ncols

    label = 'EC3_piControl'

    if var == 'sea_ice_volume':
        aggregation='UnweightedSum'
    else:
        aggregation='mean'
    ece_time_series = ece_spinup_mean_dict[area_name][aggregation][var].groupby('time.year').mean()
        
    h = (ece_time_series ).plot(ax=axs[col], label=label)    

    if n ==0:
        handles.append(h[0])
        labels.append(label)

    axs[col].set_title(f"{var}")
    axs[col].set_ylabel(f"{name_lookup[var]['abbrev']} [{name_lookup[var]['units']}]")
    axs[col].set_title(f"({string.ascii_lowercase[n]}) {name_lookup[var]['name']}")
    axs[col].set_xlabel('Time')

fig.subplots_adjust(bottom=0.3, wspace=0.33)
axs[-1].legend(handles = handles , labels=labels,loc='upper center', 
             bbox_to_anchor=(-0.3, -0.2),fancybox=False, shadow=False, ncol=4)


# %% [markdown]
# # Global averages of temperatures at depth

# %%
ece_spinup_mean_dict

# %%
