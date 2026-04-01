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
import string
import pickle
import numpy as np
import xarray as xr
from matplotlib import colormaps

# %%
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
import matplotlib.style
import matplotlib as mpl

mpl.style.use('default')

# %%
sys.path.append('/perm/ecme4254/repos/ace2_nemo_coupler')
from notebooks.plotting import plot_maps_shared_colorbar, plot_imshow_shared_axes, plot_map_grid_no_shared_colorbar, plot_map_grid_cbar_by_row, plot_map_grid_cbar_by_column
from notebooks.coupling_processing_utils import calculate_en34_spectra, is_notebook


PLOT_DIR = '/perm/ecme4254/repos/ace2_nemo_coupler/plots'
ENSO_SPECTRA_DIR = '/perm/ecme4254/repos/ace2_nemo_coupler/notebooks/processed_data/kristian_enso_spectra'
MANUSCRIPT_FIGURE_DIR = '/perm/ecme4254/repos/ace2_nemo_coupler/notebooks/manuscript_figures'

# Directories for the different experiments
BASE_DATA_DIR = '/perm/ecme4254/repos/ace2_nemo_coupler/notebooks/processed_data'
ace2_nemo_control_dir = os.path.join(BASE_DATA_DIR, 'n3.6_ace2_1951_control_compressed_19510101-20210101')
ace2_nemo_hist_dir = os.path.join(BASE_DATA_DIR, 'n3.6_ace2_1951-2021_hist_compressed_19510101-20210101')
ace2_fluxes_dir = os.path.join(BASE_DATA_DIR, 'n3.6_ace2_1951_ace2iceflux_19510101-20210101')
ace2_nemo_skintemp_dir = os.path.join(BASE_DATA_DIR, 'n3.6_ace2_historical_skt_19510101-20210101')

ace2_forced_dir = os.path.join(BASE_DATA_DIR, 'ace2_forced')

ece_control_dir = os.path.join(BASE_DATA_DIR, 'EC-Earth3P_control-1950')
ece_hist_dir = os.path.join(BASE_DATA_DIR, 'EC-Earth3P_hist-1950')
ece_hist_pablo_dir = os.path.join(BASE_DATA_DIR, 'EC-Earth3_historical_Pablo')
era5_dir =  os.path.join(BASE_DATA_DIR, 'ERA5')

sea_mask = xr.load_dataarray("/home/ecme4254/perm/ece3data/era5/era5_sea_mask_ACE2.nc")

# %%
debug=False

# %%
# Plot average fluxes
# Compare average fluxes
ranges = {'A_Qns_ice': [-300,300],
          'A_Qns_oce': [-400,400],
          'A_Qs_oce': [0,600],
          'A_Qs_ice': [0,200],
          'A_Tau_oce':  [-0.25, 0.25],
          'A_Tau_ice':  [-0.1, 0.1],
          'A_TauX_oce':  [-0.25, 0.25],
          'A_TauX_ice':  [-0.1, 0.1],
          'A_TauY_oce':  [-0.25, 0.25],
          'A_TauY_ice':  [-0.1, 0.1],
         'mean_surface_sensible_heat_flux': [-600,600],
         'mean_surface_latent_heat_flux': [-800,800],
          'mean_surface_net_long_wave_radiation_flux': [-150,150],
          'mean_surface_net_short_wave_radiation_flux': [0,1000],
          'instantaneous_eastward_turbulent_surface_stress': [-1, 1],
          'instantaneous_northward_turbulent_surface_stress': [-1,1],
         'evaporation': [-0.0003, 0.0003],
           'A_Precip_liquid': [-0.00005, 0.00005], 'A_Precip_solid': [-0.00001, 0.00001],
          'A_Evap_ice': [-0.00005, 0.00005],
         'A_Evap_total': [-0.0001, 0.0001]}

name_lookup = {'A_Evap_total': {'name':'Total evaporation', 'units': 'kg/m^2/s'},
              'A_Qns_ice': {'name':'Non-solar heat flux (Ice)', 'units': 'W m^-2'},
          'A_Qns_oce': {'name':'Non-solar heat flux (Ocean)', 'units': 'W m^-2'},
          'A_Qs_oce': {'name':'Solar heat flux (Ocean)', 'units': 'W m^-2'},
          'A_Qs_ice': {'name':'Solar heat flux (Ice)', 'units': 'W m^-2'},
          'A_TauX_oce': {'name':'Momentum flux X (Ocean)', 'units': 'N m^-2'},
          'A_TauX_ice':  {'name':'Momentum flux X (Ice)', 'units': 'N m^-2'},
          'A_TauY_oce':  {'name':'Momentum flux Y (Ocean)', 'units': 'N m^-2'},
          'A_TauY_ice':  {'name':'Momentum flux Y (Ice)', 'units': 'N m^-2'},
        'A_Tau_oce':  {'name':'Momentum flux (Ocean)', 'units': 'N m^-2'},
          'A_Tau_ice':  {'name':'Momentum flux (Ice)', 'units': 'N m^-2'},
           'A_Precip_liquid': {'name':'Liquid precipitation', 'units': 'kg/m^2/s'}, 
               'A_Precip_solid': {'name':'Solid precipitation', 'units': '$kg/m^2/s$'},
          'A_Evap_ice': {'name':'Evaporation over ice', 'units': '$kg/m^2/s$'},
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
              'mean_surface_sensible_heat_flux_oce': {'name': 'Sensible heat flux (ocean)', 'units': '$W/m^2$', 'abbrev': 'SHF'},
              'mean_surface_latent_heat_flux': {'name': 'Latent heat flux', 'units': '$W/m^2$', 'abbrev': 'LHF'},
              'mean_surface_latent_heat_flux_oce': {'name': 'Latent heat flux (ocean)', 'units': '$W/m^2$', 'abbrev': 'LHF'},
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
               'instantaneous_eastward_turbulent_surface_stress': {'name': 'Eastward wind stress', 'units': '$Nm^{-2}$', 'abbrev': r'$\tau_{x}$'},
                'instantaneous_northward_turbulent_surface_stress': {'name': 'Northward wind stress', 'units': '$Nm^{-2}$', 'abbrev': r'$\tau_{y}$'},
                'total_heat_flux': {'name': 'Total heat flux', 'units': '$Wm^{-2}$', 'abbrev': 'Total HF'},
               'total_heat_flux_oce': {'name': 'Total heat flux (ocean)', 'units': '$Wm^{-2}$', 'abbrev': 'Total HF (oce)'},
               'mean_surface_heat_flux': {'name': 'Latent + sensible heat flux', 'units': '$Wm^{-2}$', 'abbrev': 'LHF + SHF'}
}


# %% [markdown]
# ## Lagged correlations

# %%
mpl.style.use('default')

# with open(os.path.join(ace2_nemo_control_dir, f'lagged_correlations_max10.pkl'), 'rb') as ifh:
#     ace2_nemo_control_lagged_corr = pickle.load(ifh)
for (lag_var1, lag_var2) in [ ('mean_surface_heat_flux', 'sea_surface_temperature'),
                              ('10m_u_component_of_wind', '10m_u_component_of_wind') ]:

    
    with open(os.path.join(ace2_nemo_control_dir, f"lagged_correlations_max5_{lag_var1}_{lag_var2}.pkl"), 'rb') as ifh:
        ace2_nemo_control_lagged_corr = pickle.load(ifh)
    
    with open(os.path.join(ece_control_dir, f"lagged_correlations_max5_{lag_var1}_{lag_var2}.pkl"), 'rb') as ifh:
        ece_control_lagged_corr = pickle.load(ifh)
    
    lags = list(ace2_nemo_control_lagged_corr.keys())

    if lag_var1 == lag_var2:
        plot_lags = [1]
        lat_range = [-30,30]
        width_height_ratio = [8,3]
    else:
        plot_lags= [-1,0,1]
        lat_range = [-90,90]
        width_height_ratio = [8,5]
    stat= 'corr'
    da_grid = [[ace2_nemo_control_lagged_corr[l][stat].sel(lat=slice(lat_range[0], lat_range[1])),
                ece_control_lagged_corr[l][stat].sel(lat=slice(lat_range[0], lat_range[1]))] for l in plot_lags]
        
    vmax, vmin = 1,-1
    cbar_label = 'Correlation'
    
    
    fig, axs = plot_maps_shared_colorbar(da_grid, 
                              cbar_label,
                              [[f'a) ACE2-NEMO-control lag = {l}', f'b) ECE3P-control lag = {l}'] for l in plot_lags],
                              vmax, 
                              vmin,
                              width_height_ratio =width_height_ratio,
                              shrink_factor=0.7, 
                              projection = ccrs.Robinson(central_longitude=180),
                              wspace=0.001,
                              cbar_height_ratio=0.02,
                              cmap='RdBu_r', 
                              mask=None)
    for r in range(len(plot_lags)):
        for c in range(len(da_grid[0])):
    
            axs[r][c].coastlines()
    
    plt.savefig(os.path.join(MANUSCRIPT_FIGURE_DIR, f"lagged_{stat}_{lag_var1}_{lag_var2}.pdf"), format='pdf', bbox_inches='tight')

# %% [markdown]
# ## Climate mean state - Control run

# %%
mean_state_range_dict = {'total_precipitation_daily': {'vmin': 0, 'vmax': 20, 'cmap': 'Blues'}, 
                         'sea_surface_temperature': {'vmin': 270, 'vmax': 305},
                         '2m_temperature': {'vmin': 225, 'vmax': 305}, 
                         'sea_surface_height': {'vmin': -2, 'vmax': 2},
                          'mean_surface_sensible_heat_flux': {'vmin': -100, 'vmax': 100, 'cmap': 'RdBu_r'},
                         'mean_surface_latent_heat_flux': {'vmin': -200, 'vmax': 200, 'cmap': 'RdBu_r'},
                        'mean_surface_net_long_wave_radiation_flux': {'vmin': -120, 'vmax': 0, 'cmap': 'Blues_r'},
                         'mean_surface_net_short_wave_radiation_flux': {'vmin': 0, 'vmax': 350, 'cmap': 'Reds'},
                        'mean_surface_downward_long_wave_radiation_flux': {'vmin': -120, 'vmax': 120, 'cmap': 'RdBu_r'},
                         'mean_surface_downward_short_wave_radiation_flux': {'vmin': -120, 'vmax': 120, 'cmap': 'RdBu_r'},
                         'sea_ice_extent': {'vmin': 0, 'vmax': 400, 'cmap': 'viridis'},
                        'sea_ice_fraction': {'vmin': 0, 'vmax': 1, 'cmap': 'viridis'},
                        'instantaneous_eastward_turbulent_surface_stress': {'vmin': -0.3, 'vmax': 0.3, 'cmap': 'RdBu_r'},
                        'instantaneous_northward_turbulent_surface_stress': {'vmin': -0.3, 'vmax': 0.3, 'cmap': 'RdBu_r'}}

with open(os.path.join(ace2_nemo_control_dir, f'time_mean_state_dict.pkl'), 'rb') as ifh:
    ace2_nemo_control_time_mean_state_dict = pickle.load(ifh)
with open(os.path.join(ece_control_dir, f'time_mean_state_dict.pkl'), 'rb') as ifh:
    ece_control_time_mean_state_dict = pickle.load(ifh)

# %%
plot_vars= ['10m_u_component_of_wind']

time_period = 'JJA'
da_grid = [ [ace2_nemo_control_time_mean_state_dict[time_period][varname].transpose('latitude', 'longitude').sel(latitude=slice(-30,30)),  
             ece_control_time_mean_state_dict[time_period][varname].transpose('latitude', 'longitude').sel(latitude=slice(-30,30))] for varname in plot_vars]

cbar_labels= [f"{name_lookup[varname]['name']} JJA mean [{name_lookup[varname]['units']}]" for varname in plot_vars]
titles_grid = [['a) ACE2-NEMO-control', 'b) ECE3P-control'], 
               ['c) ACE2-NEMO-control', 'd) ECE3P-control']]
vmax_vals = [mean_state_range_dict.get(varname, {}).get('vmax', None) for varname in plot_vars]
vmin_vals = [mean_state_range_dict.get(varname, {}).get('vmin', None) for varname in plot_vars]
cmaps = [mean_state_range_dict.get(varname, {}).get('cmap', 'RdBu_r') for varname in plot_vars]

plot_map_grid_cbar_by_row(da_grid,
                                cbar_labels,
                                titles_grid ,
                                vmax_vals,
                                vmin_vals,
                                  projection=ccrs.Robinson(central_longitude=180),
                                  cmaps=cmaps,
                                width_height_ratio = [8,3],
                                shrink_factor= 0.7,
                                wspace=0.001,
                                cbar_height_ratio=0.02,
                                )
plt.savefig(os.path.join(MANUSCRIPT_FIGURE_DIR, 'JJA_mean_10m_u_component_of_wind.pdf'), format='pdf', bbox_inches='tight')

# %%
plot_vars= ['total_precipitation_daily', 
            '2m_temperature',
           'sea_surface_temperature', 
            'sea_surface_height']
da_grid = [ [ace2_nemo_control_time_mean_state_dict['All'][varname].transpose('latitude', 'longitude'),  
             ece_control_time_mean_state_dict['All'][varname].transpose('latitude', 'longitude')] for varname in plot_vars]
num_cols = 2
num_rows = 4
cbar_labels= [f"{name_lookup[varname]['name']} mean [{name_lookup[varname]['units']}]" for varname in plot_vars]
titles_grid = [['a) ACE2-NEMO-control', 'b) ECE3P-control'], 
               ['c) ACE2-NEMO-control', 'd) ECE3P-control'],
              ['e) ACE2-NEMO-control', 'f) ECE3P-control'],
              ['g) ACE2-NEMO-control', 'h) ECE3P-control']]
vmax_vals = [mean_state_range_dict.get(varname, {}).get('vmax', None) for varname in plot_vars]
vmin_vals = [mean_state_range_dict.get(varname, {}).get('vmin', None) for varname in plot_vars]
cmaps = [mean_state_range_dict.get(varname, {}).get('cmap', 'RdBu_r') for varname in plot_vars]

plot_map_grid_cbar_by_row(da_grid,
                             num_cols,
                                num_rows,
                                cbar_labels,
                                titles_grid ,
                                vmax_vals,
                                vmin_vals,
                                  projection=ccrs.Robinson(central_longitude=180),
                                  cmaps=cmaps,
                                width_height_ratio = [8,6],
                                shrink_factor= 0.7,
                                wspace=0.001,
                                cbar_height_ratio=0.02,
                                )
plt.savefig(os.path.join(MANUSCRIPT_FIGURE_DIR, f"control_mean_state.pdf"), format='pdf', bbox_inches='tight')

# %%
plot_vars= [
                 'mean_surface_sensible_heat_flux', 
                  'mean_surface_latent_heat_flux', 
                 'mean_surface_net_long_wave_radiation_flux', 
                 'mean_surface_net_short_wave_radiation_flux',
                'instantaneous_eastward_turbulent_surface_stress']
da_grid = [ [ xr.where(sea_mask, ace2_nemo_control_time_mean_state_dict['All'][varname], np.nan).transpose('latitude', 'longitude'),
             xr.where(sea_mask,ece_control_time_mean_state_dict['All'][varname], np.nan).transpose('latitude', 'longitude')] for varname in plot_vars]
num_cols = len(da_grid)
num_rows = len(da_grid[0])
cbar_labels= [f"{name_lookup[varname]['name']} mean [{name_lookup[varname]['units']}]" for varname in plot_vars]
titles_grid = [['a) ACE2-NEMO-control', 'b) ECE3P-control'], 
               ['c) ACE2-NEMO-control', 'd) ECE3P-control'],
               ['e) ACE2-NEMO-control', 'f) ECE3P-control'],
               ['g) ACE2-NEMO-control', 'h) ECE3P-control'],
              ['i) ACE2-NEMO-control', 'j) ECE3P-control']]
vmax_vals = [mean_state_range_dict.get(varname, {}).get('vmax', None) for varname in plot_vars]
vmin_vals = [mean_state_range_dict.get(varname, {}).get('vmin', None) for varname in plot_vars]
cmaps = [mean_state_range_dict.get(varname, {}).get('cmap', 'RdBu_r') for varname in plot_vars]

plot_map_grid_cbar_by_row(da_grid,
                                cbar_labels,
                                titles_grid ,
                                vmax_vals,
                                vmin_vals,
                                  projection=ccrs.Robinson(central_longitude=180),
                                  cmaps=cmaps,
                                width_height_ratio = [8,5],
                                shrink_factor= 0.6,
                                wspace=0.001,
                                cbar_height_ratio=0.02,
                                )

plt.savefig(os.path.join(MANUSCRIPT_FIGURE_DIR, f"control_mean_state_fluxes.pdf"), format='pdf', bbox_inches='tight')

# %%
num_rows = 2
num_cols = 4
width_height_ratio = [6,6]
shrink_factor = 0.7
wspace=0.05
sea_ice_mask = ~np.isnan(ece_control_time_mean_state_dict['All']['sea_ice_fraction'])

land_50m = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor=cfeature.COLORS['land'],
                                        facecolor=cfeature.COLORS['land'])

satellite_height = 2000000

fig = plt.figure(constrained_layout=True, figsize=(shrink_factor*width_height_ratio[0]*num_cols, shrink_factor*width_height_ratio[1]*num_rows))

gs = gridspec.GridSpec(num_rows + 1, num_cols, figure=fig, 
                    width_ratios=[1]* num_cols,
                    height_ratios=[1] * num_rows + [0.1],
                       wspace=wspace)

projections = [ccrs.NearsidePerspective(central_longitude=-140.0, 
                                                         central_latitude=90,
                                                         false_easting=0,
                                                         satellite_height=satellite_height),
              ccrs.NearsidePerspective(central_longitude=-140.0, 
                                                         central_latitude=-90,
                                                         false_easting=0,
                                                         satellite_height=satellite_height)]

plot_axs = [[fig.add_subplot(gs[m, 0:2], projection = projections[m]), fig.add_subplot(gs[m, 1:3], projection = projections[m])]
            for m in range(num_rows)]


da_list = [xr.where(sea_ice_mask, ace2_nemo_control_time_mean_state_dict['All']['sea_ice_fraction'], np.nan),
              ece_control_time_mean_state_dict['All']['sea_ice_fraction']]

titles_grid = [['a) ACE2-NEMO-control', 'b) ECE3P-control'], ['c) ACE2-NEMO-control', 'd) ECE3P-control']]

for row in range(num_rows):
    for col in range(2):

        im = da_list[col].plot(ax=plot_axs[row][col], 
                          vmax=1, vmin=0, 
                          cmap='viridis', 
                          add_colorbar=False, rasterized=True,
                          transform=ccrs.PlateCarree())
        plot_axs[row][col].coastlines()
        plot_axs[row][col].add_feature(land_50m)

        plot_axs[row][col].set_title(titles_grid[row][col])

cbar_ax = fig.add_subplot(gs[row+1, 1:2])
cbar = plt.colorbar(im, cax=cbar_ax, label='Sea ice fraction [0-1]', orientation='horizontal')
cbar.ax.tick_params(labelsize=10)
plt.savefig(os.path.join(MANUSCRIPT_FIGURE_DIR, "sea_ice_fraction.pdf"), format='pdf', bbox_inches='tight')

# %% [markdown]
# ## Difference plots for mean vars
#

# %%
mean_diff_vars = ['mean_surface_sensible_heat_flux', 'mean_surface_latent_heat_flux', 'mean_surface_net_long_wave_radiation_flux', 'mean_surface_net_short_wave_radiation_flux']
data_dict = {varname: (ace2_nemo_control_time_mean_state_dict['All'][varname] - ece_control_time_mean_state_dict['All'][varname]).transpose('latitude', 'longitude') for varname in mean_diff_vars}
limit_dict = {varname: {'vmin': -50, 'vmax': 50} for varname in mean_diff_vars}


fig, axs = plot_map_grid_no_shared_colorbar(data_dict,
                                     ncols=2,
                                nrows=int(np.ceil(len(data_dict.keys())/2)),
                                name_lookup=name_lookup,
                                limit_dict=limit_dict,
                                projection = ccrs.Robinson(central_longitude=180),
                                cbar_loc = 'bottom',
                                cmap='RdBu_r',
                                cbar_frac = 0.08,
                                cbar_shrink=0.7,
                                width_height_ratio = [8,6],
                                  shrink_factor=0.7, 
                                  wspace=0.001,
                                  cbar_height_ratio=0.02)
                                

plt.savefig(os.path.join(MANUSCRIPT_FIGURE_DIR, f"mean_state_diffs.pdf"), format='pdf', bbox_inches='tight')

# %% [markdown]
# # Historical run - evidence for SST behaviour

# %%
with open(os.path.join(ace2_nemo_hist_dir, f'time_mean_state_dict.pkl'), 'rb') as ifh:
    ace2_nemo_hist_time_mean_state_dict = pickle.load(ifh)
with open(os.path.join(ece_hist_dir, f'time_mean_state_dict.pkl'), 'rb') as ifh:
    ece_hist_time_mean_state_dict = pickle.load(ifh)

# %%
ice_mask = ace2_nemo_hist_time_mean_state_dict['All']['sea_ice_fraction'] > 0.1
sea_mask = ~np.isnan(ace2_nemo_hist_time_mean_state_dict['All']['sea_surface_temperature'])

# %%
# Change in SST and sea ice fraction 
hist_time_mean_diff_vars = ['sea_surface_temperature', '2m_temperature']


data_dict = {varname: (ace2_nemo_hist_time_mean_state_dict['Post-1980'][varname] - ace2_nemo_hist_time_mean_state_dict['Pre-1980'][varname]).transpose('latitude', 'longitude') for varname in hist_time_mean_diff_vars}
limit_dict = {'sea_surface_temperature': {'vmin': -5, 'vmax': 5} ,
             'sea_ice_fraction': {'vmin': -0.1, 'vmax': 0.1}}

fig, axs = plot_map_grid_no_shared_colorbar(data_dict,
                                     ncols=2,
                                    nrows=int(np.ceil(len(data_dict.keys())/2)),
                                    name_lookup=name_lookup,
                                    limit_dict=limit_dict,
                                    projection = ccrs.Robinson(central_longitude=180),
                                    cbar_loc = 'bottom',
                                    cmap='RdBu_r',
                                    cbar_frac = 0.08,
                                    cbar_shrink=0.7,
                                    width_height_ratio = [8,6],
                                      shrink_factor=0.7, 
                                      wspace=0.001,
                                      cbar_height_ratio=0.02)


# %%
# trends in heating pre- and post-1980
with open(os.path.join(ace2_nemo_hist_dir, f'trends_dict.pkl'), 'rb') as ifh:
    ace2_nemo_hist_trends_dict = pickle.load(ifh)
with open(os.path.join(ece_hist_dir, f'trends_dict.pkl'), 'rb') as ifh:
    ece_hist_trends_dict = pickle.load(ifh)

# %%
hist_time_mean_diff_vars = ['sea_surface_temperature']


da_grid = [ [ace2_nemo_hist_trends_dict['Pre-1980'][varname]['polyfit_coefficients'].sel(degree=1).transpose('latitude', 'longitude'),  
             ace2_nemo_hist_trends_dict['Post-1980'][varname]['polyfit_coefficients'].sel(degree=1).transpose('latitude', 'longitude')]
            for varname in hist_time_mean_diff_vars]

limit_dict = {'sea_surface_temperature': {'vmin': -5, 'vmax': 5} ,
             'sea_ice_fraction': {'vmin': -0.1, 'vmax': 0.1}}

titles_grid = [['a) ACE-NEMO-hist Pre-1980', 'b) ACE-NEMO-hist Post-1980'], ['','']]
vmax_vals = [0.25, 0.25, None, None]
vmin_vals = [-0.25,-0.25,None, None]
cmaps = ['RdBu_r']*4
cbar_labels =[ 'Sea Surface Temperature regression coefficient [K/year]', 'Sea Surface Temperature regression coefficient [K/year]', '', '']
plot_map_grid_cbar_by_row(da_grid,
                             2,
                                1,
                                cbar_labels,
                                titles_grid ,
                                vmax_vals,
                                vmin_vals,
                                  projection=ccrs.Robinson(central_longitude=180),
                                  cmaps=cmaps,
                                width_height_ratio = [8,6],
                                shrink_factor= 0.7,
                                wspace=0.001,
                                cbar_height_ratio=0.02,
                                )

# %%

hist_time_mean_diff_vars = ['sea_surface_temperature']


da_grid = [ [ace2_nemo_hist_trends_dict['Pre-1980'][varname]['polyfit_coefficients'].sel(degree=1).transpose('latitude', 'longitude'),  
             ace2_nemo_hist_trends_dict['Post-1980'][varname]['polyfit_coefficients'].sel(degree=1).transpose('latitude', 'longitude')]
            for varname in hist_time_mean_diff_vars]

limit_dict = {'sea_surface_temperature': {'vmin': -5, 'vmax': 5} ,
             'sea_ice_fraction': {'vmin': -0.1, 'vmax': 0.1}}

titles_grid = [['a) ACE-NEMO-hist Pre-1980', 'b) ACE-NEMO-hist Post-1980'], ['','']]
vmax_vals = [0.25, 0.25, None, None]
vmin_vals = [-0.25,-0.25,None, None]
cmaps = ['RdBu_r']*4
cbar_labels =[ 'Sea Surface Temperature regression coefficient [K/year]', 'Sea Surface Temperature regression coefficient [K/year]', '', '']
plot_map_grid_cbar_by_row(da_grid,
                             2,
                                1,
                                cbar_labels,
                                titles_grid ,
                                vmax_vals,
                                vmin_vals,
                                  projection=ccrs.Robinson(central_longitude=180),
                                  cmaps=cmaps,
                                width_height_ratio = [8,6],
                                shrink_factor= 0.7,
                                wspace=0.001,
                                cbar_height_ratio=0.02,
                                )


# %%
hist_time_mean_diff_vars = ['mean_surface_sensible_heat_flux', 'mean_surface_latent_heat_flux', 'mean_surface_net_long_wave_radiation_flux', 'mean_surface_net_short_wave_radiation_flux']


data_dict = {varname: (ace2_nemo_hist_time_mean_state_dict['Post-1980'][varname] - ace2_nemo_hist_time_mean_state_dict['Pre-1980'][varname]).transpose('latitude', 'longitude') for varname in hist_time_mean_diff_vars}
limit_dict = {varname: {'vmin': -20, 'vmax': 20} for varname in hist_time_mean_diff_vars}

fig, axs = plot_map_grid_no_shared_colorbar(data_dict,
                                     ncols=2,
                                    nrows=int(np.ceil(len(data_dict.keys())/2)),
                                    name_lookup=name_lookup,
                                    limit_dict=limit_dict,
                                    projection = ccrs.Robinson(central_longitude=180),
                                    cbar_loc = 'bottom',
                                    cmap='RdBu_r',
                                    cbar_frac = 0.08,
                                    cbar_shrink=0.7,
                                    width_height_ratio = [8,6],
                                      shrink_factor=0.7, 
                                      wspace=0.001,
                                      cbar_height_ratio=0.02)

plt.savefig(os.path.join(MANUSCRIPT_FIGURE_DIR, f"hist_mean_state_diffs.pdf"), format='pdf', bbox_inches='tight')

# %%
# Plot of change in temp pre-1980 and change in temp post-1980

hist_time_mean_diff_vars = ['mean_surface_sensible_heat_flux', 'mean_surface_latent_heat_flux', 'mean_surface_net_long_wave_radiation_flux', 'mean_surface_net_short_wave_radiation_flux']


data_dict = {'1951-1980': (ace2_nemo_hist_time_mean_state_dict['Pre-1980'][varname] - ace2_nemo_hist_time_mean_state_dict['Pre-1980'][varname]).transpose('latitude', 'longitude') for varname in hist_time_mean_diff_vars}
limit_dict = {varname: {'vmin': -20, 'vmax': 20} for varname in hist_time_mean_diff_vars}

fig, axs = plot_map_grid_no_shared_colorbar(data_dict,
                                     ncols=2,
                                    nrows=int(np.ceil(len(data_dict.keys())/2)),
                                    name_lookup=name_lookup,
                                    limit_dict=limit_dict,
                                    projection = ccrs.Robinson(central_longitude=180),
                                    cbar_loc = 'bottom',
                                    cmap='RdBu_r',
                                    cbar_frac = 0.08,
                                    cbar_shrink=0.7,
                                    width_height_ratio = [8,6],
                                      shrink_factor=0.7, 
                                      wspace=0.001,
                                      cbar_height_ratio=0.02)

plt.savefig(os.path.join(MANUSCRIPT_FIGURE_DIR, f"hist_mean_state_diffs.pdf"), format='pdf', bbox_inches='tight')

# %% [markdown]
# # Investigating rise in SST at start of run

# %%

# %%
hist_time_mean_diff_vars = ['mean_surface_sensible_heat_flux', 'mean_surface_latent_heat_flux', 'mean_surface_net_long_wave_radiation_flux', 'mean_surface_net_short_wave_radiation_flux', 'total_precipitation_daily']


data_dict = {varname: (ace2_nemo_hist_time_mean_state_dict['1st year'][varname] - ace2_nemo_hist_time_mean_state_dict['All'][varname]).transpose('latitude', 'longitude') for varname in hist_time_mean_diff_vars}
limit_dict = {varname: {'vmin': -50, 'vmax': 50} for varname in hist_time_mean_diff_vars}
limit_dict['total_precipitation_daily'] = {'vmin': -5, 'vmax': 5}
fig, axs = plot_map_grid_no_shared_colorbar(data_dict,
                                     ncols=2,
                                    nrows=int(np.ceil(len(data_dict.keys())/2)),
                                    name_lookup=name_lookup,
                                    limit_dict=limit_dict,
                                    projection = ccrs.Robinson(central_longitude=180),
                                    cbar_loc = 'bottom',
                                    cmap='RdBu_r',
                                    cbar_frac = 0.08,
                                    cbar_shrink=0.7,
                                    width_height_ratio = [8,6],
                                      shrink_factor=0.7, 
                                      wspace=0.001,
                                      cbar_height_ratio=0.02)

plt.savefig(os.path.join(MANUSCRIPT_FIGURE_DIR, f"hist_mean_state_diffs_1st_year.pdf"), format='pdf', bbox_inches='tight')

# %% [markdown]
# # Investigating ice fluxes

# %%
with open(os.path.join(ace2_nemo_control_dir, f'time_mean_state_dict.pkl'), 'rb') as ifh:
    ace2_nemo_control_time_mean_state_dict = pickle.load(ifh)
with open(os.path.join(ece_control_dir, f'time_mean_state_dict.pkl'), 'rb') as ifh:
    ece_control_time_mean_state_dict = pickle.load(ifh)

# %%
# Compare ice/2m temperature difference with ACE2 ice fluxes

# %%

time_range = '1st month'
first_month_ds = ace2_nemo_control_time_mean_state_dict[time_range]

ice_mask = first_month_ds['sea_ice_fraction'] > 0.5
ice_mask_ece = ece_control_time_mean_state_dict[time_range]['sea_ice_fraction'] > 0.5

diff_da = first_month_ds['2m_temperature'] - first_month_ds['sea_surface_temperature']
diff_da = xr.where(ice_mask, first_month_ds['2m_temperature'] - first_month_ds['sea_ice_temperature'], diff_da)

data_grid = [[xr.where(ice_mask, diff_da, np.nan).transpose('latitude', 'longitude'),
             xr.where(ice_mask, first_month_ds['mean_surface_sensible_heat_flux'], np.nan).transpose('latitude', 'longitude'),
            xr.where(ice_mask,  first_month_ds['mean_surface_sensible_heat_flux_raw'], np.nan).transpose('latitude', 'longitude'),
             xr.where(ice_mask_ece, ece_control_time_mean_state_dict[time_range]['mean_surface_sensible_heat_flux'], np.nan)]]

title_grid = [['a) T2m - Sea ice temperature', 'b) Sensible heat flux (ACE2-NEMO)', 'c) Sensible heat flux (ACE2)', 'd) Sensible heat flux (ECE3P-control)']]

name_lookup['mean_surface_sensible_heat_flux_raw'] = {'name': 'Sensible heat flux (ACE2)', 'units': '$W/m^2$'}
name_lookup['mean_surface_sensible_heat_flux_ece'] = {'name': 'Sensible heat flux (ECE)', 'units': '$W/m^2$'}

limit_dict = {'surface_temperature_difference': {'vmin': -10, 'vmax': 10},
              'mean_surface_sensible_heat_flux': {'vmin': -100, 'vmax': 100},
             'mean_surface_sensible_heat_flux_raw': {'vmin': -30, 'vmax': 30},
              'mean_surface_sensible_heat_flux_ece': {'vmin': -50, 'vmax': 50},
             }

vmin_grid =[ [-10,-100,-30,-50]]
vmax_grid = [ [-1*item for item in vmin_grid[0]]]
cmap_grid = [['RdBu_r']*4]
cbar_label_grid = [['K'] + ['$W/m^2$']*3]

fig, axs = plot_map_grid_no_shared_colorbar(data_grid,
                                     ncols=4,
                                    nrows=1,
                                    title_grid=title_grid,
                                    vmin_grid=vmin_grid,
                                    vmax_grid=vmax_grid,
                                    cmap_grid=cmap_grid,
                                    cbar_label_grid=cbar_label_grid,
                                    projection = ccrs.NearsidePerspective(central_longitude=-140.0, 
                                                         central_latitude=90,
                                                         false_easting=0,
                                                         satellite_height=2000000),
                                    cbar_loc = 'bottom',
                                    cbar_frac = 1.0,
                                    cbar_shrink=2.0,
                                    width_height_ratio = [5,5],
                                      shrink_factor=0.7, 
                                      wspace=0.001,
                                      cbar_height_ratio=0.05)

plt.savefig(os.path.join(MANUSCRIPT_FIGURE_DIR, f"shf_comparison.pdf"), format='pdf', bbox_inches='tight')

# %%
# Average temperature differences between SST and 2mT, for ERA5 and control run, first decade
with open(os.path.join(ace2_nemo_control_dir, f'time_mean_state_dict.pkl'), 'rb') as ifh:
    ace2_nemo_control_time_mean_state_dict = pickle.load(ifh)
with open(os.path.join(era5_dir, f'time_mean_state_dict.pkl'), 'rb') as ifh:
    era5_time_mean_state_dict = pickle.load(ifh)

ice_mask = first_month_ds['sea_ice_fraction'] > 0.15

fig, axs = plot_maps_shared_colorbar([[xr.where(ice_mask, ace2_nemo_control_time_mean_state_dict['1st month']['2m_temperature'] - ace2_nemo_control_time_mean_state_dict['1st month']['sea_ice_temperature'], np.nan).transpose('latitude', 'longitude'),
                                       era5_time_mean_state_dict['1st month']['2m_temperature'] - era5_time_mean_state_dict['1st month']['sea_ice_temperature']
                                      ]], 
                          '2-metre temperature - sea ice temperature [K]',
                          [['a) ACE2-NEMO-control', 'b) ERA5']],
                          20, 
                          -20,
                          width_height_ratio = [8,6],
                          shrink_factor=0.7, 
                          projection = ccrs.NearsidePerspective(central_longitude=-140.0, 
                                                         central_latitude=90,
                                                         false_easting=0,
                                                         satellite_height=2000000),
                          wspace=0.001,
                          cbar_height_ratio=0.02,
                          cmap='RdBu_r', 
                          mask=None)

for ax in axs[0]:
    ax.coastlines()


# %%

# %%

ice_mask = first_month_ds['sea_ice_fraction'] > 0.15

first_month_ds = xr.where(ice_mask, ace2_nemo_control_time_mean_state_dict['1st month'], np.nan)
fig, axs = plot_maps_shared_colorbar([[first_month_ds['mean_surface_latent_heat_flux'].transpose('latitude', 'longitude'),
                                       first_month_ds['mean_surface_sensible_heat_flux'].transpose('latitude', 'longitude')
                                      ],[first_month_ds['mean_surface_net_short_wave_radiation_flux'].transpose('latitude', 'longitude'),
                                       first_month_ds['mean_surface_net_long_wave_radiation_flux'].transpose('latitude', 'longitude')
                                      ]], 
                          'Flux [$W/m^2$]',
                          [['a) ', 'b)'],
                          ['c) ', 'd)']],
                          150, 
                          -150,
                          width_height_ratio = [6,6],
                          shrink_factor=1.0, 
                          projection = ccrs.NearsidePerspective(central_longitude=-140.0, 
                                                         central_latitude=90,
                                                         false_easting=0,
                                                         satellite_height=2000000),
                          wspace=0.001,
                          cbar_height_ratio=0.02,
                          cmap='RdBu_r', 
                          mask=None)

for ax in axs[0] + axs[1]:
    ax.coastlines()

# %%
# Difference between era5 bulk ice temp and skin temperature


# %% [markdown]
# ## Line plots

# %%
with open(os.path.join(ace2_fluxes_dir, f'mean_dict.pkl'), 'rb') as ifh:
    acefluxes_mean_dict = pickle.load(ifh)
with open(os.path.join(ece_control_dir, f'mean_dict.pkl'), 'rb') as ifh:
    ece_control_mean_dict = pickle.load(ifh)
    
years = sorted(set(acefluxes_mean_dict['Global']['mean']['time.year'].values))


# %%
fig, axs = plt.subplots(1,2, figsize=(5*2,4))
fig.tight_layout(pad=5)

ace2_fluxes_time_vals = sorted([pd.Timestamp(dt) for dt in acefluxes_mean_dict['Global']['mean']['time'].values])

handles = []
labels = [f'ACE2-NEMO-control (ACE2 Ice Fluxes)', f'ECE3P-control']
for n, var in enumerate(['sea_ice_volume', 'sea_surface_height']):
    
    if var == 'sea_ice_volume':
        agg_type = 'UnweightedSum'
    else:
        agg_type = 'mean'
    annual_temperature = acefluxes_mean_dict['Global'][agg_type][var].sel(member=0).sel(time=ace2_fluxes_time_vals[:120])
        
    h = (annual_temperature).plot(ax=axs[n])

    if n ==0:
        handles.append(h[0])
    
    # era5_annual_temperature = era5_mean_dict[area_name][var].sel(time=time_vals[:120])
    # (era5_annual_temperature).plot(ax=axs, color='k', label='ERA5')

    ece3_annual_temperature = ece_control_mean_dict['Global'][agg_type][var].sel(time=ace2_fluxes_time_vals[:120])
    h_ece = (ece3_annual_temperature).plot(ax=axs[n], color='r')

    if n ==0:
        handles.append(h_ece[0])


    axs[n].set_ylabel(f"{name_lookup[var]['name']} [{name_lookup[var]['units']}]")
    axs[n].set_xlabel('Time')
    axs[n].set_title(f'({string.ascii_lowercase[n]})')
    # axs.set_ylim([290, 296])
fig.subplots_adjust(bottom=0.3, wspace=0.33)

axs[1].legend(handles = handles , labels=labels,loc='upper center', 
             bbox_to_anchor=(-0.2, -0.2),fancybox=False, shadow=False, ncol=4)

plt.savefig(os.path.join(MANUSCRIPT_FIGURE_DIR, f"ace2fluxes_drift.pdf"), format='pdf', bbox_inches='tight')

# %%
## Compare SST for skin temperature, ace fluxes only, and our control run

# %%
with open(os.path.join(ace2_nemo_hist_dir, f'mean_dict.pkl'), 'rb') as ifh:
    ace2_nemo_hist_mean_dict = pickle.load(ifh)

with open(os.path.join(ace2_nemo_control_dir, f'mean_dict.pkl'), 'rb') as ifh:
    ace2_nemo_control_mean_dict = pickle.load(ifh)

with open(os.path.join(ace2_fluxes_dir, f'mean_dict.pkl'), 'rb') as ifh:
    ace2_nemo_acefluxes_mean_dict = pickle.load(ifh)

with open(os.path.join(ace2_nemo_skintemp_dir, f'mean_dict.pkl'), 'rb') as ifh:
    ace2_nemo_skintemp_mean_dict = pickle.load(ifh)

with open(os.path.join(ece_hist_pablo_dir, f'mean_dict.pkl'), 'rb') as ifh:
    ece_pablo_mean_dict = pickle.load(ifh)

with open(os.path.join(ece_hist_dir, f'mean_dict.pkl'), 'rb') as ifh:
    ece_hist_mean_dict = pickle.load(ifh)

with open(os.path.join(ece_control_dir, f'mean_dict.pkl'), 'rb') as ifh:
    ece_control_mean_dict = pickle.load(ifh)

with open(os.path.join(era5_dir, f'mean_dict.pkl'), 'rb') as ifh:
    era5_mean_dict = pickle.load(ifh)

# %%
vars_to_plot = [
                'sea_surface_height', 'sea_surface_temperature',  'sea_ice_volume',
    'mean_surface_downward_short_wave_radiation_flux', 
            'mean_surface_upward_short_wave_radiation_flux', 
            'mean_surface_net_short_wave_radiation_flux',
     'mean_surface_net_long_wave_radiation_flux',
    'mean_surface_latent_heat_flux',
    'mean_surface_sensible_heat_flux','heat_content', 'total_water_path'
]
                # 'sea_surface_height']
area_name = 'Global'
nrows = int(np.ceil(len(vars_to_plot)/2))
ncols=2
fig, axs = plt.subplots(nrows, 2, figsize=(2*6, 4*nrows))
fig.tight_layout(pad=5)
handles = []
labels = []
for n, var in enumerate(vars_to_plot):

    row = int(n/ncols)
    col = n%ncols
    for m in ace2_nemo_hist_mean_dict[area_name]['mean']['member'].values:

        label = f'ACE2-NEMO-control m{m}'
        
        ace2_nemo_time_series = ace2_nemo_control_mean_dict[area_name]['mean'][var].sel(member=m).groupby('time.year').mean()
        h = (ace2_nemo_time_series - ace2_nemo_time_series.sel(year=1951)).plot(ax=axs[row,col], label=label)

        if n ==0:
            handles.append(h[0])
            labels.append(label)
    try:

        label = 'ECE3P-control'
        
        ece_time_series = ece_control_mean_dict[area_name]['mean'][var].groupby('time.year').mean()

        if var in ['mean_surface_latent_heat_flux', 'mean_surface_sensible_heat_flux']:
            ece_time_series.loc[1974] = np.nan
            
        h = (ece_time_series - ece_time_series.sel(year=1951)).plot(ax=axs[row,col], label=label)    

        if n ==0:
            handles.append(h[0])
            labels.append(label)

    except Exception as e:
        pass

   
    axs[row,col].set_ylabel(f"{name_lookup[var]['abbrev']} - {name_lookup[var]['abbrev']}(1951) [{name_lookup[var]['units']}]")
    axs[row,col].set_title(f"({string.ascii_lowercase[n]}) {name_lookup[var]['name']}")

    new_tick_labels = [item.get_text() if item.get_text() != 'Jul' else "" for item in axs[row,col].get_xticklabels()]
    axs[row,col].set_xticklabels(new_tick_labels)
    axs[row,col].set_xlabel('Time')

fig.subplots_adjust(bottom=0.3, wspace=0.33)
axs[-1,-1].legend(handles = handles , labels=labels,loc='upper center', 
             bbox_to_anchor=(-0.3, -0.2),fancybox=False, shadow=False, ncol=4)

    # if not debug:


# %%
vars_to_plot = [
                'sea_surface_temperature',  'sea_ice_volume','heat_content', 'total_heat_flux']
area_name = 'Global'
nrows = int(np.ceil(len(vars_to_plot)/2))
ncols=2
fig, axs = plt.subplots(nrows, 2, figsize=(2*6, 4*nrows))
fig.tight_layout(pad=5)
handles = []
labels = []

time_vals = ace2_nemo_control_mean_dict[area_name]['mean']['time']
for n, var in enumerate(vars_to_plot):

    row = int(n/ncols)
    col = n%ncols
    for m in ace2_nemo_control_mean_dict[area_name]['mean']['member'].values:

        label = f'ACE2-NEMO-control m{m}'
        
        ace2_nemo_time_series = ace2_nemo_control_mean_dict[area_name]['mean'][var].sel(member=m).groupby('time.year').mean()
        
        ace2_nemo_time_series = ace2_nemo_time_series.sel(year=slice(1951,2019))
        h = (ace2_nemo_time_series ).plot(ax=axs[row,col], label=label)

        if n ==0:
            handles.append(h[0])
            labels.append(label)
    try:

        label = 'ECE3P-control'
        
        ece_time_series = ece_control_mean_dict[area_name]['mean'][var].groupby('time.year').mean()

        if var in ['mean_surface_latent_heat_flux', 'mean_surface_sensible_heat_flux']:
            ece_time_series.loc[1974] = np.nan
            
        h = (ece_time_series ).plot(ax=axs[row,col], label=label)    

        if n ==0:
            handles.append(h[0])
            labels.append(label)

    except Exception as e:
        pass

   
    axs[row,col].set_ylabel(f"{name_lookup[var]['abbrev']} [{name_lookup[var]['units']}]")
    axs[row,col].set_title(f"({string.ascii_lowercase[n]}) {name_lookup[var]['name']}")

    new_tick_labels = [item.get_text() if item.get_text() != 'Jul' else "" for item in axs[row,col].get_xticklabels()]
    axs[row,col].set_xticklabels(new_tick_labels)
    axs[row,col].set_xlabel('Time')

fig.subplots_adjust(bottom=0.3, wspace=0.33)
axs[-1,-1].legend(handles = handles , labels=labels,loc='upper center', 
             bbox_to_anchor=(-0.3, -0.2),fancybox=False, shadow=False, ncol=4)

    # if not debug:


# %%
vars_to_plot = [
                'sea_surface_temperature',  'sea_ice_volume',
'sea_surface_height', 'total_heat_flux']
area_name = 'Global'
nrows = int(np.ceil(len(vars_to_plot)/2))
ncols=2
fig, axs = plt.subplots(nrows, 2, figsize=(2*6, 4*nrows))
fig.tight_layout(pad=5)
handles = []
labels = []

time_vals = ace2_nemo_control_mean_dict[area_name]['mean']['time']
for n, var in enumerate(vars_to_plot):

    row = int(n/ncols)
    col = n%ncols
    for m in ace2_nemo_control_mean_dict[area_name]['mean']['member'].values:

        label = f'ACE2-NEMO-control m{m}'
        
        ace2_nemo_time_series = ace2_nemo_control_mean_dict[area_name]['mean'][var].sel(member=m).groupby('time.year').mean()
        
        ace2_nemo_time_series = ace2_nemo_time_series.sel(year=slice(1951,2019))
        h = (ace2_nemo_time_series ).plot(ax=axs[row,col], label=label)

        if n ==0:
            handles.append(h[0])
            labels.append(label)
    # try:

    label = 'ECE3P-control'
    
    ece_time_series = ece_control_mean_dict[area_name]['mean'][var].groupby('time.year').mean()

    if var in ['mean_surface_latent_heat_flux', 'mean_surface_sensible_heat_flux']:
        ece_time_series.loc[1974] = np.nan
        
    h = (ece_time_series ).plot(ax=axs[row,col], label=label, linestyle='--')    

    if n ==0:
        handles.append(h[0])
        labels.append(label)

    # except Exception as e:
    #     pass

   
    axs[row,col].set_ylabel(f"{name_lookup[var]['abbrev']} [{name_lookup[var]['units']}]")
    axs[row,col].set_title(f"({string.ascii_lowercase[n]}) {name_lookup[var]['name']}")

    new_tick_labels = [item.get_text() if item.get_text() != 'Jul' else "" for item in axs[row,col].get_xticklabels()]
    axs[row,col].set_xticklabels(new_tick_labels)
    axs[row,col].set_xlabel('Time')

fig.subplots_adjust(bottom=0.3, wspace=0.33)
axs[-1,-1].legend(handles = handles , labels=labels,loc='upper center', 
             bbox_to_anchor=(-0.3, -0.2),fancybox=False, shadow=False, ncol=4)

    # if not debug:
plt.savefig(os.path.join(MANUSCRIPT_FIGURE_DIR, f"global_drift.pdf"), format='pdf', bbox_inches='tight')

# %%
vars_to_plot = [
                'sea_surface_temperature',  'sea_ice_volume',
'sea_surface_height', 'total_heat_flux']
area_name = 'Global'
nrows = int(np.ceil(len(vars_to_plot)/2))
ncols=2
fig, axs = plt.subplots(nrows, 2, figsize=(2*6, 4*nrows))
fig.tight_layout(pad=5)
handles = []
labels = []

time_vals = ace2_nemo_control_mean_dict[area_name]['mean']['time']
for n, var in enumerate(vars_to_plot):

    row = int(n/ncols)
    col = n%ncols
    for m in ace2_nemo_control_mean_dict[area_name]['mean']['member'].values:

        label = f'ACE2-NEMO-control m{m}'
        
        ace2_nemo_time_series = ace2_nemo_control_mean_dict[area_name]['mean'][var].sel(member=m).groupby('time.year').mean()
        
        ace2_nemo_time_series = ace2_nemo_time_series.sel(year=slice(1951,2019))
        h = (ace2_nemo_time_series - ace2_nemo_time_series.sel(year=1951)).plot(ax=axs[row,col], label=label)

        if n ==0:
            handles.append(h[0])
            labels.append(label)
    try:

        label = 'ECE3P-control'
        
        ece_time_series = ece_control_mean_dict[area_name]['mean'][var].groupby('time.year').mean()

        if var in ['mean_surface_latent_heat_flux', 'mean_surface_sensible_heat_flux']:
            ece_time_series.loc[1974] = np.nan
            
        h = (ece_time_series - ece_time_series.sel(year=1951)).plot(ax=axs[row,col], label=label, linewidth=2, linestyle='--')    

        if n ==0:
            handles.append(h[0])
            labels.append(label)

    except Exception as e:
        pass

   
    axs[row,col].set_ylabel(r'$\Delta$' + f"{name_lookup[var]['abbrev']} [{name_lookup[var]['units']}]")
    axs[row,col].set_title(f"({string.ascii_lowercase[n]}) {name_lookup[var]['name']}")

    new_tick_labels = [item.get_text() if item.get_text() != 'Jul' else "" for item in axs[row,col].get_xticklabels()]
    axs[row,col].set_xticklabels(new_tick_labels)
    axs[row,col].set_xlabel('Time')

fig.subplots_adjust(bottom=0.3, wspace=0.33)
axs[-1,-1].legend(handles = handles , labels=labels,loc='upper center', 
             bbox_to_anchor=(-0.3, -0.2),fancybox=False, shadow=False, ncol=4)

    # if not debug:
plt.savefig(os.path.join(MANUSCRIPT_FIGURE_DIR, f"global_drift_relative.pdf"), format='pdf', bbox_inches='tight')

# %%
# Plot variables next to ERA5
mpl.style.use('default')

ncols =2
nrows=2
fig, axs = plt.subplots(nrows,ncols, figsize=(ncols*6,nrows*4))
fig.tight_layout(pad=5)

area_name = 'Global'

handles=[]
labels=[]
for n, var in enumerate(['sea_surface_temperature', 'mean_surface_latent_heat_flux','total_precipitation_daily', 'total_water_path']):

    row = int(n/ncols)
    col = n-ncols*row

    label = f'ACE2-NEMO-control'
    ace2_nemo_da = ace2_nemo_control_mean_dict[area_name]['mean'][var].sel(member=0).isel(time=range(60))
    h = (ace2_nemo_da).plot(ax=axs[row,col], label=label)

            
    if n ==0:
        handles.append(h[0])
        labels.append(label)
    # try:
    #     ace2_nemo_data_skt = ace2_nemo_skintemp_mean_dict[area_name]['mean'][var].sel(member=0).isel(time=range(60))
    #     (ace2_nemo_data_skt).plot(ax=axs[n], label=f'ACE2-NEMO-skt')
    
    #     ace2_nemo_data_aceflx = ace2_nemo_acefluxes_mean_dict[area_name]['mean'][var].sel(member=0)
    #     (ace2_nemo_data_aceflx).plot(ax=axs[n], label=f'ACE2-NEMO-control (ACE2 Ice Fluxes)')
    
    # except Exception as e:
    #     pass

    try:
        label='ECE3-historical'
        ece3_data = ece_pablo_mean_dict[area_name]['mean'][var].isel(time=range(60))
        h = (ece3_data).plot(ax=axs[row,col], color='r', label=label)

        if n ==0:
            handles.append(h[0])
            labels.append(label)
    
    except Exception as e:
        pass

    try:
        label='ECE3P-control'
        ece3_data = ece_control_mean_dict[area_name]['mean'][var].isel(time=range(60))
        h = (ece3_data).plot(ax=axs[row,col], color='k', label=label)
        if n ==0:
            handles.append(h[0])
            labels.append(label)
    
    except Exception as e:
        pass

    

    axs[row,col].set_ylabel(f"{name_lookup[var]['name']} [{name_lookup[var]['units']}]")
    axs[row,col].set_xlabel('Year')
    axs[row,col].set_title(f'({string.ascii_lowercase[n]})')
    if var =='sea_surface_temperature':
        axs[row,col].set_ylim([290.5, 294.5])
    elif var == 'total_precipitation_daily':
        axs[row,col].set_ylim([2.65, 3.1])
    # axs[row,col].legend(ncols=2)
fig.subplots_adjust(bottom=0.3, wspace=0.33)

axs[1,1].legend(handles = handles , labels=labels,loc='upper center', 
             bbox_to_anchor=(-0.2, -0.2),fancybox=False, shadow=False, ncol=4)

plt.savefig(os.path.join(MANUSCRIPT_FIGURE_DIR, f"SST_evolution_comparison.pdf"), format='pdf', bbox_inches='tight')


# %%
area_name = 'Global'
mpl.style.use('default')
fig, axs = plt.subplots(1,2, figsize=(2*6,4))
fig.tight_layout(pad=5)

handles=[]
labels=[]
for n, var in enumerate(['2m_temperature', 'sea_surface_temperature']):
            
    
    for m in range(3):
        label=f'ACE2-NEMO-hist m{m}'
        annual_temperature =  ace2_nemo_hist_mean_dict[area_name]['mean'][var].groupby('time.year').mean().sel(member=m)
        
        h = (annual_temperature - annual_temperature.sel(year=1951) ).plot(ax=axs[n], label=label)

        if n ==0:
            handles.append(h[0])
            labels.append(label)
    
    era5_annual_temperature = era5_mean_dict[area_name]['mean'][var].groupby('time.year').mean()
    h_era5 = (era5_annual_temperature - era5_annual_temperature.sel(year=1951)).plot(ax=axs[n], color='k', label='ERA5')


    ece3_annual_temperature = ece_pablo_mean_dict[area_name]['mean'][var].groupby('time.year').mean()
    h_ece3 = (ece3_annual_temperature.sel(year=slice(1951,2013)) - ece3_annual_temperature.sel(year=1951)).plot(ax=axs[n], color='r', label='ECE3-historical')
    
    ece3_annual_temperature = ece_hist_mean_dict[area_name]['mean'][var].groupby('time.year').mean()
    h_ece3p = (ece3_annual_temperature.sel(year=slice(1951,2013)) - ece3_annual_temperature.sel(year=1951)).plot(ax=axs[n], color='r', linestyle='--', label='ECE3P-hist')

    if n ==0:
        handles += [h_era5[0], h_ece3[0], h_ece3p[0]]
        labels += ['ERA5', 'ECE3-historical', 'ECE3P-hist']
    axs[n].set_ylabel(r"$\Delta$" + f"{name_lookup[var]['abbrev']} [{name_lookup[var]['units']}]")
    axs[n].set_xlabel('Year')
    axs[n].set_title(f'({string.ascii_lowercase[n]})')
fig.subplots_adjust(bottom=0.1, wspace=0.2)

axs[1].legend(handles = handles , labels=labels,loc='upper center', 
             bbox_to_anchor=(-0.2, -0.15),fancybox=False, shadow=False, ncol=6)

plt.savefig(os.path.join(MANUSCRIPT_FIGURE_DIR, f"global_historical_temp.pdf"), format='pdf', bbox_inches='tight')

# %%
with open(os.path.join(ace2_nemo_hist_dir, f'time_mean_state_dict.pkl'), 'rb') as ifh:
    ace2_nemo_hist_time_mean_state_dict = pickle.load(ifh)

with open(os.path.join(ece_hist_dir, f'time_mean_state_dict.pkl'), 'rb') as ifh:
    ece_hist_time_mean_state_dict = pickle.load(ifh)

# %%
mpl.style.use('default')


area_name = 'Global'
suffix=''

vars_to_plot = [
                f'mean_surface_upward_short_wave_radiation_flux{suffix}', 
                f'mean_surface_downward_short_wave_radiation_flux{suffix}', 
            f'mean_surface_net_short_wave_radiation_flux{suffix}']

ncols=3
nrows = 1

fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*5, 4*nrows))
fig.tight_layout(pad=5)

handles = []
labels = []
for n, var in enumerate(vars_to_plot):

    row = int(n/ncols)
    col = n%ncols
    for m in ace2_nemo_hist_mean_dict[area_name]['mean']['member'].values:
        label = f'ACE2-NEMO-hist m{m}'
        ace2_nemo_time_series = ace2_nemo_hist_mean_dict[area_name]['mean'][var].sel(member=m).groupby('time.year').mean()
        h = (ace2_nemo_time_series - ace2_nemo_time_series.sel(year=1951)).plot(ax=axs[n], label=label)
        if n ==0:
            handles.append(h[0])
            labels.append(label)
    try:
        label = 'ECE3P-hist'
        ece_time_series = ece_hist_mean_dict[area_name]['mean'][var].groupby('time.year').mean()
        h = (ece_time_series - ece_time_series.sel(year=1951)).plot(ax=axs[n], label=label)
        
        if n ==0:
            handles.append(h[0])
            labels.append(label)
            
    except Exception as e:
        pass

    try:
        label = 'ECE3-historical'
        
        ece_time_series = ece_pablo_mean_dict[area_name]['mean'][var].groupby('time.year').mean()
        h = (ece_time_series - ece_time_series.sel(year=1951)).plot(ax=axs[n], label=label) 

                
        if n ==0:
            handles.append(h[0])
            labels.append(label)
            
    except Exception as e:
        pass
        
    axs[n].set_ylabel(f"{name_lookup[var]['abbrev']} - {name_lookup[var]['abbrev']}(1951) [{name_lookup[var]['units']}]")
    axs[n].set_title(f"({string.ascii_lowercase[n]}) {name_lookup[var]['name']}")

    new_tick_labels = [item.get_text() if item.get_text() != 'Jul' else "" for item in axs[n].get_xticklabels()]
    axs[n].set_xticklabels(new_tick_labels)
    axs[n].set_xlabel('Time')

    # axs[n].legend(ncols=2)

fig.subplots_adjust(bottom=0.3, wspace=0.33)

axs[1].legend(handles = handles , labels=labels,loc='upper center', 
             bbox_to_anchor=(0.5, -0.2),fancybox=False, shadow=False, ncol=4)

plt.savefig(os.path.join(MANUSCRIPT_FIGURE_DIR, f"shortwave_radiation_hist_{area_name}{suffix}.pdf"), format='pdf', bbox_inches='tight')

# %%
vars_to_plot = [
                 'total_water_path', 'heat_content']

area_name = 'Global'
ncols=3
nrows = 1

fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*5, 4*nrows))
fig.tight_layout(pad=5)

handles = []
labels = []
for n, var in enumerate(vars_to_plot):

    row = int(n/ncols)
    col = n%ncols
    for m in ace2_nemo_hist_mean_dict[area_name]['mean']['member'].values:
        label = f'ACE2-NEMO-hist m{m}'
        ace2_nemo_time_series = ace2_nemo_hist_mean_dict[area_name]['mean'][var].sel(member=m).groupby('time.year').mean()
        h = (ace2_nemo_time_series - ace2_nemo_time_series.sel(year=1951) ).plot(ax=axs[n], label=label)
        if n ==0:
            handles.append(h[0])
            labels.append(label)
    try:
        label = 'ECE3P-hist'
        ece_time_series = ece_hist_mean_dict[area_name]['mean'][var].groupby('time.year').mean()
        h = (ece_time_series - ece_time_series.sel(year=1951) ).plot(ax=axs[n], label=label)
        
        if n ==0:
            handles.append(h[0])
            labels.append(label)
            
    except Exception as e:
        pass

    try:
        label = 'ECE3-historical'
        
        ece_time_series = ece_pablo_mean_dict[area_name]['mean'][var].groupby('time.year').mean()
        h = (ece_time_series - ece_time_series.sel(year=1951)).plot(ax=axs[n], label=label) 

                
        if n ==0:
            handles.append(h[0])
            labels.append(label)
            
    except Exception as e:
        pass
        
    axs[n].set_ylabel(f"{name_lookup[var]['abbrev']}  [{name_lookup[var]['units']}]")
    axs[n].set_title("")

    new_tick_labels = [item.get_text() if item.get_text() != 'Jul' else "" for item in axs[n].get_xticklabels()]
    axs[n].set_xticklabels(new_tick_labels)
    axs[n].set_xlabel('Time')

    # axs[n].legend(ncols=2)

fig.subplots_adjust(bottom=0.3, wspace=0.33)

axs[1].legend(handles = handles , labels=labels,loc='upper center', 
             bbox_to_anchor=(0.5, -0.2),fancybox=False, shadow=False, ncol=4)


# %%
with open(os.path.join(ace2_nemo_hist_dir, f'mean_dict.pkl'), 'rb') as ifh:
    ace2_nemo_hist_mean_dict = pickle.load(ifh)

with open(os.path.join(ece_hist_dir, f'mean_dict.pkl'), 'rb') as ifh:
    ece_hist_mean_dict = pickle.load(ifh)


vars_to_plot = [
                'mean_surface_net_long_wave_radiation_flux', 
               'mean_surface_latent_heat_flux', 'mean_surface_sensible_heat_flux', 'mean_surface_net_short_wave_radiation_flux']
                # 'sea_surface_height']
area_name = 'Global'

nrows = int(np.ceil(len(vars_to_plot)/2))
ncols=2
fig, axs = plt.subplots(nrows, 2, figsize=(2*6, 4*nrows))
fig.tight_layout(pad=4)
handles = []
labels = []
for n, var in enumerate(vars_to_plot):

    row = int(n/ncols)
    col = n%ncols
    for m in ace2_nemo_hist_mean_dict[area_name]['mean']['member'].values:

        label = f'ACE2-NEMO-hist m{m}'
        
        ace2_nemo_time_series = ace2_nemo_hist_mean_dict[area_name]['mean'][var].sel(member=m).groupby('time.year').mean()
        h = (ace2_nemo_time_series - ace2_nemo_time_series.sel(year=1951)).plot(ax=axs[row,col], label=label)

        if n ==0:
            handles.append(h[0])
            labels.append(label)
    try:

        label = 'ECE3P-hist'
        
        ece_time_series = ece_hist_mean_dict[area_name]['mean'][var].groupby('time.year').mean()
        h = (ece_time_series - ece_time_series.sel(year=1951)).plot(ax=axs[row,col], label=label, linestyle='--')    

        if n ==0:
            handles.append(h[0])
            labels.append(label)

    except Exception as e:
        pass

    try:
        label = 'ECE3-historical'
        ece_time_series = ece_pablo_mean_dict[area_name]['mean'][var].groupby('time.year').mean()
        h = (ece_time_series - ece_time_series.sel(year=1951)).plot(ax=axs[row,col], label=label)  

        
        if n ==0:
            handles.append(h[0])
            labels.append(label)

    except Exception as e:
        pass
        
    axs[row,col].set_ylabel(f"{name_lookup[var]['abbrev']} - {name_lookup[var]['abbrev']}(1951) [{name_lookup[var]['units']}]")
    axs[row,col].set_title(f"({string.ascii_lowercase[n]}) {name_lookup[var]['name']}")

    new_tick_labels = [item.get_text() if item.get_text() != 'Jul' else "" for item in axs[row,col].get_xticklabels()]
    axs[row,col].set_xticklabels(new_tick_labels)
    axs[row,col].set_xlabel('Time')

fig.subplots_adjust(bottom=0.3, wspace=0.33)
axs[-1,-1].legend(handles = handles , labels=labels,loc='upper center', 
             bbox_to_anchor=(-0.3, -0.2),fancybox=False, shadow=False, ncol=4)

    # if not debug:
plt.savefig(os.path.join(MANUSCRIPT_FIGURE_DIR, f"flux_line_plots_{area_name}.pdf"), format='pdf', bbox_inches='tight')

# %% [markdown]
# ## Map of trends

# %%
with open(os.path.join(ace2_nemo_hist_dir, f'trends_dict.pkl'), 'rb') as ifh:
    ace2_nemo_hist_trends_dict = pickle.load(ifh)

# %%
drift_vars = ['mean_surface_upward_short_wave_radiation_flux']
da_grid = [ [xr.load_dataset(os.path.join(ace2_nemo_control_dir, f'polyfit_{varname}.nc' ))['polyfit_coefficients'].sel(degree=1).transpose('latitude', 'longitude'),
             xr.load_dataset(os.path.join(ece_control_dir, f'polyfit_{varname}_ece3.nc' ))['polyfit_coefficients'].sel(degree=1).transpose('latitude', 'longitude')] for varname in drift_vars]
num_cols = 2
num_rows = 4
cbar_labels= [f"{name_lookup[varname]['name']} mean [{name_lookup[varname]['units']}]" for varname in drift_vars]
titles_grid = [['a) ACE2-NEMO', 'b) ECE3P-control'], 
               ['c) ACE2-NEMO', 'd) ECE3P-control'],
               ['e) ACE2-NEMO', 'f) ECE3P-control'],
               ['g) ACE2-NEMO', 'h) ECE3P-control']]
# vmax_vals = [mean_state_range_dict.get(varname, {}).get('vmax', None) for varname in drift_vars]
# vmin_vals = [mean_state_range_dict.get(varname, {}).get('vmin', None) for varname in drift_vars]
vmax_vals = [0.05, 0.03, 0.004, 0.005]
vmin_vals = [-1*item for item in vmax_vals]
cmaps = ['RdBu_r']*4

plot_map_grid_cbar_by_row(da_grid,
                             num_cols,
                                num_rows,
                                cbar_labels,
                                titles_grid ,
                                vmax_vals,
                                vmin_vals,
                                  projection=ccrs.Robinson(central_longitude=180),
                                  cmaps=cmaps,
                                width_height_ratio = [8,6],
                                shrink_factor= 0.7,
                                wspace=0.001,
                                cbar_height_ratio=0.02,
                                )

# %%
mpl.style.use('default')
drift_vars = ['sea_surface_temperature',  'sea_surface_height']
da_grid = [ [xr.load_dataset(os.path.join(ace2_nemo_control_dir, f'polyfit_{varname}.nc' ))['polyfit_coefficients'].sel(degree=1).transpose('latitude', 'longitude'),
             xr.load_dataset(os.path.join(ece_control_dir, f'polyfit_{varname}_ece3.nc' ))['polyfit_coefficients'].sel(degree=1).transpose('latitude', 'longitude')] for varname in drift_vars]
num_cols = 2
num_rows = 4
cbar_labels= [f"{name_lookup[varname]['name']} trend [{name_lookup[varname]['units']}/year]" for varname in drift_vars]
titles_grid = [['a) ACE2-NEMO-control', 'b) ECE3P-control'], 
               ['c) ACE2-NEMO-control', 'd) ECE3P-control']]
# vmax_vals = [mean_state_range_dict.get(varname, {}).get('vmax', None) for varname in drift_vars]
# vmin_vals = [mean_state_range_dict.get(varname, {}).get('vmin', None) for varname in drift_vars]
vmax_vals = [0.05, 0.01]
vmin_vals = [-1*item for item in vmax_vals]
cmaps = ['RdBu_r']*2

plot_map_grid_cbar_by_row(da_grid,
                                cbar_labels,
                                titles_grid ,
                                vmax_vals,
                                vmin_vals,
                                  projection=ccrs.Robinson(central_longitude=180),
                                  cmaps=cmaps,
                                width_height_ratio = [8,6],
                                shrink_factor= 0.7,
                                wspace=0.001,
                                cbar_height_ratio=0.02,
                                )
plt.savefig(os.path.join(MANUSCRIPT_FIGURE_DIR, f'surface_drifts.pdf'), format='pdf')

# %%
ocean_drift_var= 'sea_water_potential_temperature'

polyfit = xr.load_dataset(os.path.join(ace2_nemo_hist_dir, 'polyfit_toce.nc' ))
polyfit_ece3 = xr.load_dataset(os.path.join(ece_hist_dir, 'polyfit_toce_ece3.nc'))

    
da_dict = {'ACE2-NEMO': polyfit['polyfit_coefficients'].sel(degree=1),
           'ECE3P-control': polyfit_ece3['polyfit_coefficients'].sel(degree=1)
          }

fig, axs = plot_imshow_shared_axes(da_grid= [list(da_dict.values())], 
                              num_rows=1, 
                              num_cols=len(da_dict.keys()), 
                              titles_grid=[[f"{string.ascii_lowercase[n]}) {item.replace('ace2', 'ACE2').replace('era5', 'ERA5')}" for item in da_dict.keys()]],
                              cbar_label=f"{name_lookup[ocean_drift_var]['name']} change [{name_lookup[ocean_drift_var]['units']}/year]",
                              shrink_factor=0.7, 
                              cmap='RdBu_r', 
                              mask=None,
                              vmin=-0.05, 
                              vmax=0.05,
                              yincrease=False)

axs[0][0].set_title('(a) ACE2-NEMO')
axs[0][1].set_title('(b) ECE3')

for a in axs[0]:
    a.set_xlabel('Longitude')
    a.set_ylabel('Depth [m]')
    
if not is_notebook():
    plt.savefig(os.path.join(PLOT_DIR, f'ocean_drift_{var}.pdf'), format='pdf')

# %%
fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*5, 4*nrows))
fig.tight_layout(pad=5)

handles = []
labels = []
for n, var in enumerate(vars_to_plot):

    row = int(n/ncols)
    col = n%ncols
    for m in ace2_nemo_hist_mean_dict[area_name]['mean']['member'].values:
        label = f'ACE2-NEMO-hist m{m}'
        ace2_nemo_time_series = ace2_nemo_hist_mean_dict[area_name]['mean'][var].sel(member=m).groupby('time.year').mean()
        h = (ace2_nemo_time_series - ace2_nemo_time_series.sel(year=1951) ).plot(ax=axs[n], label=label)
        if n ==0:
            handles.append(h[0])
            labels.append(label)
    try:
        label = 'ECE3P-hist'
        ece_time_series = ece_hist_mean_dict[area_name]['mean'][var].groupby('time.year').mean()
        h = (ece_time_series - ece_time_series.sel(year=1951) ).plot(ax=axs[n], label=label)
        
        if n ==0:
            handles.append(h[0])
            labels.append(label)
            
    except Exception as e:
        pass

    try:
        label = 'ECE3-historical'
        
        ece_time_series = ece_pablo_mean_dict[area_name]['mean'][var].groupby('time.year').mean()
        h = (ece_time_series - ece_time_series.sel(year=1951)).plot(ax=axs[n], label=label) 

                
        if n ==0:
            handles.append(h[0])
            labels.append(label)
            
    except Exception as e:
        pass
        
    axs[n].set_ylabel(f"{name_lookup[var]['abbrev']}  [{name_lookup[var]['units']}]")
    axs[n].set_title("")

    new_tick_labels = [item.get_text() if item.get_text() != 'Jul' else "" for item in axs[n].get_xticklabels()]
    axs[n].set_xticklabels(new_tick_labels)
    axs[n].set_xlabel('Time')

    # axs[n].legend(ncols=2)

fig.subplots_adjust(bottom=0.3, wspace=0.33)

axs[1].legend(handles = handles , labels=labels,loc='upper center', 
             bbox_to_anchor=(0.5, -0.2),fancybox=False, shadow=False, ncol=4)

# %%
mpl.style.use('default')
drift_vars = ['toce_latitude']
da_grid = [ [xr.load_dataset(os.path.join(ace2_nemo_control_dir, f'polyfit_{varname}.nc' ))['polyfit_coefficients'].sel(degree=1).transpose('olevel', 'latitude'),
             xr.load_dataset(os.path.join(ece_control_dir, f'polyfit_{varname}.nc' ))['polyfit_coefficients'].sel(degree=1).transpose('olevel', 'latitude')] for varname in drift_vars]
num_cols = len(da_grid[0])
num_rows = len(da_grid)
cbar_labels= [f" drift /year" for varname in drift_vars]
titles_grid = [['a) ACE2-NEMO-control', 'b) ECE3P-control']]
# vmax_vals = [mean_state_range_dict.get(varname, {}).get('vmax', None) for varname in drift_vars]
# vmin_vals = [mean_state_range_dict.get(varname, {}).get('vmin', None) for varname in drift_vars]
vmax_vals = [0.075]
vmin_vals = [-1*item for item in vmax_vals]

width_height_ratio = [8,5]
shrink_factor=0.7
wspace=0.001
cbar_height_ratio=0.02
cmap='RdBu_r'
mask=None

fig = plt.figure(constrained_layout=True, figsize=(shrink_factor*width_height_ratio[0]*2, shrink_factor*width_height_ratio[1]))

gs = gridspec.GridSpec(num_rows + 1, num_cols, figure=fig, 
                    width_ratios=[1]* num_cols,
                    height_ratios=[1] * num_rows + [0.02],
                       wspace=wspace) 
plot_axs = [[fig.add_subplot(gs[m, n]) for n in range(num_cols)] for m in range(num_rows)]


for col in range(num_cols):
    im0 = da_grid[0][col].sortby('olevel', ascending=False).plot(ax=plot_axs[0][col], 
                      cmap=cmap, 
                      add_colorbar=False, 
                      rasterized=True,
                               vmin=-0.06,
                               vmax=0.06,
                            yincrease=False
                           )
    plot_axs[0][col].set_title(titles_grid[0][col])
    
    if col == 0:
        plot_axs[0][col].set_ylabel('Ocean level')
    else:
        plot_axs[0][col].set_yticks([])
        plot_axs[0][col].set_ylabel('')
    plot_axs[0][col].set_xlim([-80, 90])

    lat_formatter = cticker.LatitudeFormatter()
    plot_axs[0][col].xaxis.set_major_formatter(lat_formatter)
    plot_axs[0][col].set_xlabel('Latitude')
    
cbar_ax = fig.add_subplot(gs[1, :])
cbar = plt.colorbar(im0, cax=cbar_ax, label='Potential temperature drift [K/year]', orientation='horizontal')
cbar.ax.tick_params(labelsize=10)
plt.savefig(os.path.join(MANUSCRIPT_FIGURE_DIR, f'toce_latitude_drift.pdf'), format='pdf')

# %% [markdown]
# ## Hovmoller plot of ocean temperature

# %%
with open(os.path.join(ace2_nemo_control_dir, f'mean_dict.pkl'), 'rb') as ifh:
    ace2_nemo_control_mean_dict = pickle.load(ifh)

with open(os.path.join(ece_control_dir, f'mean_dict.pkl'), 'rb') as ifh:
    ece_control_mean_dict = pickle.load(ifh)

with open(os.path.join(era5_dir, f'mean_dict.pkl'), 'rb') as ifh:
    era5_mean_dict = pickle.load(ifh)

# %%
max_level=3000
pot_temp_da = ace2_nemo_control_mean_dict['Global']['mean']['sea_water_potential_temperature'].groupby('time.year').mean().sel(member=0).sel(olevel=slice(0,max_level)).transpose('olevel', 'year')
pot_temp_da_ece = ece_control_mean_dict['Global']['mean']['sea_water_potential_temperature'].groupby('time.year').mean().sel(olevel=slice(0,max_level)).transpose('olevel', 'year')

da_dict = {'ACE2-NEMO-control': pot_temp_da- pot_temp_da.sel(year=1951),
           'ECE3P-control': pot_temp_da_ece- pot_temp_da_ece.sel(year=1951)
          }

fig, axs = plot_imshow_shared_axes(da_grid= [list(da_dict.values())], 
                              num_rows=1, 
                              num_cols=len(da_dict.keys()), 
                              titles_grid=[["" for item in da_dict.keys()]],
                              cbar_label=f"Potential temperature change [K]",
                              shrink_factor=0.7, 
                              cmap='RdBu_r', 
                              mask=None,
                              vmin=-0.4, 
                              vmax=0.4,
                              yincrease=False)

axs[0][0].set_title('(a) ACE2-NEMO-control')
axs[0][1].set_title('(b) ECE3P-control')

for a in axs[0]:
    a.set_xlabel('Year')
    a.set_ylabel('Depth [m]')
    
if not is_notebook():
    plt.savefig(os.path.join(PLOT_DIR, f'ocean_hovmoller_control.pdf'), format='pdf')

# %% [markdown]
# # Climatology of seasonal cycle 

# %%
with open(os.path.join(ace2_nemo_control_dir, f'mean_dict.pkl'), 'rb') as ifh:
    ace2_nemo_control_mean_dict = pickle.load(ifh)

with open(os.path.join(ece_control_dir, f'mean_dict.pkl'), 'rb') as ifh:
   ece_control_mean_dict = pickle.load(ifh)


# %%
def calculate_relative_monthly_mean_and_std(monthly_data):
    annual_mean = monthly_data.groupby('time.year').mean()
    year_vals = [pd.Timestamp(dt).year for dt in monthly_data['time'].values]

    annual_mean_expanded = np.interp(year_vals, annual_mean['year'], annual_mean.values)

    relatively_monthly_vals = monthly_data / annual_mean_expanded
    relative_monthly_mean = relatively_monthly_vals.groupby('time.month').mean()
    relative_monthly_std = relatively_monthly_vals.groupby('time.month').std()
    
    return relative_monthly_mean, relative_monthly_std


# %%

vars_to_plot = ['sea_surface_temperature', 'mean_surface_sensible_heat_flux_oce', 'mean_surface_latent_heat_flux_oce', 'instantaneous_eastward_turbulent_surface_stress']
mpl.style.use('default')

area_list = ['Tropics', 'Northern Extratropics', 'Southern Extratropics']
nrows = len(area_list)
ncols = len(vars_to_plot)

fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*4, 4*nrows))
fig.tight_layout(pad=5)

for row, area in enumerate(area_list):
    

    for col, var in enumerate(vars_to_plot):

        monthly_mean, monthly_std = calculate_relative_monthly_mean_and_std(ace2_nemo_control_mean_dict[area]['mean'][var].sel(member=0))
        im = monthly_mean.plot(ax=axs[row,col], label=f'ACE2-NEMO-control', color=colormaps['tab10'].colors[0])  
        axs[row,col].fill_between(monthly_mean['month'].values, 
                                  monthly_mean.values + 2*monthly_std.values, 
                                  monthly_mean.values - 2*monthly_std.values, 
                                  color=colormaps['tab10'].colors[0],
                                  alpha=0.5)

        
        monthly_ece3_mean, monthly_ece3_std = calculate_relative_monthly_mean_and_std(ece_control_mean_dict[area]['mean'][var])
        im_ece = monthly_ece3_mean.plot(ax=axs[row,col], 
                                        label='ECE3P-control',
                                        color=colormaps['tab10'].colors[1])
        axs[row,col].fill_between(monthly_ece3_mean['month'].values, 
                                  monthly_ece3_mean.values + 2*monthly_ece3_std.values, 
                                  monthly_ece3_mean.values - 2*monthly_ece3_std.values,
                                  color=colormaps['tab10'].colors[1],
                                  alpha=0.5)
        
        axs[row,col].set_ylabel(f"{name_lookup[var]['name']} [{name_lookup[var]['units']}]")
        axs[row,col].set_title("")

        axs[row,col].set_xticks([2,4,6,8,10,12])
        axs[row,col].set_xticklabels([2,4,6,8,10,12])
        axs[row,col].set_xlabel('Month')

        
        axs[row,col].set_title(f"({string.ascii_lowercase[row*ncols + col]}) {name_lookup[var]['abbrev']} {area}")

    
fig.subplots_adjust(bottom=0.3, wspace=0.33)
axs[-1,2].legend(handles = [im[0], im_ece[0]] , labels=['ACE2-NEMO-control', 'ECE3P-control'],loc='upper center', 
             bbox_to_anchor=(-0.3, -0.2),fancybox=False, shadow=False, ncol=4)
plt.savefig(os.path.join(MANUSCRIPT_FIGURE_DIR, f'seasonal_cycle.pdf'), format='pdf', bbox_inches='tight')


# %%
def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


# %%
# Interannual variability in the seasonal cycle

vars_to_plot = ['sea_surface_temperature', 'mean_surface_sensible_heat_flux_oce', 'mean_surface_latent_heat_flux_oce', 'instantaneous_eastward_turbulent_surface_stress']
mpl.style.use('default')

area_list = ['Tropics', 'Northern Extratropics', 'Southern Extratropics']
nrows = len(area_list)
ncols = len(vars_to_plot)

fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*4.5, 4*nrows))
fig.tight_layout(pad=5)

for row, area in enumerate(area_list):
    

    for col, var in enumerate(vars_to_plot):

        yearly_data = ace2_nemo_control_mean_dict[area]['mean'][var].sel(member=0).groupby('time.year').mean()
        yearly_ece_data = ece_control_mean_dict[area]['mean'][var].groupby('time.year').mean()
        
        # Strange result in 1974, looks like an obvious problem in the data
        yearly_ece_data.loc[dict(year=1974)] = yearly_ece_data.mean().item()
        
        data = [yearly_data, yearly_ece_data]
        # bplot = axs[row,col].boxplot([yearly_data, yearly_ece_data],
        #                      whis=(5,95), 
        #                      showfliers=False, bootstrap=50, 
        #                      patch_artist=True,
        #                      tick_labels=['ACE2-NEMO-control', 'ECE3P-control'])
        
        
        parts = axs[row,col].violinplot(
                data, showmeans=False, showmedians=False,
                showextrema=False)

        for pc_ix, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colormaps['tab10'].colors[pc_ix])
            pc.set_edgecolor('black')
            pc.set_alpha(0.6)

        quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75], axis=1)
        whiskers = np.array([
            adjacent_values(sorted_array, q1, q3)
            for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
        whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

        inds = np.arange(1, len(medians) + 1)
        axs[row,col].scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
        axs[row,col].vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
        axs[row,col].vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

        # # set style for the axes
        labels = ['ACE2-NEMO-control', 'ECE3P-control']
        axs[row,col].set_xticks(np.arange(1, len(labels) + 1), labels=labels, rotation='horizontal')
        axs[row,col].set_xlim(0.25, len(labels) + 0.75)
        axs[row,col].set_ylabel(f"{name_lookup[var]['name']} [{name_lookup[var]['units']}]")
        axs[row,col].set_title("")

        
        axs[row,col].set_title(f"({string.ascii_lowercase[row*ncols + col]}) {name_lookup[var]['abbrev']} {area}")

        # for patch, color in zip(bplot['boxes'], colormaps['tab10'].colors):
        #     patch.set_facecolor(color)
        
        # for patch in bplot['medians']:
        #     patch.set_color('k')
plt.savefig(os.path.join(MANUSCRIPT_FIGURE_DIR, f'interannual_variability.pdf'), format='pdf', bbox_inches='tight')

# %%
ice_areas = {'Northern Hemisphere': {'min_lat': 0, 'max_lat': 90},
                    'Southern Hemisphere': {'min_lat': -90, 'max_lat': 0}}

# %%
vars_to_plot = ['sea_ice_extent', 'sea_ice_volume']
mpl.style.use('default')
nrows = 2
ncols=2

fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*6, 4*nrows))
fig.tight_layout(pad=5)

for n, var in enumerate(vars_to_plot):

    for area_ix, area_name in enumerate(['Northern Hemisphere', 'Southern Hemisphere']):
        
        # row = int(area_ix/ncols)
        # col = area_ix%ncols
        # for m in [0]:
        # NOTE: no weighting since we are adding volumes and areas
        area_ds = ace2_nemo_control_mean_dict[area_name]['UnweightedSum'][var].sel(member=0)
        monthly_mean = area_ds.groupby('time.month').mean()
        monthly_std = area_ds.groupby('time.month').std()
       
        monthly_mean.plot(ax=axs[n,area_ix], label=f'ACE2-NEMO')  
        
        ece3_area_ds = ece_control_mean_dict[area_name][f'UnweightedSum'][var]

        monthly_ece3_mean = ece3_area_ds.groupby('time.month').mean()
        monthly_ece3_std = ece3_area_ds.groupby('time.month').std()
        monthly_ece3_mean.plot(ax=axs[n,area_ix], label='ECE3P-control')
        
        axs[n,area_ix].fill_between(monthly_mean['month'].values, monthly_mean.values + 2*monthly_std.values, monthly_mean.values - 2*monthly_std.values, alpha=0.5)
        axs[n,area_ix].fill_between(monthly_ece3_mean['month'].values, monthly_ece3_mean.values + 2*monthly_ece3_std.values, monthly_ece3_mean.values - 2*monthly_ece3_std.values, alpha=0.5)
        
        axs[n,area_ix].set_ylabel(f"{name_lookup[var]['name']} [{name_lookup[var]['units']}]")
        axs[n,area_ix].set_title("")
    
        new_tick_labels = [item.get_text() if item.get_text() != 'Jul' else "" for item in axs[n,area_ix].get_xticklabels()]
        axs[n,area_ix].set_xticklabels(new_tick_labels)
        axs[n,area_ix].set_xlabel('Month')
        axs[n,area_ix].set_title(f'({string.ascii_lowercase[2*n+area_ix]}) {area_name}')
        
        axs[n,area_ix].legend()

        axs[n,area_ix].spines["bottom"].set_color("grey")
        axs[n,area_ix].spines["left"].set_color("grey")


plt.savefig(os.path.join(MANUSCRIPT_FIGURE_DIR, 'seasonal_cycle_ice.pdf'), format='pdf')

# %%
area = 'Northern Hemisphere'

vars_to_plot = ['mean_surface_latent_heat_flux', 'mean_surface_sensible_heat_flux','mean_surface_net_short_wave_radiation_flux', 'mean_surface_net_long_wave_radiation_flux']
                # 'LHTFLsfc', 'SHTFLsfc', 'DLWRFsfc', 'ULWRFsfc', 'DSWRFsfc', 'USWRFsfc', 

nrows = 2
ncols=2
fig, axs = plt.subplots(nrows, 2, figsize=(2*6, 4*nrows))
fig.tight_layout(pad=5)

for n, var in enumerate(vars_to_plot):

    row = int(n/ncols)
    col = n%ncols
    for m in [0]:
        monthly_mean = ace2_nemo_control_mean_dict[area]['mean'][var].sel(member=m).groupby('time.month').mean()
        monthly_std = ace2_nemo_control_mean_dict[area]['mean'][var].sel(member=m).groupby('time.month').std()
       
        monthly_mean.plot(ax=axs[row,col], label=f'ACE2-NEMO-control')  
    axs[row,col].fill_between(monthly_mean['month'].values, monthly_mean.values + 2*monthly_std.values, monthly_mean.values - 2*monthly_std.values, alpha=0.5)

    
    monthly_ece3_mean = ece_control_mean_dict[area]['mean'][var].groupby('time.month').mean()
    monthly_ece3_std = ece_control_mean_dict[area]['mean'][var].groupby('time.month').std()
    monthly_ece3_mean.plot(ax=axs[row,col], label='ECE3P-control')
    axs[row,col].fill_between(monthly_ece3_mean['month'].values, monthly_ece3_mean.values + 2*monthly_ece3_std.values, monthly_ece3_mean.values - 2*monthly_ece3_std.values, alpha=0.5)
    
    axs[row,col].set_ylabel(f"{name_lookup[var]['name']} [{name_lookup[var]['units']}]")
    axs[row,col].set_title("")

    new_tick_labels = [item.get_text() if item.get_text() != 'Jul' else "" for item in axs[row,col].get_xticklabels()]
    axs[row,col].set_xticklabels(new_tick_labels)
    axs[row,col].set_xlabel('Month')

    axs[row,col].legend()



# %% [markdown]
# ## ENSO analysis

# %%
# ## Calculate ENSO index

# Niño 3.4: Average SST anomalies over (5N-5S, 170W-120W)

# %%
en34_da = xr.load_dataarray(os.path.join(ace2_nemo_control_dir, 'nino3_4.nc'))
en34_da_ece = xr.load_dataarray(os.path.join(ece_control_dir, 'nino3_4_ece3.nc'))

en34_da['member'] = [int(m) for m in en34_da['member']]
# en34_da_era5['member'] = [int(m) for m in en34_da_era5['member']]

# %%
mpl.style.use('default')
# Line plot of monthly EN3.4
fig, ax = plt.subplots(1,1, figsize=(10,4))

for m in range(3):
    en34_da.sel(member=m).plot(ax=ax, label=f'ACE2-NEMO-control m{m}')
ax.set_title(f"")
ax.set_ylabel("Niño 3.4")

# new_tick_labels = [item.get_text() if item.get_text() != 'Jul' else "" for item in axs[row,col].get_xticklabels()]
# ax.set_xticklabels(new_tick_labels)
ax.set_xlabel('Time')

en34_da_ece.plot(ax=ax, label='ECE3P-control', color='grey', linestyle='--')
ax.set_title(f"")
ax.set_ylabel("Niño 3.4")
ax.set_ylim([-2,2])
# new_tick_labels = [item.get_text() if item.get_text() != 'Jul' else "" for item in axs[row,col].get_xticklabels()]
# ax.set_xticklabels(new_tick_labels)
ax.set_xlabel('Time')
plt.legend(ncols=2,loc= 'lower left')

plt.savefig(os.path.join(MANUSCRIPT_FIGURE_DIR, f'enso3.4.pdf'), format='pdf', bbox_inches='tight')

# %%
mpl.style.use('default')
nino_stats_ds = xr.load_dataset(os.path.join(ace2_nemo_control_dir, 'nino3_4_stats_total_precipitation_daily_m0.nc'))
ece_nino_stats_ds = xr.load_dataset(os.path.join(ece_control_dir, 'ece3_nino3_4_stats.nc'))

fig, axs = plot_maps_shared_colorbar([[nino_stats_ds['slope'],ece_nino_stats_ds['slope']]], 
                          'Daily Precipitation Regressed on Niño 3.4 [mm/day/K]',
                          [['a) ACE2-NEMO-control', 'b) ECE3P-control']],
                          4, 
                          -4,
                          width_height_ratio = [8,6],
                          shrink_factor=0.6, 
                          projection=ccrs.Robinson(central_longitude=180), 
                          wspace=0.001,
                          cbar_height_ratio=0.02,
                          cmap='RdBu_r', 
                          mask=None)

for ax in axs[0]:
    ax.coastlines()

plt.savefig(os.path.join(MANUSCRIPT_FIGURE_DIR, 'enso_correlation.pdf'), format='pdf',  bbox_inches='tight')

# %%
mpl.style.use('default')
da_grid = [[xr.load_dataset(os.path.join(ace2_nemo_control_dir, 'nino3_4_stats_surface_pressure_m0.nc'))['slope'], xr.load_dataset(os.path.join(ece_control_dir, 'ece3_nino3_4_stats_surface_pressure.nc'))['slope']],
           [xr.load_dataset(os.path.join(ace2_nemo_control_dir, 'nino3_4_stats_10m_u_component_of_wind_m0.nc'))['slope'], xr.load_dataset(os.path.join(ece_control_dir, 'ece3_nino3_4_stats_10m_u_component_of_wind.nc'))['slope']]]

titles_grid = [['a) ACE2-NEMO-control', 'b) ECE3P-control'], ['c) ACE2-NEMO-control', 'd) ECE3P-control']]
vmax_vals = [200, 1]
vmin_vals = [-1*v for v in vmax_vals]
plot_map_grid_cbar_by_row(da_grid,
                                ['Sea level pressure regressed on Niño 3.4 [Pa/K]', '10m Eastward wind regressed on Niño 3.4 [m/s/K]'],
                                titles_grid ,
                                vmax_vals,
                                vmin_vals,
                                  projection=ccrs.Robinson(central_longitude=180),
                                  cmaps=['RdBu_r', 'RdBu_r'],
                                width_height_ratio = [8,6],
                                shrink_factor= 0.6,
                                wspace=0.001,
                                cbar_height_ratio=0.02,
                                )

plt.savefig(os.path.join(MANUSCRIPT_FIGURE_DIR, 'enso_correlation_mslp+10mU.pdf'), format='pdf',  bbox_inches='tight')

# %%
# Surface winds autocorrelation

with open(os.path.join(ace2_nemo_control_dir, "lagged_correlations_max5_10m_u_component_of_wind_10m_u_component_of_wind.pkl"), 'rb') as ifh:
    ace2_nemo_control_lagged_corr = pickle.load(ifh)

with open(os.path.join(ece_control_dir, 'lagged_correlations_max5_10m_u_component_of_wind_10m_u_component_of_wind.pkl'), 'rb') as ifh:
    ece_control_lagged_corr = pickle.load(ifh)

lags = list(ace2_nemo_control_lagged_corr.keys())
plot_lags= [1,2,3]
stat= 'corr'
da_grid = [[ace2_nemo_control_lagged_corr[l][stat].isel(member=0).sel(lat=slice(-30,30)), 
            ece_control_lagged_corr[l][stat].sel(lat=slice(-30,30))] for l in plot_lags]
nrows= len(plot_lags)
ncols=2

if stat == 'cov':
    cbar_label = 'Covariance [$m^{2}s^{-2}$]'
    vmin=-15
    vmax=15
elif stat == 'corr':
    cbar_label = 'Correlation'
    vmin=-1
    vmax=1
               # 
fig, axs = plot_maps_shared_colorbar(da_grid,  
                          cbar_label,
                          [[f'a) ACE2-NEMO-control lag={l}', f'b) ECE3P-control lag={l}'] for l in plot_lags],
                          vmin=vmin, 
                          vmax=vmax,
                          width_height_ratio = [8,2],
                          shrink_factor=0.7, 
                          projection = ccrs.Robinson(central_longitude=180),
                          wspace=0.001,
                          cbar_height_ratio=2.0,
                          cmap='RdBu_r', 
                          mask=None)
for r in range(nrows):
    for c in range(ncols):

        axs[r][c].coastlines()


# %% [markdown]
# # Bjerknes feedback

# %%
# anomalous zonal wind stress vs SST gradient

zonal_gradient_ds = xr.load_dataset(os.path.join(ace2_nemo_control_dir, 'zonal_pacific_gradients.nc'))
zonal_gradient_ece_ds = xr.load_dataset(os.path.join(ece_control_dir, 'zonal_pacific_gradients.nc'))

with open(os.path.join(ace2_nemo_control_dir, f'bjerknes_correlations.pkl'), 'rb') as ifh:
    bjerknes_correlations = pickle.load(ifh)
    
with open(os.path.join(ace2_nemo_hist_dir, f'bjerknes_correlations.pkl'), 'rb') as ifh:
    bjerknes_correlations_hist = pickle.load(ifh)

with open(os.path.join(ece_control_dir, f'bjerknes_correlations.pkl'), 'rb') as ifh:
   bjerknes_correlations_ece = pickle.load(ifh)
   
# with open(os.path.join(ece_hist_dir, f'bjerknes_correlations.pkl'), 'rb') as ifh:
#    bjerknes_correlations_ece_hist = pickle.load(ifh)
   
with open(os.path.join(era5_dir, f'bjerknes_correlations.pkl'), 'rb') as ifh:
   bjerknes_correlations_era5 = pickle.load(ifh)
   
with open(os.path.join(ace2_forced_dir, f'bjerknes_correlations_historical.pkl'), 'rb') as ifh:
   bjerknes_correlations_ace2_forced_hist = pickle.load(ifh)
with open(os.path.join(ace2_forced_dir, f'bjerknes_correlations_control.pkl'), 'rb') as ifh:
   bjerknes_correlations_ace2_forced_control = pickle.load(ifh)

# %%
# anomalous zonal wind stress vs SST gradient
mpl.style.use('default')

                      # 'sea_surface_temperature_gradient__total_precipitation_daily': {'cbar_label': 'TP vs $d(SST)/dx$', 'vmax': 0.5, 'divide_by': 1e8, 'units': r'$\times 10^8 \;\text{Pa} (K m^{-1})^{-1}$'},
                        # 'total_precipitation_daily__surface_pressure_gradient': {'cbar_label': '$-d(SLP)/dx$ vs Daily precip', 'vmax': 1, 'divide_by': -1e-5, 'units': r'$\times 10^{-5} \;\text{mm} \text{day}^{-1}(\text{Pa}m^{-1})^{-1}$'},

# The first variable is the x-coordinate, second is the y coordinate
bjerknes_corr_lookup = {'10m_u_component_of_wind__sea_surface_temperature_gradient': {'cbar_label': '$d(SST)/dx$ vs $U_{10m}$', 'vmax': 1, 'divide_by': 1e-7, 'units': r'$\times 10^{-7} \; m s^{-1} (K m^{-1})^{-1}$' },
                        'sea_surface_temperature_gradient__mean_surface_latent_heat_flux': {'cbar_label': 'MSLHF vs $d(SST)/dx$', 'vmax': 1, 'divide_by': 1e8, 'units': r'$\times 10^8 \;\text{Pa} (K m^{-1})^{-1}$'},
                        'surface_pressure_gradient__mean_surface_latent_heat_flux': {'cbar_label': '$-d(SLP)/dx$ vs MSLHF', 'vmax': 1, 'divide_by': -1e6, 'units': r'$\times 10^{-5} \;\text{mm} \text{day}^{-1}(\text{Pa}m^{-1})^{-1}$'},
                      'surface_pressure_gradient__10m_u_component_of_wind': {'cbar_label': '$U_{10m}$ vs $-d(SLP)/dx$', 'vmax': 2, 'divide_by': -1e5, 'units': r'$\times 10^{5} \; ms^{-1} (\text{Pa}m^{-1})^{-1}$'},
                       }
da_grid = [[bjerknes_correlations[cs]['correlation'].transpose("latitude", "longitude") / v['divide_by'],
            bjerknes_correlations_hist[cs]['correlation'].transpose("latitude", "longitude") / v['divide_by'],
            bjerknes_correlations_ece[cs]['correlation'].transpose("latitude", "longitude") / v['divide_by'],
            bjerknes_correlations_era5[cs]['correlation'].transpose("latitude", "longitude") / v['divide_by']] for cs, v in bjerknes_corr_lookup.items()]

vmax_vals = [v['vmax'] for v in bjerknes_corr_lookup.values()]
vmin_vals = [-1*v for v in vmax_vals]
titles_grid = [[f'{string.ascii_lowercase[2*n]}) ACE2-NEMO-control', f'{string.ascii_lowercase[2*n+1]}) ACE2-NEMO-hist', f'{string.ascii_lowercase[2*n+2]}) ECE3P-control', f'{string.ascii_lowercase[2*n+3]}) ERA5'] for n in range(len(da_grid))]
cmaps = ['RdBu_r']*len(da_grid)
cbar_labels = [f"{v['cbar_label']} [{v['units']}]" for v in bjerknes_corr_lookup.values()]


fig, axs = plot_map_grid_cbar_by_row(da_grid,
                                cbar_labels,
                                titles_grid ,
                                vmax_vals,
                                vmin_vals,
                                  projection=ccrs.PlateCarree(central_longitude=180),
                                  cmaps=cmaps,
                                width_height_ratio = [7,3],
                                shrink_factor= 0.8,
                                wspace=0.001,
                                cbar_height_ratio=0.02,
                                     lat_ticks = [-10,0,10],
                                     lon_ticks = range(130,251,20)
                                )

# plt.savefig(os.path.join(MANUSCRIPT_FIGURE_DIR, f'bjerknes_feedback.pdf'), format='pdf', bbox_inches='tight')

# %%
# anomalous zonal wind stress vs SST gradient
mpl.style.use('default')

                      # 'sea_surface_temperature_gradient__total_precipitation_daily': {'cbar_label': 'TP vs $d(SST)/dx$', 'vmax': 0.5, 'divide_by': 1e8, 'units': r'$\times 10^8 \;\text{Pa} (K m^{-1})^{-1}$'},
                        # 'total_precipitation_daily__surface_pressure_gradient': {'cbar_label': '$-d(SLP)/dx$ vs Daily precip', 'vmax': 1, 'divide_by': -1e-5, 'units': r'$\times 10^{-5} \;\text{mm} \text{day}^{-1}(\text{Pa}m^{-1})^{-1}$'},

# The first variable is the x-coordinate, second is the y coordinate
bjerknes_corr_lookup = {'10m_u_component_of_wind__sea_surface_temperature_gradient': {'cbar_label': '$d(SST)/dx$ vs $U_{10m}$', 'vmax': 1, 'divide_by': 1e-7, 'units': r'$\times 10^{-7} \; m s^{-1} (K m^{-1})^{-1}$' },
                        'sea_surface_temperature_gradient__mean_surface_latent_heat_flux': {'cbar_label': 'MSLHF vs $d(SST)/dx$', 'vmax': 1, 'divide_by': 1e8, 'units': r'$\times 10^8 \;\text{Pa} (K m^{-1})^{-1}$'},
                        'surface_pressure_gradient__mean_surface_latent_heat_flux': {'cbar_label': '$-d(SLP)/dx$ vs MSLHF', 'vmax': 1, 'divide_by': -1e6, 'units': r'$\times 10^{-5} \;\text{mm} \text{day}^{-1}(\text{Pa}m^{-1})^{-1}$'},
                      'surface_pressure_gradient__10m_u_component_of_wind': {'cbar_label': '$U_{10m}$ vs $-d(SLP)/dx$', 'vmax': 2, 'divide_by': -1e5, 'units': r'$\times 10^{5} \; ms^{-1} (\text{Pa}m^{-1})^{-1}$'},
                       }
da_grid = [[bjerknes_correlations[cs]['correlation'].transpose("latitude", "longitude"),
            bjerknes_correlations_ece[cs]['correlation'].transpose("latitude", "longitude") ] for cs, v in bjerknes_corr_lookup.items()]

vmax_vals = [1]*len(da_grid)
vmin_vals = [-1*v for v in vmax_vals]
titles_grid = [[f'{string.ascii_lowercase[2*n]}) ACE2-NEMO-control', f'{string.ascii_lowercase[2*n+1]}) ECE3P-control'] for n in range(len(da_grid))]
cmaps = ['RdBu_r']*len(da_grid)
cbar_labels = [f"Correlation: {v['cbar_label']}" for v in bjerknes_corr_lookup.values()]


fig, axs = plot_map_grid_cbar_by_row(da_grid,
                                cbar_labels,
                                titles_grid ,
                                vmax_vals,
                                vmin_vals,
                                  projection=ccrs.PlateCarree(central_longitude=180),
                                  cmaps=cmaps,
                                width_height_ratio = [7,3],
                                shrink_factor= 0.8,
                                wspace=0.001,
                                cbar_height_ratio=0.02,
                                     lat_ticks = [-10,0,10],
                                     lon_ticks = range(130,251,20)
                                )

# vmax_vals = [v['vmax'] for v in bjerknes_corr_lookup.values()]
# vmin_vals = [-1*v for v in vmax_vals]
# titles_grid = [[f'{string.ascii_lowercase[2*n]}) ACE2-NEMO-control', f'{string.ascii_lowercase[2*n+1]}) ECE3P-control'] for n in range(len(da_grid))]
# cmaps = ['RdBu_r']*len(da_grid)
# cbar_labels = [f"{v['cbar_label']} [{v['units']}]" for v in bjerknes_corr_lookup.values()]


# fig, axs = plot_map_grid_cbar_by_row(da_grid,
#                                 cbar_labels,
#                                 titles_grid ,
#                                 vmax_vals,
#                                 vmin_vals,
#                                   projection=ccrs.PlateCarree(central_longitude=180),
#                                   cmaps=cmaps,
#                                 width_height_ratio = [7,3],
#                                 shrink_factor= 0.8,
#                                 wspace=0.001,
#                                 cbar_height_ratio=0.02,
#                                      lat_ticks = [-10,0,10],
#                                      lon_ticks = range(130,251,20)
#                                 )

# plt.savefig(os.path.join(MANUSCRIPT_FIGURE_DIR, f'bjerknes_feedback.pdf'), format='pdf', bbox_inches='tight')

# %%
# anomalous zonal wind stress vs SST gradient
mpl.style.use('default')

# The first variable is the x-coordinate, second is the y coordinate
bjerknes_corr_lookup = {'sea_surface_temperature_gradient__10m_u_component_of_wind': {'cbar_label': '$U_{10m}$ vs $d(SST)/dx$', 'vmax': 2, 'divide_by': 1e7, 'units': r'$\times 10^{7} \; (K m^{-1})} m^{-1} s $' },
                       }

lat_vals = bjerknes_correlations[list(bjerknes_correlations.keys())[0]]['latitude']
lon_vals = bjerknes_correlations[list(bjerknes_correlations.keys())[0]]['longitude']
tmp_sea_mask = sea_mask.sel(latitude=lat_vals, longitude=lon_vals)

# da_grid = [[xr.where(tmp_sea_mask, bjerknes_correlations[cs]['slope'] / v['divide_by'], np.nan).transpose("latitude", "longitude"),
#             xr.where(tmp_sea_mask, bjerknes_correlations_ece[cs]['slope'] / v['divide_by'], np.nan).transpose("latitude", "longitude")] for cs, v in bjerknes_corr_lookup.items()]
da_grid = [[xr.where(tmp_sea_mask, bjerknes_correlations[cs]['slope']  / v['divide_by'], np.nan).transpose("latitude", "longitude"),
            xr.where(tmp_sea_mask, bjerknes_correlations_hist[cs]['slope']  / v['divide_by'], np.nan).transpose("latitude", "longitude"),
            xr.where(tmp_sea_mask, bjerknes_correlations_ece[cs]['slope'] / v['divide_by'], np.nan).transpose("latitude", "longitude"),
            xr.where(tmp_sea_mask, bjerknes_correlations_era5[cs]['slope'] / v['divide_by'], np.nan).transpose("latitude", "longitude")] for cs, v in bjerknes_corr_lookup.items()]

vmax_vals = [v['vmax'] for v in bjerknes_corr_lookup.values()]

cbar_labels = [f"Regression slope: {v['cbar_label']}" for v in bjerknes_corr_lookup.values()]



for lag_var1, lag_var2 in [('mean_surface_downward_short_wave_radiation_flux', 'sea_surface_temperature'),
                           ('mean_surface_heat_flux', 'sea_surface_temperature')]:# Regression of heat fluxes against SST anomaly


    with open(os.path.join(ace2_nemo_control_dir, f"lagged_correlations_max5_{lag_var1}_{lag_var2}.pkl"), 'rb') as ifh:
        ace2_nemo_control_lagged_corr = pickle.load(ifh)
        
    with open(os.path.join(ace2_nemo_hist_dir, f"lagged_correlations_max5_{lag_var1}_{lag_var2}.pkl"), 'rb') as ifh:
        ace2_nemo_hist_lagged_corr = pickle.load(ifh)
    
    with open(os.path.join(ece_control_dir, f'lagged_correlations_max5_{lag_var1}_{lag_var2}.pkl'), 'rb') as ifh:
        ece_control_lagged_corr = pickle.load(ifh)
        
    with open(os.path.join(era5_dir, f'lagged_correlations_max1_{lag_var1}_{lag_var2}.pkl'), 'rb') as ifh:
        era5_control_lagged_corr = pickle.load(ifh)
    
    
    lags = list(ace2_nemo_control_lagged_corr.keys())
    plot_lags= [0]
    stat= 'slope'
    da_grid.append([ace2_nemo_control_lagged_corr[0][stat].rename({'lat': 'latitude', 'lon': 'longitude'}), 
                    ace2_nemo_hist_lagged_corr[0][stat].rename({'lat': 'latitude', 'lon': 'longitude'}), 
                  ece_control_lagged_corr[0][stat].rename({'lat': 'latitude', 'lon': 'longitude'}),
                  era5_control_lagged_corr[0][stat].rename({'lat': 'latitude', 'lon': 'longitude'})])

    cbar_labels.append(f"Regression: {name_lookup[lag_var1]['abbrev']} vs {name_lookup[lag_var2]['abbrev']}" )
num_rows = len(da_grid)
num_cols = len(da_grid[0])

vmax_vals += [40,20]
# vmax_vals = [1, 15, 20]
# vmin_vals = [None]*len(da_grid)
vmin_vals = [-1*v for v in vmax_vals]
titles_grid = [[f'{string.ascii_lowercase[num_rows*n]}) ACE2-NEMO-control', 
                f'{string.ascii_lowercase[num_rows*n]}) ACE2-NEMO-hist', 
                f'{string.ascii_lowercase[num_rows*n+1]}) ECE3P-control',
                f'{string.ascii_lowercase[num_rows*n+2]}) ERA5'] for n in range(len(da_grid))]



da_grid = [[da_grid[row][col].sel(longitude=slice(130,251), latitude=slice(-10,10)) for col in range(num_cols)] for row in range(num_rows)]
cmaps = ['RdBu_r']*len(da_grid)

fig, axs = plot_map_grid_cbar_by_row(da_grid,
                                cbar_labels,
                                titles_grid ,
                                vmax_vals,
                                vmin_vals,
                                  projection=ccrs.PlateCarree(central_longitude=180),
                                  cmaps=cmaps,
                                width_height_ratio = [7,3],
                                shrink_factor= 0.8,
                                wspace=0.001,
                                cbar_height_ratio=0.02,
                                 lat_ticks = [-10,0,10],
                                 lon_ticks = range(130,251,20)
                                )


# %%
# anomalous zonal wind stress vs SST gradient
mpl.style.use('default')

# The first variable is the x-coordinate, second is the y coordinate
bjerknes_corr_lookup = {'sea_surface_temperature_gradient__10m_u_component_of_wind': {'cbar_label': r'$U_{10m}$ vs $d(SST)/dx$', 'vmax': 2, 'divide_by': 1e7, 'units': r'$[\\times 10^{7} \; (K m^{-1}) m^{-1}s]$' },
                       }

lat_vals = bjerknes_correlations[list(bjerknes_correlations.keys())[0]]['latitude']
lon_vals = bjerknes_correlations[list(bjerknes_correlations.keys())[0]]['longitude']
tmp_sea_mask = sea_mask.sel(latitude=lat_vals, longitude=lon_vals)

# da_grid = [[xr.where(tmp_sea_mask, bjerknes_correlations[cs]['slope'] / v['divide_by'], np.nan).transpose("latitude", "longitude"),
#             xr.where(tmp_sea_mask, bjerknes_correlations_ece[cs]['slope'] / v['divide_by'], np.nan).transpose("latitude", "longitude")] for cs, v in bjerknes_corr_lookup.items()]
da_grid = [[xr.where(tmp_sea_mask, bjerknes_correlations[cs]['slope']  / v['divide_by'], np.nan).transpose("latitude", "longitude"),
            xr.where(tmp_sea_mask, bjerknes_correlations_hist[cs]['slope']  / v['divide_by'], np.nan).transpose("latitude", "longitude"),
            xr.where(tmp_sea_mask, bjerknes_correlations_ece[cs]['slope'] / v['divide_by'], np.nan).transpose("latitude", "longitude"),
            xr.where(tmp_sea_mask, bjerknes_correlations_era5[cs]['slope'] / v['divide_by'], np.nan).transpose("latitude", "longitude")] for cs, v in bjerknes_corr_lookup.items()]

vmax_vals = [v['vmax'] for v in bjerknes_corr_lookup.values()]
    
cbar_labels = [f"Regression slope: " + v['cbar_label'] + f" $[\\times 10^{{{int(np.log10(v['divide_by']))}}}$" + r"$\;m^{2} s^{-1}  K^{-1} ]$" for v in bjerknes_corr_lookup.values()]



for lag_var1, lag_var2 in [('mean_surface_downward_short_wave_radiation_flux', 'sea_surface_temperature'),
                           ('mean_surface_heat_flux', 'sea_surface_temperature')]:# Regression of heat fluxes against SST anomaly


    with open(os.path.join(ace2_nemo_control_dir, f"lagged_correlations_max5_{lag_var1}_{lag_var2}.pkl"), 'rb') as ifh:
        ace2_nemo_control_lagged_corr = pickle.load(ifh)
        
    with open(os.path.join(ace2_nemo_hist_dir, f"lagged_correlations_max5_{lag_var1}_{lag_var2}.pkl"), 'rb') as ifh:
        ace2_nemo_hist_lagged_corr = pickle.load(ifh)
    
    with open(os.path.join(ece_control_dir, f'lagged_correlations_max5_{lag_var1}_{lag_var2}.pkl'), 'rb') as ifh:
        ece_control_lagged_corr = pickle.load(ifh)
        
    with open(os.path.join(era5_dir, f'lagged_correlations_max1_{lag_var1}_{lag_var2}.pkl'), 'rb') as ifh:
        era5_control_lagged_corr = pickle.load(ifh)
    
    
    lags = list(ace2_nemo_control_lagged_corr.keys())
    plot_lags= [0]
    stat= 'slope'
    da_grid.append([ace2_nemo_control_lagged_corr[0][stat].rename({'lat': 'latitude', 'lon': 'longitude'}), 
                    ace2_nemo_hist_lagged_corr[0][stat].rename({'lat': 'latitude', 'lon': 'longitude'}), 
                  ece_control_lagged_corr[0][stat].rename({'lat': 'latitude', 'lon': 'longitude'}),
                  era5_control_lagged_corr[0][stat].rename({'lat': 'latitude', 'lon': 'longitude'})])

    cbar_labels.append(f"Regression slope: {name_lookup[lag_var1]['abbrev']} vs {name_lookup[lag_var2]['abbrev']}" + r"$\; [Wm^{-2} K^{-1}]$" )
num_rows = len(da_grid)
num_cols = len(da_grid[0])

vmax_vals += [40,20]
# vmax_vals = [1, 15, 20]
# vmin_vals = [None]*len(da_grid)
vmin_vals = [-1*v for v in vmax_vals]

# Create a titles grid with the same dimensions as da_grid

titles = ['ACE2-NEMO-control', 'ACE2-NEMO-hist', 'ECE3P-control', 'ERA5']
titles_grid = [[f'{string.ascii_lowercase[m*len(da_grid)+n]}) {title}' for n in range(len(da_grid))] for m, title in enumerate(titles)]

da_grid = [[da_grid[row][col].sel(longitude=slice(130,251), latitude=slice(-10,10)) for col in range(num_cols)] for row in range(num_rows)]
cmaps = ['RdBu_r']*len(da_grid)

transposed_da_grid = list(map(list, zip(*da_grid)))
# transposed_titles_grid = list(map(list, zip(*titles_grid)))

fig, axs = plot_map_grid_cbar_by_column(transposed_da_grid,
                                cbar_labels,
                                titles_grid ,
                                vmax_vals,
                                vmin_vals,
                                  projection=ccrs.PlateCarree(central_longitude=180),
                                  cmaps=cmaps,
                                width_height_ratio = [10,3],
                                shrink_factor= 0.6,
                                wspace=0.001,
                                cbar_height_ratio=0.02,
                                 lat_ticks = [-10,0,10],
                                 lon_ticks = range(130,251,20)
                                )
plt.savefig(os.path.join(MANUSCRIPT_FIGURE_DIR, f'enso_feedback.pdf'), format='pdf', bbox_inches='tight')


# %%
# Plot of ACE2 forced with SST and CO2 forcings, to see if the Bjerknes feedback is stronger in the historical forced runs

mpl.style.use('default')

# The first variable is the x-coordinate, second is the y coordinate
bjerknes_corr_lookup = {'sea_surface_temperature_gradient__10m_u_component_of_wind': {'cbar_label': r'$U_{10m}$ vs $d(SST)/dx$', 'vmax': 2, 'divide_by': 1e7, 'units': r'$[\\times 10^{7} \; (K m^{-1}) m^{-1}s]$' },
                       }

lat_vals = bjerknes_correlations_ace2_forced_control[list(bjerknes_correlations_ace2_forced_control.keys())[0]]['latitude']
lon_vals = bjerknes_correlations_ace2_forced_control[list(bjerknes_correlations_ace2_forced_control.keys())[0]]['longitude']
tmp_sea_mask = sea_mask.sel(latitude=lat_vals, longitude=lon_vals, method='nearest')
tmp_sea_mask = tmp_sea_mask.assign_coords(latitude=np.float32(lat_vals), longitude=np.float32(lon_vals))

da_grid = [[xr.where(tmp_sea_mask, bjerknes_correlations_ace2_forced_control[cs]['correlation']  , np.nan).transpose("latitude", "longitude"),
            xr.where(tmp_sea_mask, bjerknes_correlations_ace2_forced_hist[cs]['correlation'] , np.nan).transpose("latitude", "longitude")] for cs, v in bjerknes_corr_lookup.items()]

# vmax_vals = [v['vmax'] for v in bjerknes_corr_lookup.values()]
vmax_vals = [1.0 for v in bjerknes_corr_lookup.values()]

cbar_labels = [f"Regression slope: " + v['cbar_label'] + f" $[\\times 10^{{{int(np.log10(v['divide_by']))}}}$" + r"$\;m^{2} s^{-1}  K^{-1} ]$" for v in bjerknes_corr_lookup.values()]

plot_maps_shared_colorbar(da_grid, 
                          cbar_labels[0],
                          [['','']],
                          vmax_vals[0], 
                          -1*vmax_vals[0],
                          projection=ccrs.PlateCarree(central_longitude=180),
                          width_height_ratio = [8,6],
                          shrink_factor=0.7, 
                          wspace=0.001,
                          cbar_height_ratio=0.02,
                          cmap='RdBu_r', 
                          mask=None,
                          lat_ticks=np.arange(-15,16,5),
                          lon_ticks=np.arange(120,251,20))


# %%

# %%

for lag_var1, lag_var2 in [('mean_surface_downward_short_wave_radiation_flux', 'sea_surface_temperature'),
                           ('mean_surface_heat_flux', 'sea_surface_temperature')]:# Regression of heat fluxes against SST anomaly


    with open(os.path.join(ace2_nemo_control_dir, f"lagged_correlations_max5_{lag_var1}_{lag_var2}.pkl"), 'rb') as ifh:
        ace2_nemo_control_lagged_corr = pickle.load(ifh)
    
    with open(os.path.join(ece_control_dir, f'lagged_correlations_max5_{lag_var1}_{lag_var2}.pkl'), 'rb') as ifh:
        ece_control_lagged_corr = pickle.load(ifh)
    
    
    lags = list(ace2_nemo_control_lagged_corr.keys())
    plot_lags= [0]
    stat= 'corr'
    da_grid = [[ace2_nemo_control_lagged_corr[l][stat].sel(lat=slice(-20,20)), 
                ece_control_lagged_corr[l][stat].sel(lat=slice(-20,20))] for l in plot_lags]
    nrows= len(plot_lags)
    ncols=2
    
    if stat == 'cov':
        cbar_label = 'Covariance [$m^{2}s^{-2}$]'
        vmin=-15
        vmax=15
    elif stat == 'corr':
        cbar_label = f"Correlation: {name_lookup[lag_var1]['abbrev']} vs {name_lookup[lag_var2]['abbrev']}"
        vmin=-1
        vmax=1
                   # 
    fig, axs = plot_maps_shared_colorbar(da_grid,  
                              cbar_label,
                              [[f'a) ACE2-NEMO-control', f'b) ECE3P-control'] for l in plot_lags],
                              vmin=vmin, 
                              vmax=vmax,
                              width_height_ratio = [8,3],
                              shrink_factor=0.7, 
                              projection = ccrs.Robinson(central_longitude=180),
                              wspace=0.001,
                              cbar_height_ratio=2.0,
                              cmap='RdBu_r', 
                              mask=None)
    for r in range(nrows):
        for c in range(ncols):
    
            axs[r][c].coastlines()

# %%
# Surface winds autocorrelation

with open(os.path.join(ace2_nemo_control_dir, "lagged_correlations_max5_10m_u_component_of_wind_10m_u_component_of_wind.pkl"), 'rb') as ifh:
    ace2_nemo_control_lagged_corr = pickle.load(ifh)

with open(os.path.join(ece_control_dir, 'lagged_correlations_max5_10m_u_component_of_wind_10m_u_component_of_wind.pkl'), 'rb') as ifh:
    ece_control_lagged_corr = pickle.load(ifh)

lags = list(ace2_nemo_control_lagged_corr.keys())
plot_lags= [1,2,3]
stat= 'corr'
da_grid = [[ace2_nemo_control_lagged_corr[l][stat].isel(member=0).sel(lat=slice(-30,30)), 
            ece_control_lagged_corr[l][stat].sel(lat=slice(-30,30))] for l in plot_lags]
nrows= len(plot_lags)
ncols=2

if stat == 'cov':
    cbar_label = 'Covariance [$m^{2}s^{-2}$]'
    vmin=-15
    vmax=15
elif stat == 'corr':
    cbar_label = 'Correlation'
    vmin=-1
    vmax=1
               # 
fig, axs = plot_maps_shared_colorbar(da_grid,  
                          cbar_label,
                          [[f'a) ACE2-NEMO-control lag={l}', f'b) ECE3P-control lag={l}'] for l in plot_lags],
                          vmin=vmin, 
                          vmax=vmax,
                          width_height_ratio = [8,2],
                          shrink_factor=0.7, 
                          projection = ccrs.Robinson(central_longitude=180),
                          wspace=0.001,
                          cbar_height_ratio=2.0,
                          cmap='RdBu_r', 
                          mask=None)
for r in range(nrows):
    for c in range(ncols):

        axs[r][c].coastlines()


# %%
from scipy.stats import siegelslopes

bjerknes_vars = [ ['sea_surface_temperature_gradient', 'total_precipitation_daily_gradient'],
                   ['total_precipitation_daily_gradient', 'surface_pressure_gradient' ]]
fig, axs = plt.subplots(1,3, figsize=(3*6,4))
for n, (var1, var2) in enumerate(bjerknes_vars):
    print('*'*20)
    
    x = zonal_gradient_ds[var1].isel(member=0)
    y = zonal_gradient_ds[var2].isel(member=0)

    # Robust regression
    res = siegelslopes(x[:800],y[:800])

    corr_ds = calculate_linear_relationship(x,y)
    print(corr_ds)
    axs[n].plot(x, x*corr_ds['slope'].item() + corr_ds['intercept'].item(), color='k')

    
    
    x_ece = zonal_gradient_ece_ds[var1]
    y_ece = zonal_gradient_ece_ds[var2]
    # Robust regression
    res_ece = siegelslopes(x_ece[:800],y_ece[:800])
    corr_ece_ds = calculate_linear_relationship(x_ece,y_ece)

    # axs[n].plot(x_ece, x_ece*res_ece.slope + res_ece.intercept, color='k', linestyle='--')
    axs[n].plot(x, x*corr_ece_ds['slope'].item() + corr_ece_ds['intercept'].item(), color='r')

    print(corr_ece_ds)
    
    axs[n].scatter(x.values, y.values, label='ACE2-NEMO-control', marker='o')
    axs[n].scatter(x_ece,y_ece, label='ACE2-NEMO-control', marker='+')
# bjerknes_correlations_ece[[f'{var1}__{var2}']['slope'].transpose("latitude", "longitude")

# %%
with open(os.path.join(ace2_nemo_hist_dir, f'enso_spectra_dict.pkl'), 'rb') as ifh:
    enso_spectra_dict = pickle.load(ifh)

with open(os.path.join(ece_hist_dir, f'enso_spectra_dict.pkl'), 'rb') as ifh:
   ece_enso_spectra_dict = pickle.load(ifh)


# %%
def calculate_en34_spectra(da, fs =12, scaling='density'):

    nperseg = np.min([40*12, len(da['time'].values)])
    
    nino34_series =  da.sortby('time')
    f, Pxx = signal.welch(nino34_series, fs=fs, nperseg=nperseg, detrend='linear', scaling=scaling)

    return f, Pxx


# %%
from scipy import signal

num_rows=1
num_cols=3
width_height_ratio = [8,4]
shrink_factor=0.7
wspace=0.001
cbar_height_ratio=0.02
cmap='RdBu_r'
mask=None

fig = plt.figure(constrained_layout=True, figsize=(shrink_factor*width_height_ratio[0]*2, shrink_factor*width_height_ratio[1]))



gs = gridspec.GridSpec(num_rows + 1, num_cols, figure=fig, 
                    width_ratios=[1]* num_cols,
                    height_ratios=[1] * num_rows + [0.02],
                       wspace=wspace) 

plot_axs = [[fig.add_subplot(gs[:, 0])] + [fig.add_subplot(gs[0, n+1], projection = ccrs.Robinson(central_longitude=180)) for n in range(2)]]



# Power spectral density
fs = 12
scaling='density'


# Load Kristian's ACE2-NEMO spectra
fs = np.loadtxt(os.path.join(ENSO_SPECTRA_DIR, 'frequencies.txt'))
fs_ar1 = np.loadtxt(os.path.join(ENSO_SPECTRA_DIR, 'frequencies_ar1.txt'))

ps_acenemo_hist = np.loadtxt(os.path.join(ENSO_SPECTRA_DIR, 'n34_power_acenemo_hist.txt'))
ps_acenemo_ctrl = np.loadtxt(os.path.join(ENSO_SPECTRA_DIR, 'n34_power_acenemo_ctrl.txt'))
ps_era5 = np.loadtxt(os.path.join(ENSO_SPECTRA_DIR, 'n34_power_era5.txt'))
ps_ece3 = np.loadtxt(os.path.join(ENSO_SPECTRA_DIR, 'n34_power_ece3.txt'))
ps_ar1 = np.loadtxt(os.path.join(ENSO_SPECTRA_DIR, 'power_ar1.txt'))

ps_ar1_acenemo_ctrl = np.loadtxt(os.path.join(ENSO_SPECTRA_DIR, 'power_ar1_acenemo_ctrl.txt'))
power_upper_ar1_acenemo = np.loadtxt(os.path.join(ENSO_SPECTRA_DIR, 'power_upper_ar1_acenemo_ctrl.txt'))
power_lower_ar1_acenemo = np.loadtxt(os.path.join(ENSO_SPECTRA_DIR, 'power_lower_ar1_acenemo_ctrl.txt'))


# enso_spectra_dict = {}
# P_arr = []
# P_arr_hist = []
# f_arr = []
# for m in range(3):
#     en34_da = xr.load_dataarray(os.path.join(ace2_nemo_control_dir, f'nino3_4_smoothed_m{m}.nc')).dropna(dim='time')
#     en34_hist_da = xr.load_dataarray(os.path.join(ace2_nemo_hist_dir, f'nino3_4_smoothed_m{m}.nc')).dropna(dim='time')
#     f, Pxx = calculate_en34_spectra(en34_da, fs=fs, scaling=scaling)
#     f_hist, Pxx_hist = calculate_en34_spectra(en34_hist_da, fs=fs, scaling=scaling)
#     P_arr.append(Pxx[1:])
#     P_arr_hist.append(Pxx_hist[1:])

plot_axs[0][0].plot(fs, ps_era5, 'k-', lw=2, label='ERA5')
plot_axs[0][0].plot(fs, ps_acenemo_ctrl, color=colormaps['tab10'].colors[0], lw=2, label='ACE2-NEMO-control')
plot_axs[0][0].plot(fs, ps_acenemo_hist, color=colormaps['tab10'].colors[0], linestyle='--', lw=2, label='ACE2-NEMO-hist')
plot_axs[0][0].plot(fs, ps_ece3, color=colormaps['tab10'].colors[3], lw=2, label='ECE3P-control')
plot_axs[0][0].plot(fs_ar1, ps_ar1_acenemo_ctrl, color=colormaps['Set1'].colors[-1], linestyle='--', label='Theoretical AR1')
plot_axs[0][0].fill_between(fs, power_upper_ar1_acenemo, power_lower_ar1_acenemo, color=colormaps['Set1'].colors[-1], alpha=0.3)

plot_axs[0][0].set_xlabel('Period (months)')
plot_axs[0][0].set_ylabel('Power/Hz')
ticks = [0.,1./60, 1/36., 1/24., 1/18., 1/12., 1/9.]
plot_axs[0][0].set_xticks(ticks)
plot_axs[0][0].set_xlim([0.0, np.max(fs)])
labels = plot_axs[0][0].get_xticks().tolist()
months = [int(round(1.0/x,1)) for x in labels if x > 0]
months = ['T'] + months
plot_axs[0][0].set_xticklabels(months)
plot_axs[0][0].legend()
# plot_axs[0][0].grid()
plot_axs[0][0].set_title('a) ENSO Power Spectrum Density')

# plot_axs[0][0].plot(f[1:], np.array(P_arr).mean(axis=0), label=f'ACE2-NEMO-control', color='b')
# plot_axs[0][0].plot( f[1:], np.array(P_arr_hist).mean(axis=0), label=f'ACE2-NEMO-hist', color='b', linestyle='--')

# Calcualte spectra for ERA5
# era5_nino34_series =  xr.load_dataarray(os.path.join(era5_dir, f'nino3_4_era5.nc')).dropna(dim='time').sortby('time').dropna(dim='time')
# # era5_nino34_detrended = signal.detrend(era5_nino34_series.values)
# f, Pxx = calculate_en34_spectra(era5_nino34_series, fs=fs, scaling=scaling)
# plot_axs[0][0].plot(f[1:], Pxx[1:] , label='ERA5', color='k')

# ece_nino34_series =  xr.load_dataarray(os.path.join(ece_control_dir, f'nino3_4_ece3.nc')).dropna(dim='time').sortby('time').dropna(dim='time')
# f, Pxx = calculate_en34_spectra(ece_nino34_series, fs=fs, scaling=scaling)
# plot_axs[0][0].plot( f[1:], Pxx[1:], label='ECE3P-control', color='r')

# # ece_nino34_series =  xr.load_dataarray(os.path.join(ece_hist_dir, f'nino3_4_ece3.nc')).dropna(dim='time').sortby('time').dropna(dim='time')
# # f, Pxx = calculate_en34_spectra(ece_nino34_series, fs=fs)
# # plot_axs[0][0].plot( f[1:], Pxx[1:], label='ECE3P-hist', color='r', linestyle='--')

# plot_axs[0][0].set_xlabel('Frequency [cycles per year]')
# plot_axs[0][0].set_ylabel(r'Power [$K^2 / (\text{cycles per year})$]')
# plot_axs[0][0].set_xscale('log')
# plot_axs[0][0].set_xlim([1/(10),None])
# # enso_spectra_dict['ERA5'] = {'period': 1 / f[1:], 'power': Pxx[1:]*f[1:], 'Pxx': Pxx, 'f': f}
# plot_axs[0][0].legend()
# plot_axs[0][0].set_title('a)')


####################################################
## Precip correlation
da_list = [xr.load_dataset(os.path.join(ace2_nemo_control_dir, 'nino3_4_stats_total_precipitation_daily_m0.nc'))['slope'], 
           xr.load_dataset(os.path.join(ece_control_dir, 'ece3_nino3_4_stats.nc'))['slope']]

title_list = ['b) ACE2-NEMO-control', 'c) ECE3P-control']

for n, da in enumerate(da_list):
    im = da.plot(ax=plot_axs[0][n+1],
                                vmax=4, vmin=-4, 
                                  cmap='RdBu_r', 
                                  add_colorbar=False, rasterized=True,
                                  transform=ccrs.PlateCarree())


    plot_axs[0][n+1].set_title(title_list[n])
    plot_axs[0][n+1].coastlines()


cbar_ax = fig.add_subplot(gs[1, 1:])
cbar = plt.colorbar(im, cax=cbar_ax, label='Daily Precipitation Regressed on Niño 3.4 [mm/day/K]', orientation='horizontal')
cbar.ax.tick_params(labelsize=10)
plt.savefig(os.path.join(MANUSCRIPT_FIGURE_DIR, f'enso.pdf'), format='pdf')
