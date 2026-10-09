# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: ece4
#     language: python
#     name: python3
# ---

# %%
import sys, os
import numpy as np
import xarray as xr
import datetime as dt
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray_regrid
from tqdm import tqdm

sys.path.append('/home/a/antonio/repos/ace2_nemo_coupler')
from notebooks.plotting import plot_maps_shared_colorbar, plot_map_grid_cbar_by_row, name_lookup, plot_map_grid

BASE_DIR="/network/group/aopp/predict/HMC005_ANTONIO_EERIE/predictions/ace2_flux_comparison_1step"
FIGURES_DIR="/home/a/antonio/repos/ace2_nemo_coupler/notebooks/mamuscript_figures"
sea_mask = xr.load_dataarray("/network/group/aopp/predict/HMC005_ANTONIO_EERIE/ace2_data/era5_sea_mask_ACE2.nc")
grid = xr.load_dataarray("/network/group/aopp/predict/HMC005_ANTONIO_EERIE/ace2_data/grid.nc")

sea_mask = xr.align(sea_mask, grid, join='override')[0]

# %%
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

# %%
atm_types = ['ace2', 'ace2-calculated', 'era5', 'era5-calculated', 'era5-calculated-Rb']
atmospheric_ds_list_dict = {atm_type: [] for atm_type in atm_types}
oce_ds_list_dict = {atm_type: [] for atm_type in atm_types}

for year in tqdm(range(1951, 1956)):
    for month in range(1, 13):
        for hour in range(0, 24, 6):
            
            init_dt = dt.datetime(year, month, 1, hour)
            target_dt = init_dt + dt.timedelta(hours=6)
            
            # Atmospheric flux data
            for atm_type in atm_types:
                if atm_type == 'ace2':
                    suffix = 'ace2_6h.nc'
                elif atm_type == 'era5-calculated-Rb':
                    suffix = f"atm2oce_{target_dt.strftime('%Y%m%d-%H')}_era5-calculated_era5_debug.nc"
                else:
                    suffix = f"atm2oce_{target_dt.strftime('%Y%m%d-%H')}_{atm_type}_era5_debug.nc"
                
                tmp_ds = xr.load_dataset(f"{BASE_DIR}/{atm_type}/{init_dt.strftime('%Y%m%d-%H')}/{suffix}")
                
                if 'time' not in tmp_ds.coords:
                    tmp_ds = tmp_ds.expand_dims({"time": [target_dt]})
                else:
                    tmp_ds = tmp_ds.assign_coords({'time': [target_dt]})
                    
                for v in ['LHTFLsfc', 'SHTFLsfc']:
                    if v in tmp_ds:
                        tmp_ds[v] = -1 * tmp_ds[v]
                
                for v in ['init_time', 'valid_time']:
                    if v in tmp_ds:
                        tmp_ds = tmp_ds.drop_vars(v)
                                
                atmospheric_ds_list_dict[atm_type].append(tmp_ds)
            
            # Ocean data
            for atm_type in atm_types:
                if atm_type in ['ace2', 'ace2-calculated']:
                    suffix = f"oce2atm_6h_{atm_type}_era5_debug.nc"
                elif atm_type == 'era5-calculated-Rb':
                    suffix = f"oce2atm_{target_dt.strftime('%Y%m%d-%H')}_era5-calculated_era5_debug.nc"
                else:
                    suffix = f"oce2atm_{target_dt.strftime('%Y%m%d-%H')}_{atm_type}_era5_debug.nc"
                
                    
                tmp_ocean_ds = xr.load_dataset(f"{BASE_DIR}/{atm_type}/{init_dt.strftime('%Y%m%d-%H')}/{suffix}")
                
                oce_ds_list_dict[atm_type].append(tmp_ocean_ds)

# %%
atmospheric_ds_dict = {atm_type: xr.concat(ds_list, dim="time", data_vars='all', coords='minimal') for atm_type, ds_list in atmospheric_ds_list_dict.items()}  
oce_ds_dict = {atm_type: xr.concat(ds_list, dim="time", data_vars='all', coords='minimal') for atm_type, ds_list in oce_ds_list_dict.items()}  

oce_ds_dict = {atm_type: xr.align(ds, atmospheric_ds_dict[atm_type], join='override')[0] for atm_type, ds in oce_ds_dict.items()}
ice_mask_dict = {atm_type: xr.align(atmospheric_ds_dict[atm_type].isel(time=0), oce_ds_dict[atm_type]['sea_ice_fraction'].isel(time=0) > 0.1, join='override')[1] for atm_type in oce_ds_dict}

# Mask ice points; sensible heat flux is incorrect for AirSeaFluxCode, since it is using SST as input, not ice temperature. So we will mask out ice points for sensible heat flux when plotting AirSeaFluxCode results.
atmospheric_ds_dict_unmasked = atmospheric_ds_dict.copy()
atmospheric_ds_dict = {atm_type: xr.where(~ice_mask_dict[atm_type], ds, None) for atm_type, ds in atmospheric_ds_dict.items()}  

# %%
atmospheric_ds_dict['ace2'] = atmospheric_ds_dict['ace2'].rename({'PRATEsfc': 'total_precipitation', 
                        'LHTFLsfc': 'mean_surface_latent_heat_flux', 
                        'SHTFLsfc': 'mean_surface_sensible_heat_flux',
                        'DLWRFsfc': 'mean_surface_downward_long_wave_radiation_flux', 
                        'DSWRFsfc': 'mean_surface_downward_short_wave_radiation_flux',
                        'ULWRFsfc': 'mean_surface_upward_long_wave_radiation_flux', 
                        'USWRFsfc': 'mean_surface_upward_short_wave_radiation_flux',
                        'UGRD10m': '10m_u_component_of_wind',
                        'VGRD10m': '10m_v_component_of_wind',
                        'TMP2m': '2m_temperature',
                        'Q2m': 'specific_humidity_surface',
                        'PRESsfc': 'mean_sea_level_pressure'})

# %% [markdown]
# ## Compare the calculated ACE2 fluxes with the actual ACE2 fluxes

# %%
_, sea_mask = xr.align( atmospheric_ds_dict['ace2']['mean_surface_latent_heat_flux'].mean('time'), sea_mask, join='override')

plot_vars = [ 'mean_surface_latent_heat_flux', 'mean_surface_sensible_heat_flux']
da_grid = [ [xr.where(sea_mask, atmospheric_ds_dict['ace2'][varname].mean('time').transpose('latitude', 'longitude'), np.nan), 
             xr.where(sea_mask, atmospheric_ds_dict['ace2-calculated'][varname].mean('time').transpose('latitude', 'longitude'), np.nan),
             xr.where(sea_mask, (atmospheric_ds_dict['ace2-calculated'][varname] - atmospheric_ds_dict['ace2'][varname]).mean('time').transpose('latitude', 'longitude'), np.nan)] for varname in plot_vars]

cbar_labels= [[f"{name_lookup[varname]['name']} [{name_lookup[varname]['units']}]", f"Mean bias [{name_lookup[varname]['units']}]"] for varname in plot_vars]
titles_grid = [['a) ACE2', 'b) AirSeaFluxCode(ACE2)', 'c) AirSeaFluxCode(ACE2) - ACE2'], 
               ['d) ACE2', 'e) AirSeaFluxCode(ACE2)', 'f) AirSeaFluxCode(ACE2) - ACE2']]
vmax_vals = [np.array([300, 50]), np.array([75, 10])]
vmin_vals = [-1*arr for arr in vmax_vals]
cmaps = [['RdBu_r', 'RdBu_r'] for varname in plot_vars]

plot_map_grid(da_grid,
                                cbar_labels,
                                titles_grid ,
                                vmax_vals,
                                vmin_vals,
                                  projection=ccrs.Robinson(central_longitude=180),
                                  cmaps=cmaps,
                                  column_groups=[[0,1], [2]],
                                width_height_ratio = [5,4],
                                shrink_factor= 0.8,
                                wspace=0.001,
                                cbar_height_ratio=0.02,
                                )
plt.savefig(os.path.join(FIGURES_DIR, "ace2_vs_airseafluxcode_flux_comparison.pdf"),  bbox_inches='tight')

# %%
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
               'eastward_momentum_flux': {'name': 'Eastward momentum flux', 'units': '$kg m^{-1} s^{-2}$', 'abbrev': r'$\tau_{x}$'},
                'northward_momentum_flux': {'name': 'Northward momentum flux', 'units': '$kg m^{-1} s^{-2}$', 'abbrev': r'$\tau_{y}$'},
                'total_heat_flux': {'name': 'Total heat flux', 'units': '$Wm^{-2}$', 'abbrev': 'Total HF'},
               'total_heat_flux_oce': {'name': 'Total heat flux (ocean)', 'units': '$Wm^{-2}$', 'abbrev': 'Total HF (oce)'},
               'mean_surface_heat_flux': {'name': 'Latent + sensible heat flux', 'units': '$Wm^{-2}$', 'abbrev': 'LHF + SHF'},
               'evaporation': {'name': 'Evaporation', 'units': '$kg m^{-2} s^{-1}$', 'abbrev': 'Evaporation'},
}

# %%
for atm_type in ['era5', 'era5-calculated','era5-calculated-Rb','ace2-calculated']:
    for direction in ['eastward', 'northward']:
        flux_var = f"{direction}_momentum_flux"
        stress_var = f"instantaneous_{direction}_turbulent_surface_stress"
        if stress_var in atmospheric_ds_dict[atm_type]:
            atmospheric_ds_dict[atm_type][flux_var] = atmospheric_ds_dict[atm_type][stress_var].copy()

# %%
from matplotlib import colorbar, colors, gridspec

_, sea_mask = xr.align( atmospheric_ds_dict['era5']['mean_surface_latent_heat_flux'].mean('time'), sea_mask, join='override')

plot_vars = [ 'eastward_momentum_flux', 'northward_momentum_flux', 'evaporation']
da_grid = [ [xr.where(sea_mask, atmospheric_ds_dict['era5'][varname].mean('time').transpose('latitude', 'longitude'), np.nan), 
             xr.where(sea_mask, atmospheric_ds_dict['era5-calculated'][varname].mean('time').transpose('latitude', 'longitude'), np.nan),
             xr.where(sea_mask, atmospheric_ds_dict['era5-calculated'][varname].mean('time').transpose('latitude', 'longitude') - atmospheric_ds_dict['era5'][varname].mean('time').transpose('latitude', 'longitude'), np.nan)] for varname in plot_vars]

cbar_labels= [[f"{name_lookup[varname]['name']} [{name_lookup[varname]['units']}]", f"Mean bias [{name_lookup[varname]['units']}]"] for varname in plot_vars]
titles_grid = [['a) ERA5', 'b) AirSeaFluxCode(ERA5)', 'c) AirSeaFluxCode(ERA5) - ERA5'], 
               ['d) ERA5', 'e) AirSeaFluxCode(ERA5)', 'f) AirSeaFluxCode(ERA5) - ERA5'],
               ['g) ERA5', 'h) AirSeaFluxCode(ERA5)', 'i) AirSeaFluxCode(ERA5) - ERA5']]
vmax_vals = [np.array([0.5, 0.05]), np.array([0.2, 0.015]), np.array([10e-5, 2e-5])]
vmin_vals = [-1*arr for arr in vmax_vals]
cmaps = [['RdBu_r', 'RdBu_r'] for varname in plot_vars]

fig, plot_axs, colorbar_axes, row_images = plot_map_grid(da_grid,
                                cbar_labels,
                                titles_grid ,
                                vmax_vals,
                                vmin_vals,
                                  projection=ccrs.Robinson(central_longitude=180),
                                  cmaps=cmaps,
                                  column_groups=[[0,1], [2]],
                                width_height_ratio = [5,4],
                                  shrink_factor= 0.8,
                                  wspace=0.1,
                                  cbar_height_ratio=0.02,
                                  cbar_pad=0.1
)

# Change ticks of the last colorbars
colorbar_ax = colorbar_axes[-2][0]
plt.colorbar(
            row_images[-1][0],
            cax=colorbar_ax,
            label='Evaporation [$10^{-4} kg m^{-2} s^{-1}$]',
            orientation="horizontal",
        )

colorbar_ax.set_xticks([1e-4, 0.5e-4, 0, -0.5e-4, -1e-4])
colorbar_ax.set_xticklabels([1, 0.5, 0, -0.5, -1])

colorbar_ax = colorbar_axes[-1][0]
plt.colorbar(
            row_images[-1][-1],
            cax=colorbar_ax,
            label='Evaporation [$10^{-4} kg m^{-2} s^{-1}$]',
            orientation="horizontal",
        )

colorbar_ax.set_xticks([2e-5, 1e-5, 0, -1e-5, -2e-5])
colorbar_ax.set_xticklabels([0.2, 0.1, 0, -0.1, -0.2])

plt.savefig(os.path.join(FIGURES_DIR, "era5_vs_airseafluxcode_stress_comparison.pdf"), bbox_inches='tight')

# %%
from matplotlib import colorbar, colors, gridspec

_, sea_mask = xr.align( atmospheric_ds_dict['era5']['mean_surface_latent_heat_flux'].mean('time'), sea_mask, join='override')

plot_vars = [ 'eastward_momentum_flux', 'northward_momentum_flux', 'evaporation']
da_grid = [ [xr.where(sea_mask, atmospheric_ds_dict['era5'][varname].mean('time').transpose('latitude', 'longitude'), np.nan), 
             xr.where(sea_mask, atmospheric_ds_dict['era5-calculated-Rb'][varname].mean('time').transpose('latitude', 'longitude'), np.nan),
             xr.where(sea_mask, atmospheric_ds_dict['era5-calculated-Rb'][varname].mean('time').transpose('latitude', 'longitude') - atmospheric_ds_dict['era5'][varname].mean('time').transpose('latitude', 'longitude'), np.nan)] for varname in plot_vars]

cbar_labels= [[f"{name_lookup[varname]['name']} [{name_lookup[varname]['units']}]", f"Mean bias [{name_lookup[varname]['units']}]"] for varname in plot_vars]
titles_grid = [['a) ERA5', 'b) AirSeaFluxCode(ERA5)-Rb', 'c) AirSeaFluxCode(ERA5) - ERA5'], 
               ['d) ERA5', 'e) AirSeaFluxCode(ERA5)-Rb', 'f) AirSeaFluxCode(ERA5) - ERA5'],
               ['g) ERA5', 'h) AirSeaFluxCode(ERA5)-Rb', 'i) AirSeaFluxCode(ERA5) - ERA5']]
vmax_vals = [np.array([0.5, 0.05]), np.array([0.2, 0.015]), np.array([10e-5, 2e-5])]
vmin_vals = [-1*arr for arr in vmax_vals]
cmaps = [['RdBu_r', 'RdBu_r'] for varname in plot_vars]

fig, plot_axs, colorbar_axes, row_images = plot_map_grid(da_grid,
                                cbar_labels,
                                titles_grid ,
                                vmax_vals,
                                vmin_vals,
                                  projection=ccrs.Robinson(central_longitude=180),
                                  cmaps=cmaps,
                                  column_groups=[[0,1], [2]],
                                width_height_ratio = [5,4],
                                  shrink_factor= 0.8,
                                  wspace=0.1,
                                  cbar_height_ratio=0.02,
                                  cbar_pad=0.1
)

# Change ticks of the last colorbars
colorbar_ax = colorbar_axes[-2][0]
plt.colorbar(
            row_images[-1][0],
            cax=colorbar_ax,
            label='Evaporation [$10^{-4} kg m^{-2} s^{-1}$]',
            orientation="horizontal",
        )

colorbar_ax.set_xticks([1e-4, 0.5e-4, 0, -0.5e-4, -1e-4])
colorbar_ax.set_xticklabels([1, 0.5, 0, -0.5, -1])

colorbar_ax = colorbar_axes[-1][0]
plt.colorbar(
            row_images[-1][-1],
            cax=colorbar_ax,
            label='Evaporation [$10^{-4} kg m^{-2} s^{-1}$]',
            orientation="horizontal",
        )

colorbar_ax.set_xticks([2e-5, 1e-5, 0, -1e-5, -2e-5])
colorbar_ax.set_xticklabels([0.2, 0.1, 0, -0.1, -0.2])



# %%
a=1

# %%
from matplotlib import colorbar, colors, gridspec

_, sea_mask = xr.align( atmospheric_ds_dict['era5']['mean_surface_latent_heat_flux'].mean('time'), sea_mask, join='override')
_, sea_mask2 = xr.align( atmospheric_ds_dict['ace2-calculated']['mean_surface_latent_heat_flux'].mean('time'), sea_mask, join='override')

plot_vars = [ 'mean_surface_latent_heat_flux', 'mean_surface_sensible_heat_flux',  'eastward_momentum_flux', 'northward_momentum_flux', 'evaporation']

da_grid = []
for v in plot_vars:
    era5_da = xr.where(sea_mask, atmospheric_ds_dict['era5'][v].mean('time').transpose('latitude', 'longitude'), np.nan)
    ace2c_da = xr.where(sea_mask2, atmospheric_ds_dict['ace2-calculated'][v].mean('time').transpose('latitude', 'longitude'), np.nan)
    era5_da = xr.align(ace2c_da, era5_da, join='override')[1]
    da_grid.append([era5_da, ace2c_da, ace2c_da - xr.align(era5_da, ace2c_da, join='override')[0]])

cbar_labels= [[f"{name_lookup[varname]['name']} [{name_lookup[varname]['units']}]", f"Mean bias [{name_lookup[varname]['units']}]"] for varname in plot_vars]
titles_grid = [['a) ERA5', 'b) AirSeaFluxCode(ACE2)', 'c) AirSeaFluxCode(ACE2) - ERA5'], 
               ['d) ERA5', 'e) AirSeaFluxCode(ACE2)', 'f) AirSeaFluxCode(ACE2) - ERA5'],
               ['g) ERA5', 'h) AirSeaFluxCode(ACE2)', 'i) AirSeaFluxCode(ACE2) - ERA5'],
               ['j) ERA5', 'k) AirSeaFluxCode(ACE2)', 'l) AirSeaFluxCode(ACE2) - ERA5'],
               ['m) ERA5', 'n) AirSeaFluxCode(ACE2)', 'o) AirSeaFluxCode(ACE2) - ERA5']]

np.array([0.5, 0.05]), np.array([0.2, 0.015])
vmax_vals = [np.array([300, 50]), np.array([75, 10]), np.array([0.5, 0.05]), np.array([0.2, 0.015]), np.array([10e-5, 2e-5])]
vmin_vals = [-1*arr for arr in vmax_vals]
cmaps = [['RdBu_r', 'RdBu_r'] for varname in plot_vars]

fig, plot_axs, colorbar_axes, row_images = plot_map_grid(da_grid,
                                cbar_labels,
                                titles_grid ,
                                vmax_vals,
                                vmin_vals,
                                  projection=ccrs.Robinson(central_longitude=180),
                                  cmaps=cmaps,
                                  column_groups=[[0,1], [2]],
                                width_height_ratio = [5,4],
                                  shrink_factor= 0.8,
                                  wspace=0.1,
                                  cbar_height_ratio=0.02,
                                  cbar_pad=0.1
)

# Change ticks of the last colorbars
colorbar_ax = colorbar_axes[-2][0]
plt.colorbar(
            row_images[-1][0],
            cax=colorbar_ax,
            label='Evaporation [$10^{-4} kg m^{-2} s^{-1}$]',
            orientation="horizontal",
        )

colorbar_ax.set_xticks([1e-4, 0.5e-4, 0, -0.5e-4, -1e-4])
colorbar_ax.set_xticklabels([1, 0.5, 0, -0.5, -1])

colorbar_ax = colorbar_axes[-1][0]
plt.colorbar(
            row_images[-1][-1],
            cax=colorbar_ax,
            label='Evaporation [$10^{-4} kg m^{-2} s^{-1}$]',
            orientation="horizontal",
        )

colorbar_ax.set_xticks([2e-5, 1e-5, 0, -1e-5, -2e-5])
colorbar_ax.set_xticklabels([0.2, 0.1, 0, -0.1, -0.2])

plt.savefig(os.path.join(FIGURES_DIR, "era5_vs_airseafluxcode_ace2_flux_comparison.pdf"), bbox_inches='tight')

# %%
import sys, os
import numpy as np
import xarray as xr
import datetime as dt
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray_regrid
from tqdm import tqdm

sys.path.append('/home/a/antonio/repos/ace2_nemo_coupler')
from notebooks.plotting import plot_maps_shared_colorbar, plot_map_grid_cbar_by_row, name_lookup, plot_map_grid

BASE_DIR="/network/group/aopp/predict/HMC005_ANTONIO_EERIE/predictions/ace2_flux_comparison_1step"
FIGURES_DIR="/home/a/antonio/repos/ace2_nemo_coupler/notebooks/mamuscript_figures"
sea_mask = xr.load_dataarray("/network/group/aopp/predict/HMC005_ANTONIO_EERIE/ace2_data/era5_sea_mask_ACE2.nc")
grid = xr.load_dataarray("/network/group/aopp/predict/HMC005_ANTONIO_EERIE/ace2_data/grid.nc")

sea_mask = xr.align(sea_mask, grid, join='override')[0]

# %%

era5_snowfall = xr.load_dataarray("/network/group/aopp/predict/HMC005_ANTONIO_EERIE/era5/surface/era5_snowfall.nc").sortby('latitude', ascending=True).sortby('longitude', ascending=True)

# convert to kg/m2/s
era5_snowfall = era5_snowfall * 1000 / (3600 * 1)
era5_snowfall = era5_snowfall.regrid.linear(grid)
era5_snowfall = era5_snowfall.rename({'valid_time': 'time'})

# %%
_, sea_mask = xr.align( atmospheric_ds_dict['era5']['mean_surface_latent_heat_flux'].mean('time'), sea_mask, join='override')

plot_vars = [ 'solid_precipitation']
da_grid = [ [xr.where(sea_mask, era5_snowfall.mean('time').transpose('latitude', 'longitude'), np.nan), 
             xr.where(sea_mask, atmospheric_ds_dict_unmasked['era5-calculated']['solid_precipitation'].mean('time').transpose('latitude', 'longitude'), np.nan),
             xr.where(sea_mask, atmospheric_ds_dict_unmasked['era5-calculated']['solid_precipitation'].mean('time').transpose('latitude', 'longitude') - era5_snowfall.mean('time').transpose('latitude', 'longitude'), np.nan)]]

cbar_labels= [[f"Solid precipitation [$kg m^{{-2}} s^{{-1}}$]", f"Difference [$kg m^{{-2}} s^{{-1}}$]"] for varname in plot_vars]
titles_grid = [['a) ERA5', 'b) Heuristic', 'c) Heuristic - ERA5'], 
               ['d) ERA5', 'e) Heuristic', 'f) Heuristic - ERA5']]
# vmax_vals = [np.array([0.5, 0.05]), np.array([0.2, 0.02])]
# vmin_vals = [-1*arr for arr in vmax_vals]
vmax_vals = [[3e-5, 2e-5]]
vmin_vals = [[0,-2e-5]]
cmaps = [['Blues', 'RdBu_r'] for varname in plot_vars]

plot_map_grid(da_grid,
                                cbar_labels,
                                titles_grid ,
                                vmax_vals,
                                vmin_vals,
                                  projection=ccrs.Robinson(central_longitude=180),
                                  cmaps=cmaps,
                                  column_groups=[[0,1], [2]],
                                width_height_ratio = [5,4],
                                  shrink_factor= 0.8,
                                  wspace=0.1,
                                  cbar_height_ratio=0.02,
                                  cbar_pad=0.1
)

plt.savefig(os.path.join(FIGURES_DIR, "solid_precip_comparison.pdf"), bbox_inches='tight')

# %%
