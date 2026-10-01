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
from tqdm import tqdm
import numpy as np
import xarray as xr
import xesmf as xe

import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from pathlib import Path
from matplotlib import gridspec
import cartopy.crs as ccrs
# import xarray_regrid
HOME = Path(os.getcwd()).parents[1]
REPO_DIR = HOME.parents[0]
sys.path.append(str(HOME))

import cartopy.mpl.ticker as cticker
from notebook_utils.misc import is_notebook

# %%
rundir = '/home/ecme4254/scratch/run_dir/n3.6_ace2_1951_control_compressed_19510101-19610101_m0'
results_dir = '/home/ecme4254/hpcperm/model_runs/n3.6_ace2_1951_control_compressed_19510101-19610101_m0'
atmosphere_fullname = 'era5-calculated'

# %%
nemo_results_dir = os.path.join(results_dir, f'nemo_output_{atmosphere_fullname}')
nemo_output_files = glob(os.path.join(rundir, 'nemo_ocean_output_*.nc')) + glob(os.path.join(rundir, 'lim_output_icemod_*.nc'))

save_to_zarr = False

for file in nemo_output_files:
    ds = xr.open_dataset(file)
    ds = ds.astype(np.float32)
    
    if save_to_zarr:
        # Convert to zarr format
        
        zarr_fp = os.path.join(nemo_results_dir, file.split('/')[-1].replace('.nc', '.zarr'))
        ds.to_zarr(zarr_fp)
        ds.close()
    else:
        fp = os.path.join(nemo_results_dir, file.split('/')[-1])
        if compression_level != 0:
            comp = dict(zlib=True, complevel=compression_level)
            
            encoding = {var: comp for var in ds.data_vars if not var in ['time_bnds', 'time_centered_bounds', 'time_counter_bounds']}
            encoding['time_centered_bounds'] = ds['time_centered_bounds'].encoding
            ds.to_netcdf(fp, encoding=encoding)
            ds.close()
        else:
            ds.to_netcdf(fp)
            ds.close()

# %%
model_name = 'n3.6_ace2_1951_multiIceCatACE2_19510101-19560101_m0'
ml_model = 'ace2'
base_dir = f"/ec/res4/scratch/ecme4254/run_dir/{model_name}"
tmp_ocean_dir = os.path.join(base_dir, 'router')
plot_dir = os.path.join(base_dir, 'plots')

os.makedirs(plot_dir, exist_ok=True)

ERA5_DIR ='/ec/res4/scratch/ecme4254/era5'
ORAS5_DIR="/ec/res4/hpcperm/ecme4254/oras5"
dt = datetime.datetime(1951,1,1)

gcfps = sorted(glob(os.path.join(tmp_ocean_dir, f'atm2oce_*_{ml_model}_nemo.nc')))
all_dts = sorted([datetime.datetime.strptime(item.split('/')[-1].split('_')[1], '%Y%m%d-%H') for item in gcfps])

number_of_time_steps = len(all_dts)

# %%
clim_sst_da = xr.load_dataarray('/home/ecme4254/hpcperm/era5/climatology/mean_sea_surface_temperature_1979-01-01__2018-12-31.nc').sel(dayofyear=1).rename({'latitude': 'lat', 'longitude': 'lon'})
clim_sst_da = clim_sst_da.sel(lat=np.arange(-90,91,1), lon=np.arange(0,360))

# %%
# Collect flux data

latitude_vals = np.arange(-90,91,1)
longitude_vals = np.arange(0,360,1)


atmosphere_types = [ml_model]
atm2oce_dict = {k:[] for k in atmosphere_types}
oce2atm_dict = {k:[] for k in atmosphere_types}

for n, tmp_dt in enumerate(all_dts[:4]):

    # Load calculated fluxes
    for atmospheretype in atm2oce_dict.keys():
        tmp_atm2oce_ds = xr.load_dataset(os.path.join(tmp_ocean_dir, f"atm2oce_{tmp_dt.strftime('%Y%m%d-%H')}_{atmospheretype}_nemo.nc"))

        if 'time' not in tmp_atm2oce_ds.coords:
            tmp_atm2oce_ds = tmp_atm2oce_ds.expand_dims({'time': [tmp_dt]})
        else:
            tmp_atm2oce_ds = tmp_atm2oce_ds.assign_coords(time=[tmp_dt])
        
        atm2oce_dict[atmospheretype].append(tmp_atm2oce_ds)

        tmp_oce2atm_ds = xr.load_dataset(os.path.join(tmp_ocean_dir, f"oce2atm_{int((tmp_dt - min(all_dts)).total_seconds()/3600)}h_{atmospheretype}_nemo.nc"))
        if 'time' not in tmp_oce2atm_ds.coords:
            tmp_oce2atm_ds = tmp_oce2atm_ds.expand_dims({'time': [tmp_dt]})
        else:
            tmp_oce2atm_ds = tmp_oce2atm_ds.assign_coords(time=[tmp_dt])
        oce2atm_dict[atmospheretype].append(tmp_oce2atm_ds)

for atmospheretype in atm2oce_dict.keys():
    atm2oce_dict[atmospheretype] = xr.concat(atm2oce_dict[atmospheretype], dim='time', coords='minimal')
    oce2atm_dict[atmospheretype] = xr.concat(oce2atm_dict[atmospheretype], dim='time', coords='minimal')
ice_mask = (oce2atm_dict[atmosphere_types[0]].isel(time=1)['sea_ice_fraction'] >0)

# %%
# Collect ERA5 data
# Note that currently just keeping 12h and 0h since that is what we have collected

era5_vars = ['mean_surface_sensible_heat_flux', 
                  'mean_surface_latent_heat_flux', 
                  'mean_surface_net_long_wave_radiation_flux', 
                  'evaporation', 'instantaneous_eastward_turbulent_surface_stress', 
                  'instantaneous_northward_turbulent_surface_stress', 
                  'mean_surface_net_short_wave_radiation_flux',
            #     'sea_surface_temperature',
            # 'sea_ice_cover',
            #  'forecast_albedo'
            ]
era5_ds = []
era5_dts = all_dts
for n, tmp_dt in enumerate(era5_dts[:4]):

    tmp_era5_flux_ds =[]
    

    for era5_var in era5_vars:
        tmp_da = xr.load_dataarray(os.path.join(ERA5_DIR, 'surface', era5_var, f"era5_{era5_var}_{tmp_dt.strftime('%Y%m%d')}.nc")).sel(time=tmp_dt)
        tmp_da.name = era5_var
    
        if era5_var == 'evaporation':
            # Convert to kg/m^2/s
            tmp_da = tmp_da * 1000 / 3600
    
        if 'latitude' in tmp_da.coords:
    
            tmp_da = tmp_da.sel(latitude=latitude_vals, longitude=longitude_vals)
            
        tmp_era5_flux_ds.append(tmp_da)
    tmp_era5_flux_ds = xr.merge(tmp_era5_flux_ds)
    era5_ds.append(tmp_era5_flux_ds)

era5_ds = xr.concat(era5_ds, dim='time')

# %%
era5_ds['A_Qns_oce'] = era5_ds['mean_surface_sensible_heat_flux'] + era5_ds['mean_surface_latent_heat_flux'] + era5_ds['mean_surface_net_long_wave_radiation_flux']
era5_ds['A_Qns_ice']  = era5_ds['A_Qns_oce'] 
era5_ds['A_Evap_total'] = era5_ds['evaporation']
era5_ds['A_Evap_ice'] = era5_ds['A_Evap_total']
era5_ds['A_Qs_oce'] = era5_ds['mean_surface_net_short_wave_radiation_flux']
era5_ds['A_Qs_ice'] = era5_ds['A_Qs_oce']
era5_ds['A_TauX_oce'] = era5_ds['instantaneous_eastward_turbulent_surface_stress']
era5_ds['A_TauY_oce'] = era5_ds['instantaneous_northward_turbulent_surface_stress']
era5_ds['A_TauX_ice'] = era5_ds['A_TauX_oce']
era5_ds['A_TauY_ice'] = era5_ds['A_TauY_oce']
era5_ds['sensible_heat_flux_ice'] = era5_ds['mean_surface_sensible_heat_flux']
era5_ds['latent_heat_flux_ice'] = era5_ds['mean_surface_latent_heat_flux']
era5_ds['net_long_wave_radiation_flux_ice'] = era5_ds['mean_surface_net_long_wave_radiation_flux']

# # era5_ds = era5_ds.regrid.linear(sea_mask)

atm2oce_dict['era5'] = era5_ds

# %%
# era5_ds = era5_ds.rename({'sea_ice_cover': 'sea_ice_fraction', 
#                           'ice_temperature_layer_1': 'sea_ice_temperature',
#                          'forecast_albedo': 'ice_albedo'})



# %%
# Load ORAS5: note, they don't have 2010 data, so we're using 2015 to get an idea
oras5_ds = []

oras5_vars = {'all_levels': ['rotated_meridional_velocity',
              'rotated_zonal_velocity'],
              'single_level': ['sea_ice_meridional_velocity', 
                               'sea_ice_zonal_velocity',
                               'sea_ice_thickness']}
all_year_months = set([(2015, dt.month) for dt in all_dts])
for n, ym_tuple in enumerate(all_year_months):

    tmp_oras5_flux_ds =[]
    
    atmospheretype='mlatmosphere'
    
    for data_category, oras5_var_list in oras5_vars.items():
    
        for oras5_var in oras5_var_list:
            tmp_da = xr.load_dataarray(os.path.join(ORAS5_DIR, data_category, oras5_var, f"oras5_{oras5_var}_{ym_tuple[0]}{ym_tuple[1]:02d}.nc")).assign_coords({'time_counter': [ym_tuple[1]]})
            tmp_da.name = oras5_var
        
            
            if 'latitude' in tmp_da.coords:
        
                tmp_da = tmp_da.sel(latitude=latitude_vals, longitude=longitude_vals)
                
            tmp_oras5_flux_ds.append(tmp_da)
    tmp_oras5_flux_ds = xr.merge(tmp_oras5_flux_ds)
    oras5_ds.append(tmp_oras5_flux_ds)


# meridional: y
# zonal: x
# u: x
# v: y

# zonal: u
# meridional: v

oras5_ds = xr.concat(oras5_ds, dim='time_counter').rename({'time_counter': 'time', 
                                                'rotated_zonal_velocity': 'ocean_current_u', 
                                                'rotated_meridional_velocity': 'ocean_current_v', 
                                                'sea_ice_zonal_velocity': 'ice_velocity_u',
                                                'sea_ice_meridional_velocity': 'ice_velocity_v',
                                                          'nav_lat': 'latitude', 'nav_lon': 'longitude'})
oras5_ds = oras5_ds.sel(deptht=0.50576)
oce2atm_monthly_dict = {k: ds.resample({'time': 'MS'}).mean() for k, ds in oce2atm_dict.items()}
oce2atm_monthly_dict['oras5'] = oras5_ds

# %% [markdown]
# ## Investigate instantaneous fluxes

# %%
ace2_instant_flux_final_ds= xr.open_dataset(os.path.join(base_dir, 'router/atm2oce_19510228-12_ace2_nemo.nc'))
ace2_instant_flux_init_ds= xr.open_dataset(os.path.join(base_dir, 'router/atm2oce_19510102-12_ace2_nemo.nc'))


# %%
era5_instant_flux_final_ds = era5_ds.sel(time=ace2_instant_flux_final_ds['time'][0])
era5_instant_flux_init_ds = era5_ds.sel(time=ace2_instant_flux_init_ds['time'][0])

# %%


plot_vars = [
         'sensible_heat_flux_ice',
    'latent_heat_flux_ice',
    'net_long_wave_radiation_flux_ice',
         'mean_surface_sensible_heat_flux',
         'mean_surface_latent_heat_flux',
          'mean_surface_net_long_wave_radiation_flux',
          'mean_surface_net_short_wave_radiation_flux',
          'instantaneous_eastward_turbulent_surface_stress',
          'instantaneous_northward_turbulent_surface_stress',
         'evaporation']
nrows = len(plot_vars)
ncols=len(atm2oce_dict.keys())+1

                                       
for n, v in enumerate(plot_vars):

    # tmp_ice_mask = oce2atm_mean_dict['era5']['sea_ice_fraction'] > 0.1

    da_dict = {'ace2': xr.where(sea_mask, ace2_instant_flux_final_ds[v], np.nan),
               'era5': xr.where(sea_mask, era5_instant_flux_final_ds[v], np.nan)}
    if v.endswith('_ice'):
        da_dict = {k : xr.where(ice_mask, v, np.nan) for k,v in da_dict.items()}

    da_dict = {k: v.transpose('latitude', 'longitude') for k,v in da_dict.items()}
        
    # da_dict['GenCast - ERA5'] = da_dict['gencast'] - da_dict['era5']
    fig, axs = plot_grid_shared_axes(da_grid= [list(da_dict.values())], 
                              num_rows=1, 
                              num_cols=len(da_dict.keys()), 
                              titles_grid=[[item.replace('gencast', 'GenCast').replace('era5', 'ERA5') for item in da_dict.keys()]],
                              cbar_label=name_lookup.get(v,v),
                              vmin=ranges.get(v, [None])[0], 
                              vmax=ranges.get(v, [None,None])[1],
                              shrink_factor=0.7, 
                              central_longitude=180, 
                              cmap='RdBu_r', 
                              mask=None)

    # plt.savefig(os.path.join(plot_dir, f"flux_average_{v}_{all_dts[0].strftime('%Y%m%d')}-{all_dts[-1].strftime('%Y%m%d')}.pdf"))


# %% [markdown]
# ## Investigate average fluxes

# %%
def plot_grid_shared_axes(da_grid, 
                          num_rows, 
                          num_cols, 
                          cbar_label,
                          titles_grid,
                          vmax, 
                          vmin,
                          width_height_ratio = [8,6],
                          shrink_factor=0.7, 
                          central_longitude=180, 
                          wspace=0.001,
                          cbar_height_ratio=0.02,
                          cmap='RdBu_r', 
                          mask=None):
   
    fig = plt.figure(constrained_layout=True, figsize=(shrink_factor*width_height_ratio[0]*2, shrink_factor*width_height_ratio[1]))

    gs = gridspec.GridSpec(num_rows + 1, num_cols, figure=fig, 
                        width_ratios=[1]* num_cols,
                        height_ratios=[1] * num_rows + [0.02],
                           wspace=wspace) 
    plot_axs = [[fig.add_subplot(gs[m, n], projection = ccrs.PlateCarree(central_longitude=central_longitude)) for n in range(num_cols)] for m in range(num_rows)]


    for row in range(num_rows):
        for col in range(num_cols):
            
            plot_da = da_grid[row][col]
            if mask is not None:
                plot_da = xr.where(mask, plot_da, np.nan)
            im = plot_da.plot(ax=plot_axs[row][col], 
                              vmax=vmax, vmin=vmin, 
                              cmap=cmap, 
                              add_colorbar=False, rasterized=True,
                              transform=ccrs.PlateCarree())

            if row == num_rows - 1:
                plot_axs[row][col].set_xticks(np.arange(-180,181,60), crs=ccrs.PlateCarree())
                lon_formatter = cticker.LongitudeFormatter()
                plot_axs[row][col].xaxis.set_major_formatter(lon_formatter)
                plot_axs[row][col].set_xlabel('Longitude')

            if col == 0:
                plot_axs[row][col].set_yticks(np.arange(-90,91,30), crs=ccrs.PlateCarree())
                lat_formatter = cticker.LatitudeFormatter()
                plot_axs[row][col].yaxis.set_major_formatter(lat_formatter)
                plot_axs[row][col].set_ylabel('Latitude')

            plot_axs[row][col].set_title(titles_grid[row][col])

    cbar_ax = fig.add_subplot(gs[row+1, :])
    cbar = plt.colorbar(im, cax=cbar_ax, label=cbar_label, orientation='horizontal')
    cbar.ax.tick_params(labelsize=10)

    return fig, plot_axs

# %%

# %%
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
          'sensible_heat_flux_ice': [-600,600],
         'mean_surface_latent_heat_flux': [-800,800],
          'mean_surface_net_long_wave_radiation_flux': [-150,150],
          'mean_surface_net_short_wave_radiation_flux': [0,1000],
          'instantaneous_eastward_turbulent_surface_stress': [-1, 1],
          'instantaneous_northward_turbulent_surface_stress': [-1,1],
         'evaporation': [-0.0003, 0.0003],
           'A_Precip_liquid': [-0.00005, 0.00005], 'A_Precip_solid': [-0.00001, 0.00001],
          'A_Evap_ice': [-0.00005, 0.00005],
         'A_Evap_total': [-0.0001, 0.0001]}

name_lookup = {'A_Evap_total': 'Total evaporation (kg/m^2/s)',
              'A_Qns_ice': 'Non-solar heat flux (Ice, W m^-2)',
          'A_Qns_oce': 'Non-solar heat flux (Ocean, W m^-2)',
          'A_Qs_oce': 'Solar heat flux (Ocean, W m^-2)',
          'A_Qs_ice': 'Solar heat flux (Ice, W m^-2)',
          'A_TauX_oce': 'Momentum flux X (Ocean, N m^-2)',
          'A_TauX_ice':  'Momentum flux X (Ice, N m^-2)',
          'A_TauY_oce':  'Momentum flux Y (Ocean, N m^-2)',
          'A_TauY_ice':  'Momentum flux Y (Ice, N m^-2)',
        'A_Tau_oce':  'Momentum flux (Ocean, N m^-2)',
          'A_Tau_ice':  'Momentum flux (Ice)',
           'A_Precip_liquid': 'Liquid precipitation (kg/m^2/s)', 
               'A_Precip_solid': 'Solid precipitation (kg/m^2/s)',
          'A_Evap_ice': 'Evaporation over ice (kg/m^2/s)'}

# %%
plot_vars = ['A_Evap_total',
 'A_Evap_ice',
 # 'A_Precip_liquid',
 # 'A_Precip_solid',
 'A_Qns_ice',
 'A_Qns_oce',
 'A_Qs_oce',
 'A_Qs_ice',
 'A_Tau_ice',
 'A_Tau_oce',]
nrows = len(plot_vars)
ncols=len(atm2oce_dict.keys())+1


atm2oce_mean_dict = {k: v.mean('time') for k,v in atm2oce_dict.items()}
oce2atm_mean_dict = {k: v.mean('time') for k,v in oce2atm_dict.items()}

for k in atm2oce_mean_dict.keys():
    atm2oce_mean_dict[k]['A_Tau_oce'] = np.sqrt(atm2oce_mean_dict[k]['A_TauX_oce']**2 + atm2oce_mean_dict[k]['A_TauY_oce']**2)
    atm2oce_mean_dict[k]['A_Tau_ice'] = np.sqrt(atm2oce_mean_dict[k]['A_TauX_ice']**2 + atm2oce_mean_dict[k]['A_TauY_ice']**2)
                                                
for n, v in enumerate(plot_vars):

    # tmp_ice_mask = oce2atm_mean_dict['era5']['sea_ice_fraction'] > 0.1

    # da_dict = {atm_type: xr.where(sea_mask, atm_dict[v], np.nan) for atm_type, atm_dict in atm2oce_mean_dict.items()}
    da_dict = {atm_type: atm_dict[v] for atm_type, atm_dict in atm2oce_mean_dict.items()}

    # if v.endswith('_ice'):
    #     da_dict = {k : xr.where(ice_mask, v, np.nan) for k,v in da_dict.items()}

    da_dict = {k: v.transpose('latitude', 'longitude') for k,v in da_dict.items()}
        
    # da_dict['GenCast - ERA5'] = da_dict['gencast'] - da_dict['era5']
    fig, axs = plot_grid_shared_axes(da_grid= [list(da_dict.values())], 
                              num_rows=1, 
                              num_cols=len(da_dict.keys()), 
                              titles_grid=[[item.replace('gencast', 'GenCast').replace('era5', 'ERA5') for item in da_dict.keys()]],
                              cbar_label=name_lookup.get(v,v),
                              vmin=ranges.get(v, [None])[0], 
                              vmax=ranges.get(v, [None,None])[1],
                              shrink_factor=0.7, 
                              central_longitude=180, 
                              cmap='RdBu_r', 
                              mask=None)

    # plt.savefig(os.path.join(plot_dir, f"flux_average_{v}_{all_dts[0].strftime('%Y%m%d')}-{all_dts[-1].strftime('%Y%m%d')}.pdf"))

# %%


plot_vars = [
         'sensible_heat_flux_ice',
    'latent_heat_flux_ice',
    'net_long_wave_radiation_flux_ice',
         'mean_surface_sensible_heat_flux',
         'mean_surface_latent_heat_flux',
          'mean_surface_net_long_wave_radiation_flux',
          'mean_surface_net_short_wave_radiation_flux',
          'instantaneous_eastward_turbulent_surface_stress',
          'instantaneous_northward_turbulent_surface_stress',
         'evaporation']
nrows = len(plot_vars)
ncols=len(atm2oce_dict.keys())+1


atm2oce_mean_dict = {k: v.mean('time') for k,v in atm2oce_dict.items()}
oce2atm_mean_dict = {k: v.mean('time') for k,v in oce2atm_dict.items()}

for k in atm2oce_mean_dict.keys():
    atm2oce_mean_dict[k]['A_Tau_oce'] = np.sqrt(atm2oce_mean_dict[k]['A_TauX_oce']**2 + atm2oce_mean_dict[k]['A_TauY_oce']**2)
    atm2oce_mean_dict[k]['A_Tau_ice'] = np.sqrt(atm2oce_mean_dict[k]['A_TauX_ice']**2 + atm2oce_mean_dict[k]['A_TauY_ice']**2)
                                                
for n, v in enumerate(plot_vars):

    # tmp_ice_mask = oce2atm_mean_dict['era5']['sea_ice_fraction'] > 0.1

    # da_dict = {atm_type: xr.where(sea_mask, atm_dict[v], np.nan) for atm_type, atm_dict in atm2oce_mean_dict.items()}
    da_dict = {atm_type: atm_dict[v] for atm_type, atm_dict in atm2oce_mean_dict.items()}

    # if v.endswith('_ice'):
    #     da_dict = {k : xr.where(ice_mask, v, np.nan) for k,v in da_dict.items()}

    da_dict = {k: v.transpose('latitude', 'longitude') for k,v in da_dict.items()}
        
    # da_dict['GenCast - ERA5'] = da_dict['gencast'] - da_dict['era5']
    fig, axs = plot_grid_shared_axes(da_grid= [list(da_dict.values())], 
                              num_rows=1, 
                              num_cols=len(da_dict.keys()), 
                              titles_grid=[[item.replace('gencast', 'GenCast').replace('era5', 'ERA5') for item in da_dict.keys()]],
                              cbar_label=name_lookup.get(v,v),
                              vmin=ranges.get(v, [None])[0], 
                              vmax=ranges.get(v, [None,None])[1],
                              shrink_factor=0.7, 
                              central_longitude=180, 
                              cmap='RdBu_r', 
                              mask=None)

    # plt.savefig(os.path.join(plot_dir, f"flux_average_{v}_{all_dts[0].strftime('%Y%m%d')}-{all_dts[-1].strftime('%Y%m%d')}.pdf"))


# %% [markdown]
# ## Check ocean and atmosphere is consistent after restart

# %%
def plot_grid_shared_axes(da_grid, 
                          num_rows, 
                          num_cols, 
                          cbar_label,
                          titles_grid,
                          vmax, 
                          vmin,
                          width_height_ratio = [8,6],
                          shrink_factor=0.7, 
                          central_longitude=180, 
                          wspace=0.001,
                          cbar_height_ratio=0.02,
                          cmap='RdBu_r', 
                          mask=None):
   
    fig = plt.figure(constrained_layout=True, figsize=(shrink_factor*width_height_ratio[0]*2, shrink_factor*width_height_ratio[1]))

    gs = gridspec.GridSpec(num_rows + 1, num_cols, figure=fig, 
                        width_ratios=[1]* num_cols,
                        height_ratios=[1] * num_rows + [0.02],
                           wspace=wspace) 
    plot_axs = [[fig.add_subplot(gs[m, n], projection = ccrs.PlateCarree(central_longitude=central_longitude)) for n in range(num_cols)] for m in range(num_rows)]


    for row in range(num_rows):
        for col in range(num_cols):
            
            plot_da = da_grid[row][col]
            if mask is not None:
                plot_da = xr.where(mask, plot_da, np.nan)
            im = plot_da.plot(ax=plot_axs[row][col], 
                              vmax=vmax, vmin=vmin, 
                              cmap=cmap, 
                              add_colorbar=False, rasterized=True,
                              transform=ccrs.PlateCarree())

            if row == num_rows - 1:
                plot_axs[row][col].set_xticks(np.arange(-180,181,60), crs=ccrs.PlateCarree())
                lon_formatter = cticker.LongitudeFormatter()
                plot_axs[row][col].xaxis.set_major_formatter(lon_formatter)
                plot_axs[row][col].set_xlabel('Longitude')

            if col == 0:
                plot_axs[row][col].set_yticks(np.arange(-90,91,30), crs=ccrs.PlateCarree())
                lat_formatter = cticker.LatitudeFormatter()
                plot_axs[row][col].yaxis.set_major_formatter(lat_formatter)
                plot_axs[row][col].set_ylabel('Latitude')

            plot_axs[row][col].set_title(titles_grid[row][col])

    cbar_ax = fig.add_subplot(gs[row+1, :])
    cbar = plt.colorbar(im, cax=cbar_ax, label=cbar_label, orientation='horizontal')
    cbar.ax.tick_params(labelsize=10)

    return fig, plot_axs

# %%
router_dir_1 = "/home/ecme4254/scratch/run_dir/n3.6_ace2_restart_test_19510101-19510201_m0/router"
router_dir_2 = "/home/ecme4254/scratch/run_dir/n3.6_ace2_restart_test_19510201-19510301_m0/router"
ocean_ds_1 = xr.open_dataset(os.path.join(router_dir_1, "oce2atm_738h_ace2_nemo.nc"))
ocean_ds_2 = xr.open_dataset(os.path.join(router_dir_2, "oce2atm_6h_ace2_nemo.nc"))

# %%
ocean_vars = ['sea_surface_temperature',
 'sea_ice_temperature',
 'sea_ice_fraction',
 'ice_albedo']

for n, v in enumerate(ocean_vars):

    da_dict = {'before': ocean_ds_1[v].isel(time=0).transpose('latitude', 'longitude'), 'after': ocean_ds_2[v].isel(time=0).transpose('latitude', 'longitude')}


    fig, axs = plot_grid_shared_axes(da_grid= [list(da_dict.values())], 
                              num_rows=1, 
                              num_cols=len(da_dict.keys()), 
                              titles_grid=[[f"{item.replace('gencast', 'GenCast').replace('era5-uncoupled', 'ERA5')}" for item in da_dict.keys()]],
                              cbar_label="",
                              vmin=None, 
                              vmax=None,
                              shrink_factor=0.7, 
                              central_longitude=180, 
                              cmap='RdBu_r', 
                              width_height_ratio=[8,6],
                              mask=None)


# %% [markdown]
# ## Investigate the change in ocean surface 

# %%
atm2oce_monthly_dict = {k: v.resample(time='MS').mean() for k,v in atm2oce_dict.items()}
oce2atm_monthly_dict = {k: v.resample(time='MS').mean() for k,v in oce2atm_dict.items()}

month_vals = sorted([pd.Timestamp(dt) for dt in oce2atm_monthly_dict[atmosphere_types[0]]['time'].values])

# %%
# Save monthly data 

# %%
oce2atm_monthly_dict['era5-uncoupled'] = era5_ds.resample(time='MS').mean()

# %%
ocean_vars = ['sea_surface_temperature',
 'sea_ice_temperature',
 'sea_ice_fraction',
 'ice_albedo']

range_lookup = {'sea_surface_temperature': dict(vmin=-5, vmax=5),
               'sea_ice_temperature': dict(vmin=-25, vmax=25),
                'sea_ice_fraction': dict(vmin=-1, vmax=1),
               'ocean_current_u': dict(vmin=-1, vmax=1),
             'ocean_current_v': dict(vmin=-1, vmax=1),
               'ice_velocity_u': dict(vmin=-0.2, vmax=0.2),
             'ice_velocity_v': dict(vmin=0.2, vmax=-0.2),
               'sea_ice_thickness': dict(vmin=-15, vmax=15),
               'ice_albedo': dict(vmin=-0.6, vmax=0.6)}

name_lookup = {'sea_surface_temperature': 'Sea Surface Temperature (K)',
 'sea_ice_temperature': 'Sea Ice Temperature (K)',
 'sea_ice_fraction': 'Sea Ice Fraction',
 'ice_albedo': 'Ice Albedo'}
for n, v in enumerate(ocean_vars):
    da_dict = {atm_type: xr.where(sea_mask, oce_dict[v], np.nan).sel(time=month_vals[-1]) -  xr.where(sea_mask, oce_dict[v], np.nan).sel(time=month_vals[0]) for atm_type, oce_dict in oce2atm_monthly_dict.items()}
    da_dict = {k:v for k,v in da_dict.items() if k in ['gencast', 'era5-uncoupled']}
    if v.endswith('_ice'):
        da_dict = {k : xr.where(tmp_ice_mask, v, np.nan) for k,v in da_dict.items()}
    
    da_dict = {k: v.transpose('latitude', 'longitude') for k,v in da_dict.items()}

    # da_dict['GenCast-ERA5'] = da_dict['gencast'] - da_dict['era5']
        
    fig, axs = plot_grid_shared_axes(da_grid= [list(da_dict.values())], 
                              num_rows=1, 
                              num_cols=len(da_dict.keys()), 
                              titles_grid=[[f"{item.replace('gencast', 'GenCast').replace('era5-uncoupled', 'ERA5')}" for item in da_dict.keys()] + ['GenCast - ERA5']],
                              cbar_label=name_lookup.get(v,v),
                              vmin=range_lookup.get(v, {'vmin': None})['vmin'], 
                              vmax=range_lookup.get(v, {'vmax': None})['vmax'],
                              shrink_factor=0.7, 
                              central_longitude=180, 
                              cmap='RdBu_r', 
                              width_height_ratio=[8,6],
                              mask=None)

    
    plt.savefig(os.path.join(plot_dir, f"ocean_change_{v}_{month_vals[0].strftime('%Y%m')}-{month_vals[-1].strftime('%Y%m')}.pdf"))

# %%
if not is_notebook():
    raise SystemExit("End of non-notebook code!")

# %% [markdown]
# ## Compare ERA5 calculated over ocean with ERA5 actual fluxes

# %%
n=1

era5_flux_vars = ['mean_surface_sensible_heat_flux', 
                  'mean_surface_latent_heat_flux', 
                  'mean_surface_net_long_wave_radiation_flux', 
                  'evaporation', 'instantaneous_eastward_turbulent_surface_stress', 'instantaneous_northward_turbulent_surface_stress', 'mean_surface_net_short_wave_radiation_flux']
sea_mask = ~np.isnan(xr.load_dataarray(os.path.join(ERA5_DIR, 'surface', 'sea_surface_temperature', f"era5_sea_surface_temperature_{dt.strftime('%Y%m%d')}.nc")).sel(time=dt)).sel(latitude=latitude_vals, longitude=longitude_vals)

ranges = {'mean_surface_sensible_heat_flux': [-600,600],
         'mean_surface_latent_heat_flux': [-800,800],
          'mean_surface_net_long_wave_radiation_flux': [-150,150],
          'mean_surface_net_short_wave_radiation_flux': [0,1000],
          'instantaneous_eastward_turbulent_surface_stress': [-1, 1],
          'instantaneous_northward_turbulent_surface_stress': [-1,1],
         'evaporation': [-0.0003, 0.0003]}
 
fig, ax = plt.subplots(len(era5_flux_vars),4, figsize=(8*4, len(era5_flux_vars)*6))

for n, v in enumerate(era5_flux_vars):

    for ix, atmospheretype in enumerate(['gencast', 'era5']):
        xr.where(sea_mask, atm2oce_dict[atmospheretype][v], np.nan).plot(ax=ax[n,ix], cmap='coolwarm', x='longitude', y='latitude',)
                                                      # vmin=ranges[v][0], vmax=ranges[v][1])

        ax[n,ix].set_title(f"{atmospheretype} {v} \n (max={atm2oce_dict[atmospheretype][v].max().item():0.2f}, min={atm2oce_dict[atmospheretype][v].min().item():0.2f})")

    # Load ERA5 SST
    era5_sst_da = xr.load_dataarray(os.path.join(ERA5_DIR, 'surface', 'sea_surface_temperature', f"era5_sea_surface_temperature_{tmp_dt.strftime('%Y%m%d')}.nc")).sel(time=tmp_dt).sel(latitude=latitude_vals, longitude=longitude_vals)

    era5_sst_da.plot(ax=ax[n,2])

    # Load NEMO SST diff
    (oce2atm_ds['sea_surface_temperature'] - era5_sst_da).plot(ax=ax[n,3], cmap='RdBu_r', x='longitude', y='latitude')

    

# %%
# Specific Latent heat of vaporization: roughly 2264.705 kJ/kg, or 2.3e6 J/kg
# Latent heat is in W/m2 = J/(s m^2)
# evaporation is in m
# Total latent heat transferred per unit area = t * latent heat = 3600*latent heat (J / m^2)
# Weight of air per m^2 for a given column of height h = density*h
# t * latent heat / latent heat of vaporization  = (3600*latent heat ) / 2.3e6 J/kg  (kg/m^2)
# Divide by density of water to volume of water per metre^2, or metres equivalent (3600*latent heat ) / (2.3e6 * 1000) m

# %%
# Compare fluxes between different atmosphere types, at one point in time
ranges = {'A_Qns_ice': [-800,800],
          'A_Qns_oce': [-800,800],
          'A_Qs_oce': [0,1000],
          'A_Qs_ice': [0,1000],
          'A_TauX_oce':  [-1, 1],
          'A_TauX_ice':  [-1, 1],
          'A_TauY_oce':  [-1, 1],
          'A_TauY_ice':  [-1, 1],
         {'mean_surface_sensible_heat_flux': [-600,600],
         'mean_surface_latent_heat_flux': [-800,800],
          'mean_surface_net_long_wave_radiation_flux': [-150,150],
          'mean_surface_net_short_wave_radiation_flux': [0,1000],
          'instantaneous_eastward_turbulent_surface_stress': [-1, 1],
          'instantaneous_northward_turbulent_surface_stress': [-1,1],
         'evaporation': [-0.0003, 0.0003],
           'A_Precip_liquid': [-0.0003, 0.0003], 'A_Precip_solid': [-0.0003, 0.0003],
          'A_Evap_ice': [-0.0003, 0.0003],
         'A_Evap_total': [-0.0003, 0.0003]}

oasis_data_vars = ['A_Evap_total',
 'A_Evap_ice',
 'A_Precip_liquid',
 'A_Precip_solid',
 'A_Qns_ice',
 'A_Qns_oce',
 'A_Qs_oce',
 'A_Qs_ice',
 'A_TauX_ice',
 'A_TauX_oce',
 'A_TauY_ice',
 'A_TauY_oce',
 'A_dQns_dT']
nrows = len(oasis_data_vars)
fig, ax = plt.subplots(nrows, 3, figsize=(8*3, nrows*6))

for n, v in enumerate(oasis_data_vars):

    for ix, atmospheretype in enumerate(['mlatmosphere', 'era5-calculated', 'era5']):
        xr.where(sea_mask, atm2oce_dict[atmospheretype].isel(time=1)[v], np.nan).plot(ax=ax[n,ix], cmap='coolwarm', x='longitude', y='latitude',
                                                      vmin=ranges.get(v, [None])[0], vmax=ranges.get(v, [None,None])[1])
        ax[n,ix].set_title(f"{atmospheretype} {v}")
    # xr.where(sea_mask, atm2oce_dict['era5-calculated_debug'][v], np.nan).plot(ax=ax[n,1], cmap='coolwarm', x='longitude', y='latitude',
    #                                               vmin=ranges[v][0], vmax=ranges[v][1])
    # ax[n,ix].set_title(f"{atmospheretype} {var} \n (max={atm2oce_dict[atmospheretype][var].max().item():0.2f}, min={atm2oce_dict[atmospheretype][var].min().item():0.2f})")

# %%
n=1
print(dt + datetime.timedelta(hours=n*6))
atm2oce_dict = {}
for atmospheretype in ['era5-calculated', 'era5']:
    atm2oce_dict[atmospheretype] = xr.load_dataset(os.path.join(tmp_ocean_dir, f"atm2oce_{(dt + datetime.timedelta(hours=n*12)).strftime('%Y%m%d-%H')}_{atmospheretype}.nc"))

for var in atm2oce_dict['era5'].data_vars:
    fig, ax = plt.subplots(1,3, figsize=(3*8,6))

    standard_dev = atm2oce_dict['era5'][var].std().item()
    mean = atm2oce_dict['era5'][var].mean().item()
    for ix, atmospheretype in enumerate(atm2oce_dict.keys()):
        
        atm2oce_dict[atmospheretype][var].plot(ax=ax[ix], vmax=mean+2*standard_dev, vmin=mean-2*standard_dev)
        ax[ix].set_title(f"{atmospheretype} {var} \n (max={atm2oce_dict[atmospheretype][var].max().item():0.2f}, min={atm2oce_dict[atmospheretype][var].min().item():0.2f})")

    (atm2oce_dict['era5-calculated'][var] - atm2oce_dict['era5'][var]).plot(ax=ax[2])

# %%
fig, ax = plt.subplots(1,3, figsize=(8*3, 6))

xr.where(np.logical_and(sea_mask, ~ice_mask), atm2oce_ds['A_Evap_total'], np.nan).plot(ax=ax[0], cmap='coolwarm', x='longitude', y='latitude',
                                                  vmin=-0.0003, vmax=0.0003)
xr.where(np.logical_and(sea_mask, ~ice_mask), era5_flux_ds['evaporation'], np.nan).plot(ax=ax[1], cmap='coolwarm', x='longitude', y='latitude',
                                              vmin=-0.0003, vmax=0.0003)



# %% [markdown]
# ## Explore how SST is changing with heating

# %%
dt = datetime.datetime(2010,1,1)
for n in range(16):
    ncols = 2
    
    oce2atm_ds = xr.load_dataset(os.path.join(tmp_ocean_dir, f"oce2atm_{(dt + datetime.timedelta(hours=n*6)).strftime('%Y%m%d-%H')}.nc"))
    atm2oce_ds = xr.load_dataset(os.path.join(tmp_ocean_dir, f"atm2oce_{(dt + datetime.timedelta(hours=n*6)).strftime('%Y%m%d-%H')}.nc"))

    if oce2atm_ds['longitude'].min().item() < 0:
        oce2atm_ds['longitude'] = (oce2atm_ds['longitude'] + 360)%360
        oce2atm_ds = oce2atm_ds.sortby(oce2atm_ds.longitude)
    if atm2oce_ds['longitude'].min().item() < 0:
        atm2oce_ds['longitude'] = (atm2oce_ds['longitude'] + 360)%360
        atm2oce_ds = atm2oce_ds.sortby(atm2oce_ds.longitude)
    if n ==1:
        sst_t0 = oce2atm_ds['sea_surface_temperature']
    
    fig, ax = plt.subplots(1,ncols, figsize=(ncols*4,3))

    if n > 1:
        (oce2atm_ds['sea_surface_temperature'] - sst_t0).plot.imshow(ax=ax[0], x='longitude', y='latitude')
    else:
        oce2atm_ds['sea_surface_temperature'].plot.imshow(ax=ax[0], x='longitude', y='latitude')
    atm2oce_ds['A_Qs_oce'].plot(ax=ax[1], x='longitude', y='latitude')


# %%
def sensible_heat_flux_over_ice(atmosphere_ds, ice_ds):
    # Sensible heat flux; note that it is defined as positive when heat is transferred from the air to the ice
    sensible_heat_flux = air_density * specific_heat_capacity_air * C_ice * atmosphere_ds['relative_wind_speed_ice'] * (atmosphere_ds['skin_temperature'] - ice_ds['sea_ice_temperature'])
    sensible_heat_flux.name = 'sensible_heat_flux_ice'
    return sensible_heat_flux


# %% [markdown]
# ## Compare ERA5 calculated over ice with ERA5 actual fluxes

# %%
paired_variables = [['latent_heat_flux_ice', 'mean_surface_latent_heat_flux', [-150,100]], 
                    ['sensible_heat_flux_ice', 'mean_surface_sensible_heat_flux', [-300, 300]],
                    ['net_long_wave_radiation_flux_ice', 'mean_surface_net_long_wave_radiation_flux', [-150,150]],
                    ['A_Qns_ice', 'A_Qns_oce', [-400,400]]]

# %%
# Compare data when ERA5 is given for all inputs

oce2atm_ds = xr.load_dataset(os.path.join(tmp_ocean_dir, f"oce2atm_{(dt + datetime.timedelta(hours=6)).strftime('%Y%m%d-%H')}_debug.nc"))
atm2oce_ds = xr.load_dataset(os.path.join(tmp_ocean_dir, f"atm2oce_{(dt + datetime.timedelta(hours=6)).strftime('%Y%m%d-%H')}_debug.nc"))
ice_mask = oce2atm_ds['sea_ice_fraction'] > 0.1

fig, ax = plt.subplots(len(paired_variables),2, figsize=(8*2, len(paired_variables)*6))

for n, pv in enumerate(paired_variables):
    xr.where(ice_mask, atm2oce_ds[pv[0]], np.nan).plot(ax=ax[n,0], cmap='coolwarm', x='longitude', y='latitude', vmin=pv[2][0], vmax=pv[2][1])
    xr.where(ice_mask, atm2oce_ds[pv[1]], np.nan).plot(ax=ax[n,1], cmap='coolwarm', x='longitude', y='latitude', vmin=pv[2][0], vmax=pv[2][1])

    ax[n,0].set_title(pv[0])
    ax[n,1].set_title(pv[1])

# %%

# %%
dt = datetime.datetime(2010,1,1)
for n in range(8):
    ncols = 6
    
    oce2atm_ds = xr.load_dataset(os.path.join(tmp_ocean_dir, f"oce2atm_{(dt + datetime.timedelta(hours=n*6)).strftime('%Y%m%d-%H')}.nc"))
    atm2oce_ds = xr.load_dataset(os.path.join(tmp_ocean_dir, f"atm2oce_{(dt + datetime.timedelta(hours=n*6)).strftime('%Y%m%d-%H')}.nc"))

    if oce2atm_ds['longitude'].min().item() < 0:
        oce2atm_ds['longitude'] = (oce2atm_ds['longitude'] + 360)%360
        oce2atm_ds = oce2atm_ds.sortby(oce2atm_ds.longitude)
    if atm2oce_ds['longitude'].min().item() < 0:
        atm2oce_ds['longitude'] = (atm2oce_ds['longitude'] + 360)%360
        atm2oce_ds = atm2oce_ds.sortby(atm2oce_ds.longitude)
    if n ==1:
        sst_t0 = oce2atm_ds['A_SST']

    
    fig, ax = plt.subplots(1,ncols, figsize=(ncols*8,6))
    if n > 0:
        if n > 1:
            (oce2atm_ds['A_SST'] - sst_t0).plot.imshow(ax=ax[0], x='longitude', y='latitude')
        else:
            oce2atm_ds['A_SST'].plot.imshow(ax=ax[0], x='longitude', y='latitude')
        atm2oce_ds['A_TauX_oce'].plot(ax=ax[1], vmin=-1, vmax=1, cmap='RdBu_r', x='longitude', y='latitude')
        atm2oce_ds['A_Qns_oce'].plot(ax=ax[2], x='longitude', y='latitude')
        atm2oce_ds['A_Qs_oce'].plot(ax=ax[3], x='longitude', y='latitude')
        atm2oce_ds['A_Precip_liquid'].plot(ax=ax[4], vmin=0, vmax=0.001, x='longitude', y='latitude')
        atm2oce_ds['A_Evap_total'].plot(ax=ax[5], x='longitude', y='latitude')
        # titles = [f'SST_diff (day = {n})', 'A_Qs_mix']
        # for n in range(len(titles)):
        #     ax[n].set_title(titles[n])

# %%
for var in ['A_Evap_ice',
 'A_Evap_total',
 'A_Precip_liquid',
 'A_Precip_solid',
 'A_Qns_ice',
 'A_Qns_oce',
 'A_Qs_mix',
 'A_TauX_ice',
 'A_TauX_oce',
 'A_TauY_ice',
 'A_TauY_oce',
 'A_dQns_dT']:
    fig, ax = plt.subplots(1,1, figsize=(8,6))
    oce2atm_ds = xr.load_dataset(os.path.join(tmp_ocean_dir, f"oce2atm_{(dt + datetime.timedelta(hours=4*24)).strftime('%Y%m%d-%H')}.nc"))
    atm2oce_ds = xr.load_dataset(os.path.join(tmp_ocean_dir, f"atm2oce_{(dt + datetime.timedelta(hours=4*24)).strftime('%Y%m%d-%H')}.nc"))

    xr.where(sea_mask, atm2oce_ds[var], np.nan).plot(ax=ax)
    ax.set_title(var)
