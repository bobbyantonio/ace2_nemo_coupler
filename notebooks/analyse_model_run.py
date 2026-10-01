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
# import xesmf as xe
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from pathlib import Path
from matplotlib import gridspec
import cartopy.crs as ccrs
import calendar
from itertools import chain
import cartopy.mpl.ticker as cticker

sys.path.append('/home/ecme4254/perm/repos/nwp_notebooks')
from notebook_utils.misc import is_notebook
# from notebook_utils.plotting import plot_grid_shared_axes

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
               'A_Precip_solid': {'name':'Solid precipitation', 'units': 'kg/m^2/s'},
          'A_Evap_ice': {'name':'Evaporation over ice', 'units': 'kg/m^2/s'},
           'surface_temperature': {'name':'Surface Temperature', 'units': 'K'},
            'sea_surface_temperature': {'name': 'Sea surface temperature', 'units': 'K'},
              'sea_ice_fraction': {'name':'Sea Ice Fraction', 'units': '0-1'},
              'LHTFLsfc': {'name':'Latent heat flux', 'units': 'W/m^2'}, 
               'SHTFLsfc': {'name':'Sensible heat flux', 'units': 'W/m^2'}, 
               'DLWRFsfc': {'name':'LW flux down', 'units': 'W/m^2'}, 
               'ULWRFsfc': {'name':'LW flux up', 'units': 'W/m^2'},
               'DSWRFsfc': {'name':'SW flux down', 'units': 'W/m^2'},
               'USWRFsfc': {'name':'SW flux up', 'units': 'W/m^2'},
               'PRATEsfc': {'name':'Precipitation rate', 'units': 'kg/m^2/s'}}


# %%
if is_notebook():
    glob_str = '*'
else:
    glob_str = '*'

# %%
experiment_id = 'n3.6_ace2_1951_spinupCMIP6_19510101-20210101_m0'
base_dir = os.path.join('/home/ecme4254/hpcperm/model_runs/', experiment_id)
model_name = 'ace2'
atmosphere_types = ['ace2']

# %%
atm2oce_dict = {k:[] for k in atmosphere_types} 
atm2oce_daily_dict = {k:[] for k in atmosphere_types} 
for atmospheretype in atm2oce_dict.keys():
    atm2oce_dict[atmospheretype] = xr.open_mfdataset(glob(os.path.join(base_dir, f'atm2oce_MS_{atmospheretype}_nemo_{glob_str}.nc'))[:-1])
    # atm2oce_daily_dict[atmospheretype] = xr.open_mfdataset(glob(os.path.join(base_dir, f'atm2oce_D_{atmospheretype}_nemo_*.nc')))
    # atm2oce_dict[atmospheretype]['A_Tau_oce'] = np.sqrt(atm2oce_dict[atmospheretype]['A_TauX_oce']**2 + atm2oce_dict[atmospheretype]['A_TauY_oce']**2)
    # atm2oce_dict[atmospheretype]['A_Tau_ice'] = np.sqrt(atm2oce_dict[atmospheretype]['A_TauX_ice']**2 + atm2oce_dict[atmospheretype]['A_TauY_ice']**2)

    # atm2oce_dict[atmospheretype]['A_EmP'] = atm2oce_dict[atmospheretype]['A_Evap_total'] - (atm2oce_dict[atmospheretype]['A_Precip_liquid'] + atm2oce_dict[atmospheretype]['A_Precip_solid'])

# %%
oce2atm_dict = {k:[] for k in atmosphere_types}
oce2atm_daily_dict = {k:[] for k in atmosphere_types}
for atmospheretype in oce2atm_dict.keys():
    
    for fp in glob(os.path.join(base_dir, f'oce2atm_MS_{atmospheretype}_nemo_{glob_str}.nc')):
        tmp_ds = xr.load_dataset(fp)
        if 'time' not in tmp_ds.coords:
            continue
        else:
            oce2atm_dict[atmospheretype].append(tmp_ds)
    oce2atm_dict[atmospheretype] = xr.concat( oce2atm_dict[atmospheretype], dim='time')
    # oce2atm_dict[atmospheretype] = xr.open_mfdataset(glob(os.path.join(base_dir, f'oce2atm_MS_{atmospheretype}_nemo_{glob_str}.nc'))[:-1])
    # oce2atm_daily_dict[atmospheretype] = xr.open_mfdataset(glob(os.path.join(base_dir, f'oce2atm_D_{atmospheretype}_nemo_*.nc')), combine='nested')


# %%

# ace2_calc_monthly_ds = xr.load_dataset(os.path.join(f"/home/ecme4254/hpcperm/model_runs/n3.6_ace2-calculated_20100101-20100501_m0", f'{model_name}_MS.nc'))

# ace2_calc_basedir = base_dir.replace('ace2', 'ace2-calculated')
# for atmospheretype in ['ace2-calculated']:
#     atm2oce_dict[atmospheretype] = xr.load_dataset(os.path.join(ace2_calc_basedir, f'atm2oce_MS_{atmospheretype}_nemo.nc'))
#     oce2atm_dict[atmospheretype] = xr.load_dataset(os.path.join(ace2_calc_basedir, f'oce2atm_MS_{atmospheretype}_nemo.nc'))

#     atm2oce_dict[atmospheretype]['A_Tau_oce'] = np.sqrt(atm2oce_dict[atmospheretype]['A_TauX_oce']**2 + atm2oce_dict[atmospheretype]['A_TauY_oce']**2)
#     atm2oce_dict[atmospheretype]['A_Tau_ice'] = np.sqrt(atm2oce_dict[atmospheretype]['A_TauX_ice']**2 + atm2oce_dict[atmospheretype]['A_TauY_ice']**2)

# %%
sea_mask = ~np.isnan(oce2atm_dict[atmosphere_types[0]]['sea_surface_temperature'].isel(time=0))
ice_mask = oce2atm_dict[atmosphere_types[0]]['sea_ice_fraction'].mean('time') > 0.05

# %%
atmosphere_monthly_ds = xr.open_mfdataset(glob(os.path.join(base_dir, f'{model_name}_MS_{model_name}_nemo_{glob_str}.nc'))[:-1])

# %%
time_vals = [pd.Timestamp(dt) for dt in sorted(atm2oce_dict['ace2']['time'].values)]
month_limits = [ (dt, datetime.datetime(dt.year, dt.month, calendar.monthrange(dt.year, dt.month)[1])) for dt in time_vals]
expanded_time_vals = sorted(chain.from_iterable([ list(pd.date_range(ml[0], ml[1])) for ml in month_limits]))

# %%
# Gather ERA5 data for the same period
# ym_vals = [pd.Timestamp(dt).strftime('%Y%m') for dt in time_vals]
# years = sorted(set([pd.Timestamp(dt).strftime('%Y') for dt in time_vals]))
# era5_ds = []
# era5_vars = ['2m_temperature', 'sea_surface_temperature']
# for v in era5_vars:
#     era5_da = []
#     for y in years:
#         tmp_da = xr.open_dataarray(f'/home/ecme4254/scratch/era5_monthly/surface/2m_temperature/era5M_2m_temperature_year{y}.nc')
#         tmp_da.name = v
#         era5_da.append(tmp_da)
#     era5_da = xr.concat(era5_da, dim='time')

#     era5_ds.append(era5_da)
# era5_ds = xr.merge(era5_ds).rename({'latitude': 'lat', 'longitude':'lon'})
# era5_monthly_ds = xr.open_dataset('/home/ecme4254/hpcperm/era5_monthly/era5_monthly_1951-1952.nc').rename({'t2m': '2m_temperature'})

# %%
# Look at global temperature variation
# var = '2m_temperature'
# fig, ax = plt.subplots(1,1, figsize=(7,5))
# atmosphere_monthly_ds['TMP2m'].mean(['latitude', 'longitude']).plot(ax=ax, linestyle='--',label='ACE2-NEMO')
# era5_ds.isel(time=0)['2m_temperature'].mean(['lat', 'lon']).plot(ax=ax, label='ERA5')
# plt.legend()
# ax.set_title('')
# ax.set_ylabel('Global 2m Temperature')
# ax.set_xlabel('Time')

# plt.savefig('2mt_monthly_comparison_ace2.pdf', format='pdf')

# %%
for var in atm2oce_dict['ace2'].data_vars:
    fig, ax = plt.subplots(1,1, figsize=(8,5))

    for atm_type, ds in atm2oce_dict.items():

        da =  ds[var].sel(time=time_vals)

        if var.endswith('ice'):
            da = xr.where(ice_mask, da, np.nan)
        else:
            da = xr.where(~ice_mask, da, np.nan)
        mean_series = da.mean(['longitude', 'latitude'], skipna=True).values
        std_series = da.std(['longitude', 'latitude'], skipna=True).values
        ax.plot(time_vals, mean_series, label=atm_type)
        # ax.fill_between(atm2oce_dict['ace2']['time'], y1=mean_series+std_series,y2=mean_series-std_series )
        ax.set_title(var)
    plt.legend()

# %% [markdown]
# ## Line plots

# %%
# vars_to_plot = ['sea_surface_height', 'surface_temperature', 'LHTFLsfc', 'SHTFLsfc', 'DLWRFsfc', 'ULWRFsfc', 'DSWRFsfc', 'USWRFsfc', 'PRATEsfc', 'sea_surface_temperature', 'sea_ice_fraction']
vars_to_plot = ['sea_surface_height', 'surface_temperature',  'sea_surface_temperature', 'sea_ice_fraction']


# vars_to_plot = ['surface_temperature', 'A_Qns_oce', 'A_Qs_oce', 'A_Tau_oce', 'A_Precip_liquid', 'sea_surface_temperature', 'sea_ice_fraction']

# nrows = int(np.ceil(len(vars_to_plot)/2))
# ncols=2
nrows = len(vars_to_plot)
ncols=2
fig, axs = plt.subplots(nrows, ncols, figsize=(6*ncols, 4*nrows))
fig.tight_layout(pad=5)

for n, var in enumerate(vars_to_plot):

    # # row = int(n/ncols)
    # col = n%ncols
    row = n
    col=0
    if var in atmosphere_monthly_ds.data_vars:
        atmosphere_monthly_ds[var].mean(['latitude', 'longitude']).groupby('time.year').mean().plot(ax=axs[row,col])
    elif var in atm2oce_dict['ace2'].data_vars:
        atm2oce_dict['ace2'].mean(['latitude', 'longitude'])[var].groupby('time.year').mean().plot(ax=axs[row,col])
    elif var in oce2atm_dict['ace2'].data_vars:
        oce2atm_dict['ace2'].mean(['latitude', 'longitude'])[var].groupby('time.year').mean().plot(ax=axs[row,col])
        

    axs[row,col].set_title(f"{name_lookup.get(var, {}).get('name',var)} [{name_lookup.get(var, {}).get('units','')}]")
    axs[row,col].set_ylabel("")

    new_tick_labels = [item.get_text() if item.get_text() != 'Jul' else "" for item in axs[row,col].get_xticklabels()]
    axs[row,col].set_xticklabels(new_tick_labels)
    axs[row,col].set_xlabel('Year  ')

    axs[row,col].legend()


# %%
# Compare last step of first leg, and first step of second leg, to check for any discontinuities in the restart process
ds_last = xr.load_dataset('/home/ecme4254/scratch/run_dir/n3.6_ace2_1951_spinupCMIP6_test_19510101-19520101_m0/router/oce2atm_8760h_ace2_nemo.nc')
ds_first = xr.load_dataset('/home/ecme4254/scratch/run_dir/n3.6_ace2_1951_spinupCMIP6_test_19520101-19530101_m0/router/oce2atm_0h_ace2_nemo.nc')

# %%
# Plot difference of sea surface temperature at the restart point, to check for any discontinuities in the restart process
plt.figure(figsize=(10,5))
(ds_first['sea_surface_temperature'].isel(time=0) - ds_last['sea_surface_temperature'].isel(time=-1)).plot()
plt.title('Difference in sea surface temperature at restart point')
print('Max difference in SST at restart point:', np.nanmax(np.abs(ds_first['sea_surface_temperature'].isel(time=0) - ds_last['sea_surface_temperature'].isel(time=-1))))
# plt.savefig(f'{experiment_id}_restart_sst_difference.pdf', format='pdf')

# %%
# check daily data

# %% [markdown]
# # Check ACE2 surface temp compared to NEMO

# %%
fig, axs = plt.subplots(1, 1, figsize=(6,6))

xr.where(sea_mask, oce2atm_dict['ace2']['sea_surface_temperature'],np.nan).sel(latitude=slice(-1,1)).mean(['latitude', 'longitude']).plot(ax=axs, label='NEMO', color='b')
xr.where(sea_mask, atmosphere_monthly_ds['surface_temperature'],np.nan).sel(latitude=slice(-1,1)).mean(['latitude', 'longitude']).plot(ax=axs, label='ACE', color='r')

plt.legend()

# %% [markdown]
# # Compare to fluxes in restart file

# %%
range_lookup = {'qns_b': {'vmin': -500, 'vmax': 500},
                'utau_b': {'vmin': -0.5, 'vmax': 0.5},
                'vtau_b': {'vmin': -0.5, 'vmax': 0.5},
                'emp_b': {'vmin': -0.0001, 'vmax': 0.0001}}

# %%
restart_ds = xr.load_dataset("/home/ecme4254/hpcperm/ece3data/nemo/restart/ORCA1/19510101/restart_oce.nc")[list(range_lookup.keys()) + ['nav_lon', 'nav_lat', 'nav_lev']]

# %%
restart_ds = restart_ds.assign_coords(longitude=restart_ds['nav_lon'], latitude=restart_ds['nav_lat'])

# %%
instantaneous_flux_ds = xr.load_dataset('/home/ecme4254/scratch/run_dir/n3.6_ace2_19510101-19610101_m0/router/atm2oce_19510102-06_ace2_nemo.nc')
instantaneous_era5_flux_ds = xr.load_dataset('/scratch/ecme4254/run_dir/n3.6_ace2_20100101-20100501_m0/router/atm2oce_20100102-06_era5_nemo.nc')

# %%
instantaneous_flux_ds['A_EmP'] = instantaneous_flux_ds['A_Evap_total'] - (instantaneous_flux_ds['A_Precip_liquid'] + instantaneous_flux_ds['A_Precip_solid'])
instantaneous_era5_flux_ds['A_EmP'] = instantaneous_era5_flux_ds['A_Evap_total'] - (instantaneous_era5_flux_ds['A_Precip_liquid'] + instantaneous_era5_flux_ds['A_Precip_solid'])

# %%
import xesmf as xe

regridder = xe.Regridder(restart_ds, instantaneous_flux_ds, "bilinear", unmapped_to_nan=True, ignore_degenerate=True)
restart_ds = regridder(restart_ds)

# %%
# Compare distributions of EmP fluxes
fig, ax = plt.subplots(1,1)
pd.Series(restart_ds['emp_b'].values.flatten()).plot.kde(ax=ax, label='Restart file')
pd.Series(instantaneous_flux_ds['A_EmP'].values.flatten()).plot.kde(ax=ax, label='ACE flux')
pd.Series(instantaneous_era5_flux_ds['A_EmP'].values.flatten()).plot.kde(ax=ax, label='ERA5 flux')
ax.set_yscale('log')
ax.set_ylim([10e-1, 1e5])
ax.legend()

# %%
# Evaluate total freshwater budget

# %%
fig, ax = plt.subplots(1,2, figsize=(8*2,6))
xr.where(sea_mask, instantaneous_flux_ds['A_EmP'], np.nan).plot(x='longitude', y='latitude', vmax=0.0004, vmin=-0.0004, cmap='RdBu_r', ax=ax[0])

restart_ds['emp_b'].isel(t=0).plot(vmax=0.0004, vmin=-0.0004, cmap='RdBu_r', ax=ax[1])

# %%
flux_name_mapping = {'qns_b': 'A_Qns_oce', 'utau_b': 'A_TauX_oce', 'vtau_b': 'A_TauY_oce', 'emp_b': 'A_EmP'}
flux_era5_mapping = {'qns_b': 'A_Qns_oce', 'utau_b': 'A_TauX_oce', 'vtau_b': 'A_TauY_oce', 'emp_b': 'A_EmP'}

# %%
era5_flux_vars = ['mean_surface_sensible_heat_flux', 
                  'mean_surface_latent_heat_flux', 
                  'mean_surface_net_long_wave_radiation_flux', 
                  'evaporation', 'instantaneous_eastward_turbulent_surface_stress', 
                  'instantaneous_northward_turbulent_surface_stress', 
                  'mean_surface_net_short_wave_radiation_flux', 'total_precipitation'
            ]
era5_flux_ds = []

for era5_var in era5_flux_vars:
    fp = os.path.join(ERA5_DIR, 'surface', era5_var, f"era5_{era5_var}_19510101.nc")
    tmp_era5_flux_ds = xr.open_dataset(fp)
    tmp_era5_flux_ds = tmp_era5_flux_ds.rename({list(tmp_era5_flux_ds.data_vars)[0]: era5_var})

    era5_flux_ds.append(tmp_era5_flux_ds)


era5_flux_ds = xr.merge(era5_flux_ds)
era5_flux_ds['A_Qns_oce'] = era5_flux_ds['mean_surface_sensible_heat_flux'] + era5_flux_ds['mean_surface_latent_heat_flux'] + era5_flux_ds['mean_surface_net_long_wave_radiation_flux']
era5_flux_ds['A_TauX_oce'] = era5_flux_ds['instantaneous_eastward_turbulent_surface_stress']
era5_flux_ds['A_TauY_oce'] = era5_flux_ds['instantaneous_northward_turbulent_surface_stress']
era5_flux_ds['A_EmP']= (-1*era5_flux_ds['evaporation'] - era5_flux_ds['total_precipitation']) * 1000 / (3600)

# %%
# regrid to the atmosphere
grid = atm2oce_dict['ace2'][list(atm2oce_dict['ace2'].data_vars)[0]]
regridder = xe.Regridder(era5_flux_ds, grid, "bilinear")

era5_flux_ds = regridder(era5_flux_ds)

# %%

for n, (flux_var, atmosphere_var) in enumerate({'emp_b': 'A_EmP'}.items()): 
    fig, ax = plt.subplots(1,3, figsize=(3*8,5))
    restart_ds[flux_var].isel(t=0).plot(ax=ax[0], vmin=range_lookup[flux_var]['vmin'], vmax=range_lookup[flux_var]['vmax'])
    xr.where(sea_mask, instantaneous_flux_ds[atmosphere_var], np.nan).plot(ax=ax[1], x='longitude', y='latitude',  vmin=range_lookup[flux_var]['vmin'], vmax=range_lookup[flux_var]['vmax'])

    if atmosphere_var in era5_flux_ds.data_vars:
        xr.where(sea_mask, era5_flux_ds[atmosphere_var].isel(time=0), np.nan).plot(ax=ax[2], x='longitude', y='latitude',  vmin=range_lookup[flux_var]['vmin'], vmax=range_lookup[flux_var]['vmax'])

    ax[0].set_title('NEMO restart flux')
    ax[1].set_title('ACE2 flux')
    ax[2].set_title('ERA5 flux')

# %% [markdown]
# ## NEMO ocean output

# %%

# %%
# ERA5 experiment
nemo_t_grid_dict = {'ace2': xr.open_mfdataset(glob(os.path.join(base_dir, 'nemo_output_ace2/nemo_ocean_output_grid_T*.nc')), decode_times=False).compute()}
# for k in ['noheat', 'nofreshwater', 'nomomentum', 'conserved']:
#     nemo_t_grid_dict[k] = xr.open_mfdataset(glob(os.path.join(base_dir, f'nemo_output_{k}/nemo_ocean_output_grid_T*.nc'))).compute()

# nemo_u_grid_fps = glob(os.path.join(base_dir, 'nemo_output_era5_nofreshwater/nemo_ocean_output_grid_U_3D_*.nc'))

# %%
time_vals = nemo_t_grid_dict['ace2']['time_counter'].values

# %%
da_dict = {'base': nemo_t_grid_ds['ssh'].isel(time_counter=0),
           'noheat': nemo_t_grid_noheat_ds['ssh'].isel(time_counter=0),
           'nofreshwater': nemo_t_grid_noheat_ds['ssh'].isel(time_counter=0)}
    
# da_dict['GenCast - ERA5'] = da_dict['gencast'] - da_dict['era5']
fig, axs = plot_grid_shared_axes(da_grid= [list(da_dict.values())], 
                          num_rows=1, 
                          num_cols=len(da_dict.keys()), 
                          titles_grid=[[item.replace('ace2', 'ACE2').replace('era5', 'ERA5') for item in da_dict.keys()]],
                          cbar_label=name_lookup.get(v,v),
                          vmin=ranges.get(v, [None])[0], 
                          vmax=ranges.get(v, [None,None])[1],
                          shrink_factor=0.7, 
                          central_longitude=180, 
                          cmap='RdBu_r', 
                          mask=None)

# %%
# Normal data
nemo_t_grid_dict = {atmosphere_type: xr.open_mfdataset(glob(os.path.join(base_dir, f'nemo_output_{atmosphere_type}/nemo_ocean_output_grid_T*.nc'))).compute() for atmosphere_type in atmosphere_types}

# %%
nemo_time_vals = sorted(nemo_t_grid_dict['ace2']['time_counter'].values)
ace2_time_vals = atmosphere_monthly_ds['time'].values

# %%
# Time series of overall averages

for var in ['ssh', 'sst', 'sss']:

    fig, ax = plt.subplots(1,1, figsize=(8,5))
    for atm_type, ds in nemo_t_grid_dict.items():
        
        
        mean_series = ds[var].sel(time_counter=time_vals).mean(['x', 'y']).values
        std_series = ds[var].sel(time_counter=time_vals).mean(['x', 'y']).values
        ax.plot(time_vals, mean_series, label=atm_type)
        # ax.fill_between(atm2oce_dict['ace2']['time'], y1=mean_series+std_series,y2=mean_series-std_series )
        ax.set_title(var)

        if var == 'sst':
            ax.plot(time_vals, mean_series, label=atm_type)
        ax.set_xticks(time_vals)
        ax.set_xticklabels(ace2_time_vals[:len(time_vals)], rotation=45)

    plt.legend()

# %%
t_vars = ['sst', 'ssh', 'sss']
vmin_lookup = {'ssh': {'vmax': 2, 'vmin': -2}, 'sst': {'vmax': 30, 'vmin': -30},
              'sss': {'vmax': 40, 'vmin': 10}}
fig, ax = plt.subplots(len(t_vars), 3, figsize=(8*3, len(t_vars)*5))

for m, var in enumerate(t_vars):
    
    # for n, dt in enumerate([min(time_vals), max(time_vals)]):
    
    nemo_t_grid_dict['ace2'].sel(time_counter=min(time_vals))[var].plot(ax=ax[m,0], vmin=vmin_lookup[var]['vmin'],vmax=vmin_lookup[var]['vmax'], cmap='RdBu_r')
    nemo_t_grid_dict['ace2'].sel(time_counter=max(time_vals))[var].plot(ax=ax[m,1], vmin=vmin_lookup[var]['vmin'],vmax=vmin_lookup[var]['vmax'], cmap='RdBu_r')
    # nemo_t_grid_dict['era5'].sel(time_counter=max(time_vals))[var].plot(ax=ax[m,2], vmin=vmin_lookup[var]['vmin'],vmax=vmin_lookup[var]['vmax'], cmap='RdBu_r')

    ax[m,0].set_title('Init')
    ax[m,1].set_title('ACE2')
    ax[m,2].set_title('ERA5')
# (nemo_t_grid_ds.isel(time_counter=0)['sst'] - init_ds.isel(time_counter=0)['sst']).plot(ax=ax[n,m+1])

# %%
t_vars = ['sst', 'ssh', 'sss']
vmin_lookup = {'ssh': {'vmax': 2, 'vmin': -2}, 'sst': {'vmax': 30, 'vmin': -30},
              'sss': {'vmax': 40, 'vmin': 10}}
fig, ax = plt.subplots(2, len(t_vars) + 1, figsize=(8*(len(t_vars) + 1), 3*5))

for n, fp in enumerate([min(nemo_t_grid_fps), sorted(nemo_t_grid_fps)[-1]]):
    nemo_t_grid_ds = xr.open_dataset(fp)

    for m, var in enumerate(t_vars):
        nemo_t_grid_ds.isel(time_counter=0)[var].plot(ax=ax[n,m], vmin=vmin_lookup[var]['vmin'],vmax=vmin_lookup[var]['vmax'], cmap='RdBu_r')

    if fp == min(nemo_t_grid_fps):
        init_ds = nemo_t_grid_ds
    else:
        (nemo_t_grid_ds.isel(time_counter=0)['sst'] - init_ds.isel(time_counter=0)['sst']).plot(ax=ax[n,m+1])

# %%
u_vars = ['uoce']
vmin_lookup = {'ssh': 2, 'sst': 30}
fig, ax = plt.subplots(2, 1, figsize=(8, 2*5))

for n, fp in enumerate([min(nemo_u_grid_fps), max(nemo_u_grid_fps)]):
    nemo_u_grid_ds = xr.open_dataset(fp).sel(olevel=0.50576)

    nemo_u_grid_ds.isel(time_counter=0)['uoce'].plot(ax=ax[n], cmap='RdBu_r')

# %% [markdown]
#  ## Ice model output

# %%
experiment_names = ['ace2']

ice_dict = {atmosphere_type: xr.open_mfdataset(glob(os.path.join(base_dir, f'nemo_output_{atmosphere_type}/lim_output_icemod_*.nc')), preprocess = lambda x: x[['sithic', 'siconc', 'sivolu', 'sitemp']], decode_timedelta=False).compute() for atmosphere_type in experiment_names}


# %%
ice_dict={}
fps = chain.from_iterable([glob(os.path.join('/home/ecme4254/scratch/run_dir/n3.6_ace2_19510101-19610101_crash_experiment_m0', f'lim_output_icemod_{y}*.nc')) for y in range(1951,1957)])
ice_dict['ace2'] = xr.open_mfdataset(fps)

# %%
time_vals = sorted(ice_dict['ace2']['time_counter'].values)

# %%
for atm_type, ds in ice_dict.items():
    ds['sea_ice_extent'] = ds['siconc'] > 0.1

# %%

var_lookup = {'sithic': {'vmax': 20, 'vmin': 1},
              'siconc': {'vmax': 1, 'vmin':0},
               'sivolu': {'vmax': 20, 'vmin': 0},
                'sitemp': {'vmax': 0, 'vmin': -40}}


for var in ['sithic', 'siconc', 'sivolu', 'sitemp' ]:
    fig, ax = plt.subplots(1,len(ice_dict.keys())+1,figsize=(8*len(ice_dict.keys()),6))

    ice_dict['ace2'][var].sel(time_counter=time_vals[0]).plot(ax=ax[0], vmin=var_lookup[var]['vmin'], vmax=var_lookup[var]['vmax'])
    ice_dict['ace2'][var].sel(time_counter=time_vals[-1]).plot(ax=ax[1], vmin=var_lookup[var]['vmin'], vmax=var_lookup[var]['vmax'])
    # ice_dict['era5'][var].sel(time_counter=time_vals[-1]).plot(ax=ax[2], vmin=var_lookup[var]['vmin'], vmax=var_lookup[var]['vmax'])
    # ice_dict['ace2-bulk-ice'][var].sel(time_counter=time_vals[-1]).plot(ax=ax[3], vmin=var_lookup[var]['vmin'], vmax=var_lookup[var]['vmax'])
    ax[0].set_title(f'{var} init')
    ax[1].set_title(f'ACE2 {var} final')
    # ax[2].set_title(f'ERA5 {var} final')
    # ax[3].set_title(f'ACE2-bulk-ice {var} final ')

# %%
for var in ['sithic', 'siconc', 'sivolu', 'sitemp', 'sea_ice_extent']:

    fig, ax = plt.subplots(1,1, figsize=(8,5))
    for atm_type, ds in ice_dict.items():
            
    
        ds.sel(time_counter=time_vals)[var].mean(['y', 'x']).plot(ax=ax, label=atm_type)
    plt.legend()

fig, ax = plt.subplots(1,1, figsize=(8,5))
ds.sel(time_counter=time_vals)['sithic'].max(['y', 'x']).plot(ax=ax, label=atm_type)

# %% [markdown]
# ### Look at 6-hourly output from NEMO, to see what fluxes it is seeing, compared to what we're sending

# %%
dt = datetime.datetime(2010,1,1,18)

# %%
ocean_flux_output = xr.load_dataset(f"/hpcperm/ecme4254/run_dir/n3.6_ace2_20100101-20100201_m0/nemo_ocean_output_grid_T_6h_{dt.strftime('%Y%m%d')}-{dt.strftime('%Y%m%d')}.nc").sel(time_counter=dt + datetime.timedelta(hours=3))

# %%

# %%
atm2oce_ds = xr.load_dataset(f"/home/ecme4254/scratch/run_dir/n3.6_ace2_20100101-20100201_m0/router/atm2oce_{dt.strftime('%Y%m%d-%H')}_ace2_nemo.nc").sel(time=dt)
atm2oce_ds['A_Tau_oce'] = np.sqrt(atm2oce_ds['A_TauX_oce']**2 + atm2oce_ds['A_TauY_oce']**2)
atm2oce_ds['A_EmP_oce'] = atm2oce_ds['A_Evap_total'] - atm2oce_ds['A_Precip_liquid']- atm2oce_ds['A_Precip_solid']

# %%
flux_name_mapping = {'qns': 'A_Qns_oce', 'taum': 'A_Tau_oce', 'emp_oce': 'A_EmP_oce'}


# %%
range_lookup = {'qns': {'vmin': -500, 'vmax': 500},
                'taum': {'vmin': -0.5, 'vmax': 0.5},
                'emp_oce': {'vmin': -0.0001, 'vmax': 0.0001}}
for n, (flux_var, atmosphere_var) in enumerate(flux_name_mapping.items()): 
    fig, ax = plt.subplots(1,2, figsize=(2*8,5))
    ocean_flux_output[flux_var].plot(ax=ax[0], vmin=range_lookup[flux_var]['vmin'], vmax=range_lookup[flux_var]['vmax'], cmap='RdBu_r')
    xr.where(sea_mask, atm2oce_ds[atmosphere_var], np.nan).plot(ax=ax[1], x='longitude', y='latitude', vmin=range_lookup[flux_var]['vmin'], vmax=range_lookup[flux_var]['vmax'], cmap='RdBu_r')

    ax[0].set_title('NEMO restart flux')
    ax[1].set_title('ACE2 flux')

# %% [markdown]
# ## Compare raw fluxes

# %%

# %%
# Collect ERA5 flux data
# Note that currently just keeping 12h and 0h since that is what we have collected

era5_flux_vars = ['mean_surface_sensible_heat_flux', 
                  'mean_surface_latent_heat_flux', 
                  'mean_surface_net_long_wave_radiation_flux', 
                  'evaporation', 'instantaneous_eastward_turbulent_surface_stress', 
                  'instantaneous_northward_turbulent_surface_stress', 
                  'mean_surface_net_short_wave_radiation_flux'
            ]
era5_flux_ds = []

for era5_var in era5_flux_vars:
    fps = [os.path.join(ERA5_DIR, 'surface', era5_var, f"era5_{era5_var}_{tmp_dt.strftime('%Y%m%d')}.nc") for tmp_dt in expanded_time_vals[1:]]
    tmp_era5_flux_ds = xr.open_mfdataset(fps)
    tmp_era5_flux_ds = tmp_era5_flux_ds.rename({list(tmp_era5_flux_ds.data_vars)[0]: era5_var})

    era5_flux_ds.append(tmp_era5_flux_ds)


era5_flux_ds = xr.merge(era5_flux_ds)
era5_flux_ds = era5_flux_ds.resample(time='6h', 
                            label='right', 
                            ).mean()

era5_flux_ds = era5_flux_ds.rename({v: f'{v}_6hr' for v in era5_flux_vars})
era5_flux_ds_monthly = era5_flux_ds.resample(time='MS').mean()

# %%
# Collect ERA5 data
# Note that currently just keeping 12h and 0h since that is what we have collected

era5_sea_vars = [
                'sea_surface_temperature',
            'sea_ice_cover',
             'ice_temperature_layer_1',
             'forecast_albedo'
            ]
era5_sea_ds = []

for era5_var in era5_sea_vars:
    fps = [os.path.join(ERA5_DIR, 'surface', era5_var, f"era5_{era5_var}_{tmp_dt.strftime('%Y%m%d')}.nc") for tmp_dt in expanded_time_vals[1:]]

    tmp_era5_sea_ds = xr.open_mfdataset(fps)
    tmp_era5_sea_ds = tmp_era5_sea_ds.rename({list(tmp_era5_sea_ds.data_vars)[0]: era5_var})

    era5_sea_ds.append(tmp_era5_sea_ds)

era5_sea_ds = xr.merge(era5_sea_ds)

era5_sea_ds_monthly = era5_sea_ds.resample(time='MS').mean().compute()
era5_sea_ds_monthly = era5_sea_ds_monthly.rename({'sea_ice_cover': 'sea_ice_fraction'})

# %%
# regrid to the atmosphere
grid = atm2oce_dict['era5'][list(atm2oce_dict['era5'].data_vars)[0]]
regridder = xe.Regridder(era5_flux_ds_monthly, grid, "bilinear")


era5_sea_ds_monthly = regridder(era5_sea_ds_monthly)
era5_flux_ds_monthly = regridder(era5_flux_ds_monthly)

# %%
atmosphere_monthly_ds = atmosphere_monthly_ds.rename({'LHTFLsfc': 'mean_surface_latent_heat_flux_6hr',
                                                     'SHTFLsfc': 'mean_surface_sensible_heat_flux_6hr'})

# ACE has a different sign convention to ERA5 (which is taken into account in the flux calculation, but not in this raw data)
atmosphere_monthly_ds['mean_surface_latent_heat_flux_6hr'] = -1*atmosphere_monthly_ds['mean_surface_latent_heat_flux_6hr']
atmosphere_monthly_ds['mean_surface_sensible_heat_flux_6hr'] = -1*atmosphere_monthly_ds['mean_surface_sensible_heat_flux_6hr']

# %%
plot_vars = ['mean_surface_latent_heat_flux_6hr','mean_surface_latent_heat_flux_6hr']

nrows = len(plot_vars)
ncols=3
                                  
for n, v in enumerate(plot_vars):

    # tmp_ice_mask = oce2atm_mean_dict['era5']['sea_ice_fraction'] > 0.1

    da_dict = {model_name: xr.where(sea_mask, atmosphere_monthly_ds[v].isel(time=0), np.nan),
              'ERA5': xr.where(sea_mask, era5_flux_ds_monthly[v].isel(time=0), np.nan),}

    da_dict = {k: v.transpose('latitude', 'longitude') for k,v in da_dict.items()}
        
    fig, axs = plot_grid_shared_axes(da_grid= [list(da_dict.values())], 
                              num_rows=1, 
                              num_cols=len(da_dict.keys()), 
                              titles_grid=[[item.replace('ace2', 'ACE2').replace('era5', 'ERA5') for item in da_dict.keys()]],
                              cbar_label=name_lookup.get(v,v),
                              vmin=ranges.get(v, [None])[0], 
                              vmax=ranges.get(v, [None,None])[1],
                              shrink_factor=0.7, 
                              central_longitude=180, 
                              cmap='RdBu_r', 
                              mask=None)

# %% [markdown]
# ## Compare sea ice

# %%
plot_vars = ['sea_ice_fraction']
nrows = len(plot_vars)
ncols=3

                                  
for n, v in enumerate(plot_vars):

    # tmp_ice_mask = oce2atm_mean_dict['era5']['sea_ice_fraction'] > 0.1

    da_dict = {atm_type: xr.where(sea_mask, oce2atm_dict[atm_type][v].isel(time=0), np.nan) for atm_type in oce2atm_dict.keys()}
    da_dict['ERA5'] = era5_sea_ds_monthly[v].isel(time=0)

    if v.endswith('_ice'):
        da_dict = {k : xr.where(ice_mask, v, np.nan) for k,v in da_dict.items()}

    da_dict = {k: v.transpose('latitude', 'longitude') for k,v in da_dict.items()}
        
    # da_dict['GenCast - ERA5'] = da_dict['gencast'] - da_dict['era5']
    fig, axs = plot_grid_shared_axes(da_grid= [list(da_dict.values())], 
                              num_rows=1, 
                              num_cols=len(da_dict.keys()), 
                              titles_grid=[[item.replace('ace2', 'ACE2').replace('era5', 'ERA5') for item in da_dict.keys()]],
                              cbar_label=name_lookup.get(v,v),
                              vmin=ranges.get(v, [None])[0], 
                              vmax=ranges.get(v, [None,None])[1],
                              shrink_factor=0.7, 
                              central_longitude=180, 
                              cmap='RdBu_r', 
                              mask=None)

# %%
