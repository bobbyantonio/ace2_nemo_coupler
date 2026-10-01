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
import xarray as xr
import matplotlib.pyplot as plt

# %%

# %%
model_for_comparison = "spinupCMIP6"
for component in ['oce', 'ice']:
    print(component)
    for n in range(32):

        # Load 40 year restart file for spinupCMIP6 run
        rstart_40yr = xr.load_dataset(f"/home/ecme4254/scratch/run_dir/n3.6_ace2_1951_{model_for_comparison}_19810101-19910101_m0/n3.6_ace2_1951_{model_for_comparison}_19810101-19910101_m0_00116864_restart_oce_{n:04d}.nc")

        # The restart I gave to Pablo for the 40 year run
        rstart_40yr_pablo = xr.load_dataset(f"/home/ecme4254/perm/repos/ace2_nemo_coupler/ace2-nemo-40yr-spinup/restart_oce_{n:04d}.nc")

        max_temp_diff = (rstart_40yr['tn'] - rstart_40yr_pablo['tn']).max().item()
        print(max_temp_diff)
        
        if max_temp_diff != 0:
            raise ValueError
        
        if (rstart_40yr['time_counter'] - rstart_40yr_pablo['time_counter']).item() != 0.0:
            raise ValueError('bad time_counter')

# %%
model_for_comparison = "spinupCMIP6"
for component in ['oce', 'ice']:
    print(component)
    for n in range(32):

        # Load 40 year restart file for spinupCMIP6 run
        rstart_40yr = xr.load_dataset(f"/home/ecme4254/hpcperm/model_runs/n3.6_ace2_1951_{model_for_comparison}_19510101-20210101_m0/restarts/n3.6_ace2_1951_{model_for_comparison}_19810101-19910101_m0_00116864_restart_oce_{n:04d}.nc")

        # The restart I gave to Pablo for the 40 year run
        rstart_40yr_pablo = xr.load_dataset(f"/home/ecme4254/perm/repos/ace2_nemo_coupler/ace2-nemo-40yr-spinup/restart_oce_{n:04d}.nc")

        max_temp_diff = (rstart_40yr['tn'] - rstart_40yr_pablo['tn']).max().item()
        print(max_temp_diff)
        
        if max_temp_diff != 0:
            raise ValueError

# %%
for component in ['oce', 'ice']:
    print(component)
    for n in range(32):

        # Load 40 year restart file for spinupCMIP6 run
        rstart_40yr = xr.load_dataset(f"/home/ecme4254/hpcperm/model_runs/n3.6_ace2_1951_spinup_19510101-20210101_m0/restarts/n3.6_ace2_1951_spinup_19810101-19910101_m0_00131472_restart_oce_{n:04d}.nc")

        # The restart I gave to Pablo for the 40 year run
        rstart_40yr_pablo = xr.load_dataset(f"/home/ecme4254/perm/repos/ace2_nemo_coupler/ace2-nemo-40yr-spinup/restart_oce_{n:04d}.nc")

        max_temp_diff = (rstart_40yr['tn'].sel(z=0) - rstart_40yr_pablo['tn'].sel(z=0)).max().item()
        print(max_temp_diff)
        
        if (rstart_40yr['time_counter'] - rstart_40yr_pablo['time_counter']).item() != 0.0:
            raise ValueError('bad time_counter')
             
        if max_temp_diff != 0:
            raise ValueError


# %%
# Same test for the 70 year restarts
model_for_comparison = "spinupCMIP6"
for component in ['oce', 'ice']:
    print(component)
    for n in range(32):

        # Load 40 year restart file for spinupCMIP6 run
        rstart_70yr = xr.load_dataset(f"/home/ecme4254/scratch/run_dir/n3.6_ace2_1951_{model_for_comparison}_20110101-20210101_m0/n3.6_ace2_1951_{model_for_comparison}_20110101-20210101_m0_00116896_restart_oce_{n:04d}.nc")

        # The restart I gave to Pablo for the 70 year run
        rstart_70yr_pablo = xr.load_dataset(f"/home/ecme4254/perm/repos/ace2_nemo_coupler/ace2-nemo-70yr-spinup/restart_oce_{n:04d}.nc")

        max_temp_diff = (rstart_70yr['tn'] - rstart_70yr_pablo['tn']).max().item()
        print(max_temp_diff)
        
        if max_temp_diff != 0:
            raise ValueError


# %%
# Same test for the 70 year restarts
model_for_comparison = "spinupCMIP6"
for component in ['oce', 'ice']:
    print(component)
    for n in range(32):

        # Load 40 year restart file for spinupCMIP6 run
        rstart_70yr = xr.load_dataset(f"/home/ecme4254/hpcperm/model_runs/n3.6_ace2_1951_spinupCMIP6_19510101-20210101_m0/restarts/n3.6_ace2_1951_{model_for_comparison}_20110101-20210101_m0_00116896_restart_oce_{n:04d}.nc")

        # The restart I gave to Pablo for the 70 year run
        rstart_70yr_pablo = xr.load_dataset(f"/home/ecme4254/perm/repos/ace2_nemo_coupler/ace2-nemo-70yr-spinup/restart_oce_{n:04d}.nc")

        max_temp_diff = (rstart_70yr['tn'] - rstart_70yr_pablo['tn']).max().item()
        print(max_temp_diff)
        
        if max_temp_diff != 0:
            raise ValueError


# %%
for component in ['oce', 'ice']:
    print(component)
    for n in range(32):

        # Load 70 year restart file for spinupCMIP6 run
        rstart_70yr = xr.load_dataset(f"/home/ecme4254/hpcperm/model_runs/n3.6_ace2_1951_spinup_19510101-20210101_m0/final_restart_files/n3.6_ace2_1951_spinup_20110101-20210101_m0_00131508_restart_oce_{n:04d}.nc")

        # The restart I gave to Pablo for the 70 year run
        rstart_70yr_pablo = xr.load_dataset(f"/home/ecme4254/perm/repos/ace2_nemo_coupler/ace2-nemo-70yr-spinup/restart_oce_{n:04d}.nc")

        max_temp_diff = (rstart_70yr['tn'].sel(z=0) - rstart_70yr_pablo['tn'].sel(z=0)).max().item()
        print(max_temp_diff)
        
        if max_temp_diff != 0:
            raise ValueError

# %%
ds = xr.load_dataset("/home/ecme4254/perm/ece3data/nemo/domain/ORCA1/domain_cfg.nc")

# %%
xr.load_dataset(f'/home/ecme4254/scratch/run_dir/n3.6_ace2_spinupCMIP6_currentTest_19510101-19520101_m0/nemo_ocean_output_grid_V_195108-195108.nc', decode_times=False)['ssv']

# %%
# Check for checkerboard pattern
variable = 'voce'
test_ds = xr.load_dataset(f'/home/ecme4254/hpcperm/model_runs/n3.6_ace2_spinupCMIP6_laplacian_19510101-20210101_m0/nemo_output_ace2/nemo_ocean_output_grid_V_3D_200808-200808.nc', decode_times=False)
transect = test_ds[variable].sel(x=slice(50,270), y=190).isel(time_counter=0)
(transect / transect.std('olevel')).plot(vmin=-6, vmax=6, cmap='RdBu_r')

# %%
# Check for checkerboard pattern
fig, ax = plt.subplots(1,2, figsize=(2*8,4))

month = 6
variable = 'voce'
baseline_ds = xr.load_dataset(f'/home/ecme4254/hpcperm/model_runs/n3.6_ace2_1951_spinupCMIP6_19510101-20210101_m0/nemo_output_ace2/nemo_ocean_output_grid_V_3D_1951{month:02d}-1951{month:02d}.nc', decode_times=False)
baseline_current = baseline_ds[variable].sel(x=slice(50,270), y=190).isel(time_counter=0)
(baseline_current / baseline_current.std('x')).plot(ax=ax[0])


# laplac_ds = xr.load_dataset(f'/home/ecme4254/hpcperm/model_runs/n3.6_ace2_spinupCMIP6_laplacian_19510101-20210101_m0/nemo_output_ace2/nemo_ocean_output_grid_V_3D_1951{month:02d}-1951{month:02d}.nc', decode_times=False)
# laplac = laplac_ds[variable].sel(x=slice(50,270), y=190).isel(time_counter=0)
# (laplac / laplac.std('x')).plot(ax=ax[1])

# coords_and_laplac_ds = xr.load_dataset(f'/home/ecme4254/hpcperm/model_runs/n3.6_ace2_spinupCMIP6_currentTest_19510101-19520101_m0/nemo_output_ace2/nemo_ocean_output_grid_V_3D_1951{month:02d}-1951{month:02d}.nc', decode_times=False)
# coords_and_laplac = coords_and_laplac_ds[variable].sel(x=slice(50,270), y=190).isel(time_counter=0)
# (coords_and_laplac / coords_and_laplac.std('x')).plot(ax=ax[2])

jperio_ds = xr.load_dataset(f'/home/ecme4254/scratch/run_dir/n3.6_ace2_spinupCMIP6_jperioTest_19510101-19520101_m0/nemo_ocean_output_grid_V_3D_1951{month:02d}-1951{month:02d}.nc', decode_times=False)
jperio = jperio_ds[variable].sel(x=slice(50,270), y=190)
(jperio / jperio.std('x')).plot(ax=ax[1])

ax[0].set_title('Baseline')
# ax[1].set_title('With Laplacian viscosity')
# ax[2].set_title('With Laplacian viscosity + consistent coordinates')
ax[1].set_title('jperio=6')

for a in ax:
    a.set_ylabel('Depth')



# %%
ds = xr.load_dataset("/home/ecme4254/scratch/ece3_cmip6_data_download/EC-Earth3P_control-1950/tas/tas_Amon_EC-Earth3P_control-1950_r1i1p2f1_gr_195101-195112.nc")

# %%
xr.load_dataset("/home/ecme4254/perm/cmip6_data_from_pablo/tos/tos_Omon_EC-Earth3_historical_r1i1p1f1_gn_195001-195012.nc")

# %%
xr.load_dataset("/home/ecme4254/perm/cmip6_data_from_pablo/tas/tas_Amon_EC-Earth3_historical_r1i1p1f1_gr_195001-195012.nc")

# %%
ds_2 = xr.load_dataset("/home/ecme4254/scratch/ece3_cmip6_data_download/EC-Earth3P_control-1950/tos/tos_Omon_EC-Earth3P_control-1950_r1i1p2f1_gn_195101-195112.nc")

# %%
ds_2['longitude']

# %%
ds['lat']

# %%
test_ds = xr.load_dataset(f'/home/ecme4254/scratch/run_dir/n3.6_ace2_spinupCMIP6_currentTest_19510101-19520101_m0/nemo_ocean_output_grid_V_3D_195108-195108.nc', decode_times=False)
transect = test_ds[variable].sel(x=slice(50,270), y=190).isel(time_counter=0)
(transect / transect.std('olevel')).plot()

# %%
test_ds[variable].sel(x=slice(50,270), y=190).plot()

# %%
test_ds['voce'].isel(time_counter=0).sel(y=190).plot()

# %%
# Load the namelists and combine them



# %%
from glob import glob
import os

import shutil

atmosphere_source='ace2'

previous_rundir = '/home/ecme4254/scratch/run_dir/n3.6_ace2_spinupCMIP6_laplacian_19510101-19610101_m0'
previous_expid='n3.6_ace2_spinupCMIP6_laplacian_19510101-19610101_m0'
previous_router=os.path.join(previous_rundir, 'router')

rundir='/home/ecme4254/scratch/run_dir/n3.6_ace2_spinupCMIP6_laplacian_19560101-19610101_m0'
router_dir=os.path.join(rundir, 'router')
new_expid ='n3.6_ace2_spinupCMIP6_laplacian_19560101-19610101_m0'

nemo_restart_files = glob(os.path.join(previous_rundir, f'{previous_expid}_*_restart_oce_*.nc')) + glob(os.path.join(previous_rundir, f'{previous_expid}_*_restart_ice_*.nc'))


# %%

for file in nemo_restart_files:
    shutil.copy(file, os.path.join(rundir, file.split('/')[-1].replace(previous_expid, new_expid)))
    


# %%
# Move final ocean file from previous run to be initial condition for this run
final_oce2atm_fp = os.path.join(previous_router, f'oce2atm_87672h_{atmosphere_source}_nemo.nc')

shutil.copy(final_oce2atm_fp, os.path.join(router_dir, f'restart_oce2atm_0h_{atmosphere_source}_nemo.nc'))

# ML atmosphere restart file
final_atm_restart_fp = os.path.join(previous_rundir, f'restart/restart_{atmosphere_source}.nc')
shutil.copy(final_atm_restart_fp, os.path.join(rundir, f'restart_{atmosphere_source}.nc'))

# Also copy final atmosphere file to router directory for use as initial condition
final_atmosphere_fp = os.path.join(previous_router, f'{atmosphere_source}_87672h.nc')
shutil.copy(final_atmosphere_fp, os.path.join(router_dir, f'{atmosphere_source}_0h.nc'))

# %%
