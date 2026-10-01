#!/bin/bash 
#SBATCH --job-name=fetch-wb-clim
#SBATCH --output=logs/fetch-wb-clim-%A.txt 
#SBATCH --qos=nf
#SBATCH --ntasks=1
#SBATCH --time=00-10:00:00 
#SBATCH --mem=50gb
#SBATCH --account=spgbanto

source ~/.kshrc
source ~/.initConda.sh
conda activate graphcast

python -m scripts.fetch_weatherbench_climatology --output-dir ${HPCPERM}/era5/climatology --vars evaporation mean_surface_sensible_heat_flux mean_surface_latent_heat_flux mean_surface_net_long_wave_radiation_flux instantaneous_eastward_turbulent_surface_stress instantaneous_northward_turbulent_surface_stress