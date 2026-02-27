#!/usr/bin/bash
#SBATCH --nodes=3
#SBATCH --qos=np
#SBATCH --time=1-0:00:00
#SBATCH --mem=50gb
#SBATCH --output=logs/compile-nemo-%A.txt

module load prgenv/intel
module load intel/2021.4.0
module load intel-mkl/19.0.5
module load hpcx-openmpi/2.9.0
module load hdf5-parallel/1.12.2
module load netcdf4-parallel/4.9.1
module load ecmwf-toolbox/2023.04.1.0

source ~/.kshrc
source ~/.initConda.sh
conda activate ece4

cd /perm/ecme4254/repos/ecearth4/scripts/build

CONFIG_NAME="nemo-mlatmosphere-coupled-config.yml"

se --loglevel debug user-settings.yml /perm/ecme4254/repos/ecearth4/scripts/platforms/ecmwf-hpc2020-intel+openmpi.yml compile-oasis.yml;

# # Notice that we need to include the experiment yaml file too
se --loglevel debug user-settings.yml /perm/ecme4254/repos/ecearth4/scripts/runtime/$CONFIG_NAME /perm/ecme4254/repos/ecearth4/scripts/platforms/ecmwf-hpc2020-intel+openmpi.yml compile-nemo.yml;

se --loglevel debug user-settings.yml /perm/ecme4254/repos/ecearth4/scripts/platforms/ecmwf-hpc2020-intel+openmpi.yml compile-xios.yml;

cd /perm/ecme4254/repos/ecearth4/scripts/runtime
se --loglevel debug user-config.yml /perm/ecme4254/repos/ecearth4/scripts/platforms/ecmwf-hpc2020-intel+openmpi.yml $CONFIG_NAME scriptlib/basic-setup-only.yml;