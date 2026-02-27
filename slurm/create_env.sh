#!/usr/bin/bash
#SBATCH --nodes=1
#SBATCH --time=1-0:00:00
#SBATCH --mem=50gb
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=create-env
#SBATCH --output=logs/create-env-%A.txt

source ~/.kshrc

module load prgenv/intel
module load intel/2021.4.0
module load intel-mkl/19.0.5
module load hpcx-openmpi/2.9.0
module load hdf5-parallel/1.12.2
module load netcdf4-parallel/4.9.1
module load ecmwf-toolbox/2023.04.1.0

conda env create -f environment.yml;

conda activate ece4
cd $PERM/repos/rdy2cpl;
pip install .;

cd $PERM/repos/AirSeaFluxCode
pip install .;