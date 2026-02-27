#!/usr/bin/bash
#SBATCH --nodes=1
#SBATCH --time=1-0:00:00
#SBATCH --mem=50gb
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/compile-nemo3.6-%A.txt

source ~/.kshrc
source ~/.initConda.sh
conda activate ece4

module load prgenv/intel
module load intel/2021.4.0
module load intel-mkl/19.0.5
module load hpcx-openmpi/2.9.0
module load hdf5-parallel/1.12.2
module load netcdf4-parallel/4.9.1
module load ecmwf-toolbox/2023.04.1.0

ecearth4_dir="/home/ecme4254/perm/repos/ecearth4"
SRCDIR=${ecearth4_dir}/sources
cd $SRCDIR
# ec-conf3 -p ecmwf-hpc2020-intel-openmpi_BA --overwrite-parameter PLT:ACTIVE:ECEARTH_SRC_DIR=/perm/ecme4254/repos/ec-earth3/sources config-build-BA.xml

# CONFIG_NAME="nemo-mlatmosphere-coupled-config.yml"
# se --loglevel debug user-settings.yml /perm/ecme4254/repos/ecearth4/scripts/platforms/ecmwf-hpc2020-intel+openmpi.yml compile-oasis.yml;

# cd ${PERM}/repos/ec-earth3/sources/xios-2.5/
# ./make_xios --arch ecearth --full --use-oasis oasis3_mct

# Copy patches to WORK directory
cd ${ecearth4_dir}/nemo_patches/*.f90 ${SRCDIR}/nemo-3.6/CONFIG/ORCA1L75_LIM3/WORK/
./makenemo -n ORCA1L75_LIM3 clean
./makenemo -n ORCA1L75_LIM3 -m ecconf -j 1

# cd ${PERM}/repos/ec-earth3/sources/ifs-36r4
# ./makeifs

# cd ${PERM}/repos/ec-earth3/sources/runoff-mapper/src
# make

# cd ${PERM}/repos/ec-earth3/sources/lpjg/build
# cmake ..
# make
