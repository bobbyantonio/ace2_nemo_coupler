#!/usr/bin/bash
#SBATCH --nodes=1
#SBATCH --time=1-0:00:00
#SBATCH --mem=50gb
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/compile-oasis.txt

source ~/.kshrc
source ~/.initConda.sh
conda activate ece4
cd /home/ecme4254/perm/repos/ecearth4/scripts/build

se --loglevel debug user-settings.yml /home/ecme4254/perm/repos/ecearth4/scripts/platforms/ecmwf-hpc2020-intel+openmpi.yml compile-oasis.yml;