#!/usr/bin/bash
#SBATCH --nodes=1
#SBATCH --time=0-5:00:00
#SBATCH --qos=np
#SBATCH --output=logs/run-router-only-%A.txt

source ~/.kshrc
source ~/.initConda.sh
conda activate ece4


RUN_DIR=/ec/res4/hpcperm/ecme4254/run_dir/NG12

cd /perm/ecme4254/repos/ecearth4/python_scripts/

rm $RUN_DIR/log/router.log

python -m router --model-directory $RUN_DIR --atmosphere-source era5 --router-data-directory ${RUN_DIR}/router --ocean-source era5 --climatology-directory /home/ecme4254/hpcperm/era5/climatology --era5-directory /home/ecme4254/hpcperm/era5 --atmospheric-timestep-hrs 12 --atmospheric-resolution 1 --coupling-timestep-secs 43200;