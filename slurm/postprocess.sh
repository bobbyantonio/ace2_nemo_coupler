#!/bin/bash
#SBATCH --job-name=postprocess-ace2
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --qos=np
#SBATCH --mem=50gb
#SBATCH --output=logs/postprocess-ace2-%A.txt

source ~/.kshrc
source ~/.initConda.sh
conda activate ece4

EXPID=n3.6_ace2_19510101-19610101_crash_experiment_m0
cp $PERM/repos/ecearth4/python_scripts/postprocess.py /home/ecme4254/scratch/run_dir/${EXPID};

cd /home/ecme4254/scratch/run_dir/${EXPID};

python postprocess.py --model-directory /home/ecme4254/scratch/run_dir/${EXPID} --ocean-source nemo --atmosphere-source ace2 --router-data-directory /home/ecme4254/scratch/run_dir/${EXPID}/router --results-data-directory /ec/res4/hpcperm/ecme4254/model_runs/${EXPID}/ --coupling-timestep-secs 21600 --overwrite;