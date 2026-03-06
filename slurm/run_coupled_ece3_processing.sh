#!/bin/bash 
#SBATCH --job-name=coupled-ece3-proc
#SBATCH --output=logs/process_ece3-%A.txt 
#SBATCH --qos=np
#SBATCH --ntasks=1
#SBATCH --time=1-00:00:00 
#SBATCH --mem=100gb

source ~/.bashrc

jupytext --sync notebooks/process_ece_data.ipynb;

conda activate ece4

# python -m notebooks.process_ece_data --ece3-experiment-id EC-Earth3P_control-1950 --years 1951-1961 --debug --month-lag-max 1;

python -m notebooks.process_ece_data --ece3-experiment-id EC-Earth3P_control-1950 --years 1951-2020 --month-lag-max 5 --ece3-data-dir /home/ecme4254/scratch/ece3_cmip6_data_download/EC-Earth3P_control-1950 --ace2-data-dir /home/ecme4254/hpcperm/ml_model_data/ace2 --base-output-dir /home/ecme4254/perm/repos/ace2_nemo_coupler/notebooks/processed_data;

# python -m notebooks.process_ece_data --ece3-experiment-id EC-Earth3P_hist-1950 --years 1951-2013 --month-lag-max 5;

# python -m notebooks.process_ece_data --ece3-experiment-id EC-Earth3_historical --years 1951-2013;