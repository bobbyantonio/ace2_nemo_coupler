#!/bin/bash 
#SBATCH --job-name=coupled-ece3-proc
#SBATCH --output=logs/process_ece3-%A.txt 
#SBATCH --qos=np
#SBATCH --ntasks=1
#SBATCH --time=1-00:00:00 
#SBATCH --mem=100gb

source ~/.bashrc

jupytext --sync eerie/coupled_experiments/process_ece_data.ipynb;

conda activate ece4

# python -m eerie.coupled_experiments.process_ece_data --ece3-experiment-id EC-Earth3P_control-1950 --years 1951-1961 --debug --month-lag-max 1;

python -m eerie.coupled_experiments.process_ece_data --ece3-experiment-id EC-Earth3P_control-1950 --years 1951-2020 --month-lag-max 5;

# python -m eerie.coupled_experiments.process_ece_data --ece3-experiment-id EC-Earth3P_hist-1950 --years 1951-2013 --month-lag-max 5;

# python -m eerie.coupled_experiments.process_ece_data --ece3-experiment-id EC-Earth3_historical --years 1951-2013;