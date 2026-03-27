#!/bin/bash 
#SBATCH --job-name=coupled-ece3-proc
#SBATCH --output=logs/process_ece3-%A.txt 
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --time=00-04:00:00 
#SBATCH --account=eerie
#SBATCH --qos=short
#SBATCH --mem=100gb

source ~/.bashrc

jupytext --sync notebooks/process_ece_data.ipynb;

conda activate ece4

python -m notebooks.process_ece_data --ece3-experiment-id EC-Earth3_piControl --years 1951-2020 --month-lag-max 1 --ece3-data-dir /work/scratch-pw4/portega --ace2-data-dir /home/users/bantonio --base-output-dir /home/users/bantonio/repos/ace2_nemo_coupler/notebooks/processed_data --analysis-vars thetao sithick tas tos sos so --var-glob-string "*/{var}/*/*" --ace2-data-dir /home/users/bantonio;
