#!/bin/bash 
#SBATCH --job-name=coupled-reanaly-proc
#SBATCH --output=logs/process_reanaly-%A.txt 
#SBATCH --qos=np
#SBATCH --ntasks=1
#SBATCH --time=1-00:00:00 
#SBATCH --mem=50gb

source ~/.bashrc

jupytext --sync notebooks/process_reanalysis.ipynb;

conda activate ece4

python -m notebooks.process_reanalysis --years 1951-2020 --month-lag-max 5;