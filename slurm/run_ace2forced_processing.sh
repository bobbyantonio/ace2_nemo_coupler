#!/bin/bash 
#SBATCH --job-name=ace2forced-processing
#SBATCH --output=logs/ace2forced-processing-%A.txt 
#SBATCH --qos=np
#SBATCH --ntasks=1
#SBATCH --time=00-05:00:00 
#SBATCH --mem=100gb

source ~/.bashrc

jupytext --sync notebooks/process_ace2_run.ipynb;

conda activate ece4

python -m notebooks.process_ace2_run;

