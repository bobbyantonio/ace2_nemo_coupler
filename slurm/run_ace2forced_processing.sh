#!/bin/bash 
#SBATCH --job-name=ace2forced-processing
#SBATCH --output=logs/ace2forced-processing-%A.txt 
#SBATCH --qos=np
#SBATCH --ntasks=1
#SBATCH --time=1-00:00:00 
#SBATCH --mem=200gb

source ~/.bashrc

jupytext --sync notebooks/process_forced_ace2_run.ipynb;

conda activate ece4

python -m notebooks.process_forced_ace2_run;
