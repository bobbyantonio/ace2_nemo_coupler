#!/bin/bash 
#SBATCH --job-name=ace2-forcing
#SBATCH --output=logs/ace2-forcing-%A.txt 
#SBATCH --partition=priority-atmproc,shared
#SBATCH --ntasks=1
#SBATCH --time=1-00:00:00 
#SBATCH --mem=50gb

source ~/.bashrc

jupytext --sync notebooks/create_ace2_ece3_forcing.ipynb;

conda activate ece4

python -m notebooks.create_ace2_ece3_forcing;