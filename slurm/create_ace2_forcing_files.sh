#!/bin/bash 
#SBATCH --job-name=ace2-forcing
#SBATCH --output=logs/ace2-forcing-%A.txt 
#SBATCH --qos=nf
#SBATCH --ntasks=1
#SBATCH --time=1-00:00:00 
#SBATCH --mem=50gb

source ~/.bashrc

conda activate ece4

python -m python_scripts.create_ace2_forcing_files;