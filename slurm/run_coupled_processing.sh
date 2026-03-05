#!/bin/bash 
#SBATCH --job-name=coupled-processing
#SBATCH --output=logs/coupled-processing-%A.txt 
#SBATCH --qos=np
#SBATCH --ntasks=1
#SBATCH --time=1-00:00:00 
#SBATCH --mem=200gb

source ~/.bashrc

jupytext --sync notebooks/process_model_run.ipynb;

conda activate ece4

# python -m notebooks.process_model_run --experiment-id n3.6_ace2_1951-2021_hist_compressed_19510101-20210101 --ensemble-members 0 1 --model-run-dir /home/ecme4254/hpcperm/model_runs --month-lag-max 1 --debug;

python -m notebooks.process_model_run --experiment-id n3.6_ace2_1951-2021_hist_compressed_19510101-20210101 --ensemble-members 0 1 2  --month-lag-max 5 --model-run-dir /home/ecme4254/hpcperm/model_runs;

# python -m notebooks.process_model_run --experiment-id n3.6_ace2_1951_control_compressed_19510101-20210101 --ensemble-members 0 1 2 --month-lag-max 5 --model-run-dir /home/ecme4254/perm/old_model_runs;

# python -m notebooks.process_model_run --experiment-id n3.6_ace2_historical_skt_19510101-20210101 --ensemble-members 0 --model-run-dir /home/ecme4254/hpcperm/model_runs;

# python -m notebooks.process_model_run --experiment-id n3.6_ace2_1951_ace2iceflux_19510101-20210101 --ensemble-members 0 --model-run-dir /home/ecme4254/hpcperm/model_runs;

# python -m notebooks.process_model_run --experiment-id n3.6_ace2_1951_spinupCMIP6_19510101-20210101 --ensemble-members 0 --model-run-dir /home/ecme4254/hpcperm/model_runs;
