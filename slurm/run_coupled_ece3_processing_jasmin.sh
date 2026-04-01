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

python -m notebooks.process_ece_data --ece3-experiment-id EC-Earth3_piControl --output-ece3-experiment-id EC-Earth3_spinup --years 1951-2049 --month-lag-max 1 --ece3-data-dir /gws/nopw/j04/iecdt/bantonio/EC-Earth3_spinup --ace2-data-dir /home/users/bantonio --base-output-dir /gws/nopw/j04/iecdt/bantonio/processed_spinup_data --analysis-vars thetao sithick tas tos sos so siconc zos --var-glob-string "*/{var}/*/*";

# python -m notebooks.process_ece_data --ece3-experiment-id EC-Earth3_piControl --output-ece3-experiment-id ace2-nemo-40yr-spinup-eval --years 1951-2014 --month-lag-max 1 --ece3-data-dir /gws/nopw/j04/iecdt/bantonio/ace2-nemo-40yr-spinup-eval --ace2-data-dir /home/users/bantonio --base-output-dir /gws/nopw/j04/iecdt/bantonio/processed_spinup_data --analysis-vars thetao sithick tas tos sos so siconc zos --var-glob-string "*/{var}/*/*";

# python -m notebooks.process_ece_data --ece3-experiment-id EC-Earth3_piControl --output-ece3-experiment-id ace2-nemo-70yr-spinup-eval --years 1951-1956 --month-lag-max 1 --ece3-data-dir /gws/nopw/j04/iecdt/bantonio/ace2-nemo-70yr-spinup-eval --ace2-data-dir /home/users/bantonio --base-output-dir /gws/nopw/j04/iecdt/bantonio/processed_spinup_data --analysis-vars sithick tos sos siconc zos --var-glob-string "*/{var}/*/*";


