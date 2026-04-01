# ace2_nemo_coupler
Code for coupling the ACE2 atmospheric emulator with the NEMO ocean model and analysing the output, compared to EC-Earth3 and ERA5.


# Creating the Python environment

There are two environment yaml files; environment.yml is for creating the full environment for coupling with NEMO, and environment_analysis.yml is for creating the environment for running the analysis notebooks (a fairly standard setup of xarray, matplotlib, and xesmf for regridding)

## Setup for coupling ACE2 to NEMO

There are some extra dependencies for coupling to NEMO and using the EC-Earth 4 infrastructure. 

Requires a forked version of EC-Earth 4; clone the repository at https://git.smhi.se/e8118/ecearth4 (requires first registering with the EC-Earth 4 project; see https://ec-earth-4-docs.readthedocs.io/en/latest/)

Creating the python environment:
```
conda env create -f environment.yml;


cd $PERM/repos/rdy2cpl
pip install .

cd $PERM/repos/AirSeaFluxCode
pip install .
```

Or just do this on slurm using `slurm/create_env.sh`.

Clone the modified ACE2 repo: 
```
git clone https://github.com/bobbyantonio/ace
```
and make sure to amend the `atmosphere_repo_dir` field in your config when running setup_and_run.

## Running the coupled model

First you need to create a config file. Examples can be found in this repo, in the configs/ folder. Example for a run starting from restart files:
```
start_year: 1951
end_year: 2021
start_month: 1
end_month: 1
nemo_version: 3.6
atmosphere_source: ace2
frequency: monthly
atmosphere_model_dir: /home/ml_model_data/ace2
config_file: /home/repos/ace/configs/coupled_inference_config.yaml
atmosphere_repo_dir: /home/repos/ace
era5_dir: /hpcperm/ecme4254/era5
ece_script_dir: /perm/ecme4254/repos/ecearth4/scripts
coupling_timestep_secs: 21600
forcing_data_dir: /home/ecme4254/scratch/ace2_forcing_data/control_1951-2051
from_restart: true
```

Run using the following command
```
nohup python -m setup_and_run --ecearth-dir [ECEARTH4_DIR] [--src-dir /path/to/ecearth/sources] --config-file [PATH_TO_CONFIG_FILE] --experiment-nickname my_experiment --num-months-per-leg 120 --ensemble-member 0 --start-at-leg 0 --compression-level 8 > output.log 2>&1 &
```
--ecearth-dir specifies where you have cloned the EC-Earth4 repo to, from https://git.smhi.se/e8118/ecearth4.

The --src-dir points to a folder containing all of the sources for EC-Earth 4 (NEMO, OASIS, XIOS,...). If left unspecified this will default to ECEARTH4_DIR/sources.



## Jupyter notebooks

This requires jupytext (https://jupytext.readthedocs.io/en/latest/) to be installed:
```
conda install -c conda-forge jupytext
```

Each directory contains base .py files for different projects, from which notebooks can be created using jupytext (only the .py files are committed to version control, to avoid commiting figures and data that cause the repo to be bloated, and to avoid messy merge conflicts when you change figures).

To create paired notebooks, run e.g.
```
jupytext --set-formats ipynb,py:percent notebooks/process_ece_data.py
``` 

So sync all notebooks in a folder their paired .py files, run e.g.
```
jupytext --sync notebooks/*.py
```
This will take the changes from the most recently updated ipynb / py file and copy those to the paired file. For this reason, the suggested workflow is to only edit notebooks, and when it is time to commit then run the sync command

## Existing raw data from ACE2-NEMO runs

So far we have control and historical runs, both with 3 ensemble members each (created using a lagged ensemble)

The three ensemble members from the control run (mostly monthly but some daily data) are in these folders:
```
/home/ecme4254/perm/old_model_runs/n3.6_ace2_1951_control_compressed_19510101-20210101_m0
/home/ecme4254/perm/old_model_runs/n3.6_ace2_1951_control_compressed_19510101-20210101_m1
/home/ecme4254/perm/old_model_runs/n3.6_ace2_1951_control_compressed_19510101-20210101_m2
```
Where m0, m1, m2 refer to the ensemble member number.

The historical run data (mostly monthly but some daily data) are in these folders:
```
/home/ecme4254/hpcperm/model_runs/n3.6_ace2_1951-2021_hist_compressed_19510101-20210101_m0
/home/ecme4254/hpcperm/model_runs/n3.6_ace2_1951-2021_hist_compressed_19510101-20210101_m1
/home/ecme4254/hpcperm/model_runs/n3.6_ace2_1951-2021_hist_compressed_19510101-20210101_m2
```

## Processing the data for plotting

Existing processed data can be found in the Zenodo repository https://zenodo.org/records/19187005.

In order to process the ACE2-NEMO, EC-Earth3 and ERA5 data, there are python scripts in notebooks/ to do this; `notebooks/process_model_run.py` for the ACE2-NEMO runs, `notebooks/process_ece_data.py` for the EC-Earth3 data, and `notebooks\process_reanalysis.py` for the ERA5 reanalysis. 

These scripts have been generated from Jupyter notebooks using jupytext (See above), so if you want to add extra analysis or debug the code, you can use e.g. `jupytext --sync notebooks/process_model_run.py` to create the notebook. Then once you have made changes to the notebook (taking care to delete any unnecessary cells), sync the changes back to the Python file using `jupytext --sync notebooks/process_model_run.py` again.

There are slurm scripts created to run the processing for each data type: `slurm/run_coupled_processing.sh` for the ACE2-NEMO data, `slurm/run_coupled_ece3_processing.sh` for EC-Earth3 data, and `slurm/run_coupled_reanalysis_processing.sh` for ERA5 reanalysis.

`slurm/run_coupled_ece3_processing_jasmin.sh` was written to run the ece3 processing on JASMIN (for the spin-up experiment primarily)


## Downloading EC-Earth3 data

The EC-Earth data from the PRIMAVERA experiments is available from CEDA. There is a script to download this in `data_download/download_ece2_data.sh`.

The historical data used in the ACE2-NEMO experiment was provided by colleagues from BSC, and can be made available on reasonable request.

## ACE2 forcing and other data

A bash script for downloading the ACE2 historical forcing data can be found at `data_download/download_ace2_forcing_data.sh`. In order to create the 'control' (constant 1950s) forcing, use the `notebooks/create_ace2_forcing_files.py` script (slurm script for this at `slurm/create_ace2_forcing_files.sh`).

Grid files for ACE2 (required for the coupling and for plotting) can be found in the Zenodo repository https://zenodo.org/records/19187005