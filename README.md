# ace2_nemo_coupler
Code for coupling the ACE2 atmospheric emulator with the NEMO ocean model


Creating the python environment:
```
conda env create -f environment.yml;

cd $PERM/repos/rdy2cpl
pip install .

cd $PERM/repos/AirSeaFluxCode
pip install .
```

Or just do this on slurm using `slurm/create_env.sh`.


Run using the following command
```
nohup python -m python_scripts.setup_and_run --src-dir [ECEARTH4_DIR]/sources --config-file  [PATH_TO_CONFIG_FILE] --experiment-nickname my_experiment --num-months-per-leg 120 --ensemble-member 0 --start-at-leg 0 --compression-level 8 > output.log 2>&1 &
```