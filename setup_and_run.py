import os, sys
import shutil
import yaml
import time
import numpy as np
import pandas as pd
import xarray as xr
import subprocess
import logging
import polling2
from pathlib import Path

from glob import glob
from argparse import ArgumentParser

import scriptengine.helpers.terminal_colors
from scriptengine.cli.se import main, parse_files
from scriptengine.engines import SimpleScriptEngine

scriptengine.helpers.terminal_colors.set_theme(
    "standard"
)

scriptengine.logging.configure(
    logging.DEBUG
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

def get_slurm_job_status(job_id):
    
    status = subprocess.run(['sacct', '-u', os.environ['USER'], '--format', 'State', '--jobs', str(job_id), '--noheader', '-X'], capture_output=True).stdout
    
    if status is None or status.decode().replace('\n', '').strip() == '':
        return 'NOT_FOUND'
    else:
        status = status.decode().replace('\n', '').strip()
    return status


def get_experiment_id(atmosphere_source, 
                      flux_calculation, 
                      start_date, 
                      end_date, 
                      ensemble_member, 
                      experiment_nickname,
                      nemo_version):
    
    if atmosphere_source == 'ace2-calculated':
        exp_id_prefix=f'n{nemo_version}_{atmosphere_source}-{flux_calculation}'
    else:
        exp_id_prefix=f'n{nemo_version}_{atmosphere_source}'
        
    exp_id_prefix = f"{exp_id_prefix}{ f'_{experiment_nickname}' if experiment_nickname != '' else '' }_{start_date}-{end_date}"

    expid=f'{exp_id_prefix}_m{ensemble_member}'
    
    return expid

if __name__ == "__main__":
    
    # Input arguments
    parser = ArgumentParser(description="Setup and run NEMO experiments")
    parser.add_argument("--config-file", type=str, help="Config file containing experiment parameters")
    parser.add_argument("--experiment-nickname", type=str, help="Nickname for the experiment", default='')
    parser.add_argument("--ensemble-member", type=int, help="Ensemble member index (for parallel runs)", default=0)
    parser.add_argument("--num-months-per-leg", type=int, help="Number of months for each leg of the simulation", default=12)
    parser.add_argument("--start-at-leg", type=int, help="Leg index to start from (0-indexed)", default=0)
    parser.add_argument("--save-to-zarr", action="store_true",
                        help="save output to zarr format instead of netcdf")
    parser.add_argument('--compression-level', type=int, default=0,
                        help='Compression level for netcdf output (1-9)')
    parser.add_argument('--ecearth-dir', type=str, default=str(Path(__file__).resolve().parent.parent),
                        help='Path to the ecearth source directory. Defaults to the parent directory of this script.')
    parser.add_argument('--src-dir', type=str, default=None,
                        help='Path to nemo and other packages.')
    parser.add_argument('--setup-only', action='store_true',
                        help='Only setup the experiment, do not run it.')
    args = parser.parse_args()
    
    print(args)
    
    if args.src_dir is None:
        args.src_dir = os.path.join(args.ecearth_dir, 'sources')
        
    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Define experiment parameters
    start_year=config["start_year"]
    end_year=config["end_year"]
    start_month=config["start_month"]
    end_month=config["end_month"]
    nemo_version=config["nemo_version"]
    nemo_config_source = config.get("nemo_config_source", None) 
    atmosphere_source=config["atmosphere_source"]
    frequency=config['frequency'].upper()
    atmosphere_model_dir=config.get('atmosphere_model_dir', None)  # Optional, only needed for some atmosphere sources
    atmosphere_config_file=config.get('config_file', None)  # Optional, only needed for some atmosphere sources
    atmosphere_repo_dir=config.get('atmosphere_repo_dir', None)  # Optional, only needed for some atmosphere sources
    era5_dir=config['era5_dir']  # Directory containing ERA5 data for flux calculations and/or climatology, depending on atmosphere source
    coupling_timestep_s=config['coupling_timestep_secs']  # Default to 6-hourly coupling if not specified
    forcing_data_dir = config['forcing_data_dir']  # Optional, only needed for ACE2 atmosphere source
    first_step_polling_timeout = config.get('first_step_polling_timeout', 12*3600)  # Timeout in seconds for polling ocean model output files on first step
    from_restart = config['from_restart']
    
    ece_script_dir=os.path.join(args.ecearth_dir, 'scripts')
    
    global_start_date=f'{start_year}{start_month:02d}01'
    global_end_date=f'{end_year}{end_month:02d}01'
    num_days_in_leg = (pd.Timestamp(global_end_date) - pd.Timestamp(global_start_date)).days 

    all_dates = pd.date_range(start=global_start_date, end=global_end_date, freq='D')
    all_yms = sorted(set([dt.strftime('%Y%m') for dt in all_dates]))
    
    if len(all_yms) < args.num_months_per_leg != 0:
        raise ValueError(f'Number of months per leg {args.num_months_per_leg} is greater than total number of months in simulation {len(all_yms)}')
    
    legs = [all_yms[n*args.num_months_per_leg:n*args.num_months_per_leg + args.num_months_per_leg + 1] for n in range(int(np.round(len(all_yms)/args.num_months_per_leg)))]
    if args.num_months_per_leg == 1:
        legs = legs[:-1]
        
    if atmosphere_source == 'ace2-calculated':
        atmosphere_source = 'ace2'  # Use ace2 as the source, but with calculated fluxes
        atmosphere_fullname = 'ace2-calculated'
        flux_calculation = 'calculated'
    else:
        flux_calculation = 'standard'
        atmosphere_fullname = atmosphere_source
        
    
    global_expid = get_experiment_id(atmosphere_source, flux_calculation, global_start_date, global_end_date, args.ensemble_member, args.experiment_nickname, nemo_version)
            
    for n, leg in enumerate(legs):
        

        start_date = f'{leg[0]}01'
        end_date = f'{leg[-1]}01'
        num_days_in_leg = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days 
        interval=(pd.Timestamp(end_date).to_period('M') - pd.Timestamp(start_date).to_period('M')).n
        
        expid = get_experiment_id(atmosphere_source, flux_calculation, start_date, end_date, args.ensemble_member, args.experiment_nickname, nemo_version)

        rundir=f'/ec/res4/scratch/ecme4254/run_dir/{expid}'
        results_dir=f'/ec/res4/hpcperm/ecme4254/model_runs/{global_expid}'
        router_dir=f'{rundir}/router'        

        if n < args.start_at_leg:
            # Skip this leg,
            # Set variables for next iteration
            previous_expid = expid
            previous_start_date = start_date
            previous_end_date = end_date
            previous_rundir = rundir
            previous_router = router_dir
            previous_num_days_in_leg = num_days_in_leg
            
            logger.info(f'Skipping leg {n} of {len(legs)}: {leg[0]} to {leg[-1]}')
        
            continue
        

        logger.info(f'Setting up leg {n} of {len(legs)}: {leg[0]} to {leg[-1]}')
                
        yaml_content=f"""
- base.context:
    experiment:
      id: {expid}
      schedule:
        all: !rrule >
          DTSTART:{start_date}
          RRULE:FREQ={frequency};INTERVAL={interval};UNTIL={end_date}
      flux_calculation: {flux_calculation}
      era5_dir: "{era5_dir}"
      base_dir: "{args.ecearth_dir}"
      src_dir: "{args.src_dir}"
      run_dir: "{rundir}"
      results_dir: "{results_dir}"
      run_from_scratch: true
      nemo:
        start_from: 
            restart: { 'true' if (from_restart or n > 0) else 'false' }  
            ts_state: { 'false' if (from_restart or n > 0) else 'true' }
        config_name: { nemo_config_source if nemo_config_source is not None else 'null' }
      initialise_atmosphere_from_era5: { 'true' if n==0 else 'false' }
      nemo_coupling_frequency: {coupling_timestep_s}
    """ 
        logger.info(yaml_content)
        
        #####################################################
        # Load base config file for this atmosphere source / nemo version       
        configpath=os.path.join(ece_script_dir, 'runtime', f"nemo{nemo_version}-{atmosphere_source}-6hr-coupled-config_noid.yml")
        os.chdir(os.path.join(ece_script_dir, 'runtime'))
        
        # Write the config file to the ece script dir, as it needs to be read by ScriptEngine
        tmp_yaml_filename = os.path.join(ece_script_dir, 'runtime', f'experiment_{expid}.yaml')
        with open(tmp_yaml_filename, 'w+') as tmp_yaml_file:
            tmp_yaml_file.write(yaml_content)

        ######################################################
        ## Run setup using ScriptEngine
        ######################################################
        
        logger = logging.getLogger("se.cli")
        logger.info("Logging configured and started")

        files = [
                    tmp_yaml_filename,
                    os.path.join(ece_script_dir, "platforms/ecmwf-hpc2020-intel+openmpi.yml"), 
                    configpath, 
                    os.path.join(ece_script_dir, 'runtime', "scriptlib/basic-setup-only.yml")]
                            
        script = parse_files(logger, files)

        script_path = tuple(
            dict.fromkeys((os.path.dirname(file) for file in files))
        )
        
        context = {
            "se": {
                "cli": {
                    "cwd": os.getcwd(),
                    "script_path": script_path,
                },
                "tasks": {
                    "timing": {
                        "mode": None,
                        "logging": None,
                        "timers": {},
                    },
                },
                "instance": SimpleScriptEngine(),
            }
        }

        # Call ScriptEngine instance to run the script
        context["se"]["instance"].run(script, context)
        
        # Copy config files to run directory
        for file in files:
            shutil.copy(file, rundir)
        shutil.copy(args.config_file, rundir)
            
        # Remove all pkl and nc files from router directory
        for file in os.listdir(router_dir):
            if file.endswith('.pkl') or file.endswith('.nc'):
                os.remove(os.path.join(router_dir, file))

        if args.setup_only:
            logger.info('Setup only flag set, skipping run')
            sys.exit(0)
        
        if n > 0:
            ######################################################
            # Rearrange restart files if not the first chunk
            ######################################################
            # Note: assumes that ML atmosphere files are already written to the run directory
            
            nemo_restart_files = glob(os.path.join(previous_rundir, f'{previous_expid}_*_restart_oce_*.nc')) + glob(os.path.join(previous_rundir, f'{previous_expid}_*_restart_ice_*.nc'))
            logger.debug(f'Moving {len(nemo_restart_files)} NEMO restart files from previous run')
            
            for file in nemo_restart_files:
                shutil.copy(file, os.path.join(rundir, '_'.join(file.split('_')[-3:])))
                
            # Remove redundant restart_oce.nc files (only added if the restart file exists for the start date)
            if os.path.exists(os.path.join(rundir, 'restart_oce.nc')):
                os.remove(os.path.join(rundir, 'restart_oce.nc'))
                
            if os.path.exists(os.path.join(rundir, 'restart_ice.nc')):
                os.remove(os.path.join(rundir, 'restart_ice.nc'))
                
            # Move final ocean file from previous run to be initial condition for this run
            final_oce2atm_fp = os.path.join(previous_router, f'oce2atm_{int(previous_num_days_in_leg*24)}h_{atmosphere_source}_nemo.nc')
            logger.debug(f"Moving final oce2atm file from previous run {final_oce2atm_fp} to {os.path.join(router_dir, f'oce2atm_0h_{atmosphere_source}_nemo.nc')}")
            shutil.copy(final_oce2atm_fp, os.path.join(router_dir, f'restart_oce2atm_0h_{atmosphere_source}_nemo.nc'))
            
            # ML atmosphere restart file
            final_atm_restart_fp = os.path.join(previous_rundir, f'restart/restart_{atmosphere_source}.nc')
            logger.debug(f"Moving final atmosphere restart file from previous run {final_atm_restart_fp} to {os.path.join(rundir, f'restart_{atmosphere_source}.nc')} and {os.path.join(router_dir, f'{atmosphere_source}_0h.nc')}")
            shutil.copy(final_atm_restart_fp, os.path.join(rundir, f'restart_{atmosphere_source}.nc'))
            
            # Also copy final atmosphere file to router directory for use as initial condition
            final_atmosphere_fp = os.path.join(previous_router, f'{atmosphere_source}_{int(previous_num_days_in_leg*24)}h.nc')
            logger.debug(f"Moving final oce2atm file from previous run {final_atmosphere_fp} to {os.path.join(router_dir, f'{atmosphere_source}_0h.nc')}")
            shutil.copy(final_atmosphere_fp, os.path.join(router_dir, f'{atmosphere_source}_0h.nc'))
            
        ########################################################
        ## Submit jobs to SLURM
        ######################################################
        os.chdir(rundir)
        atmosphere_job_filename = f'run_{atmosphere_source}.sh'
        if atmosphere_source == 'ace2':
            
            # For ACE, we run a lagged ensemble, with each member starting on a different day
            # This means we need to create the slurm job on-the-fly to set the start date correctly
            start_day = 1 + args.ensemble_member
            
            atmosphere_slurm_text=f"""#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=50gb
#SBATCH --qos=ng
#SBATCH --gpus=1
#SBATCH --job-name={atmosphere_fullname}-cpl
#SBATCH --output=log/{atmosphere_fullname}-coupled-%A.txt

source ~/.kshrc
source ~/.initConda.sh

conda activate ace

cd {atmosphere_repo_dir}

OCEAN_MODEL_DIR={router_dir}

# Note that start datetime here means first init datetime, not first target datetime like the Gencast code.

python -m run_ace --inference-config {atmosphere_config_file} \
--model-name {atmosphere_fullname} \
--model-dir {atmosphere_model_dir} \
--output-dir {rundir}/restart \
--logging-dir {rundir}/log \
--initial-condition-path {os.path.join(atmosphere_model_dir, 'initial_conditions', f"ic_{leg[0]}{start_day:02d}.nc") if n==0 else os.path.join(rundir, f'restart_ace2.nc') } \
--forcing-data-dir {forcing_data_dir} \
--start-datetime "{leg[0]}{start_day:02d}-00" \
--num-steps-per-initialisation {4*num_days_in_leg} \
--sst-input coupled \
--ocean-model-dir {router_dir} \
--first-step-polling-timeout {first_step_polling_timeout};
""" 
            
            logger.debug(f'Writing postprocess slurm script to {atmosphere_job_filename}')
            with open(os.path.join(rundir, atmosphere_job_filename), 'w+') as atm_job_file:
                atm_job_file.write(atmosphere_slurm_text)

        atmosphere_slurm_return_code = subprocess.run(['sbatch', '--parsable', atmosphere_job_filename], capture_output=True)
        atmosphere_jobid = atmosphere_slurm_return_code.stdout.decode().replace('\n', '')
        
        logger.info(f'Submitted job {atmosphere_jobid} for experiment {expid}')
        
        logger.info('Waiting for Atmosphere job to start running')
        polling2.poll(lambda: get_slurm_job_status(atmosphere_jobid) == 'RUNNING', step=10, poll_forever=True)
        
        logger.info('Waiting a few minutes for atmosphere job to initialise before submitting NEMO job')
        
        if atmosphere_source == 'gencast':
            time.sleep(5*60)
        else:
            time.sleep(3*60)
        
        # Submit NEMO job
        nemo_slurm_return_code = subprocess.run(['sbatch', '--parsable', 'run_nemo_coupled.sh'], capture_output=True)
        nemo_jobid = nemo_slurm_return_code.stdout.decode().replace('\n', '')
        logger.info(f'Submitted NEMO job {nemo_jobid} for experiment {expid}')
        
        time.sleep(1)
        
        ######################################################
        # Run postprocessing of results (in background)
        ######################################################
        os.makedirs(results_dir, exist_ok=True)
        
        postprocess_slurm_text =f"""#!/usr/bin/bash
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --qos=np
#SBATCH --mem=50gb
#SBATCH --output=log/postprocess-{atmosphere_fullname}-m{args.ensemble_member}-%A.txt

source ~/.kshrc
source ~/.initConda.sh
conda activate ece4

cp $PERM/repos/ecearth4/python_scripts/postprocess.py {rundir};

cd {rundir};

# Copy config files to results directory
cp *.yaml {results_dir}/

# Copy log files to results directory
cp log/* {results_dir}

python postprocess.py --model-directory {rundir} --ocean-source nemo --atmosphere-source {atmosphere_fullname} --router-data-directory {router_dir} --results-data-directory {results_dir} --overwrite --coupling-timestep-secs {coupling_timestep_s} --run-in-background{' --save-to-zarr' if args.save_to_zarr else ''} --compression-level {args.compression_level};"""

        logger.debug(postprocess_slurm_text)

        tmp_postprocess_filename = f'postprocess_{expid}.sh'
        logger.debug(f'Writing postprocess slurm script to {tmp_postprocess_filename}')
        with open(os.path.join(rundir, tmp_postprocess_filename), 'w+') as tmp_postprocess_file:
            tmp_postprocess_file.write(postprocess_slurm_text)

        postprocess_slurm_return_code = subprocess.run(['sbatch', '--parsable', tmp_postprocess_filename], capture_output=True)
        logger.debug(str(postprocess_slurm_return_code))
        
        postprocess_jobid = postprocess_slurm_return_code.stdout.decode().replace('\n', '')

        logger.info(f'Submitted postprocessing job {postprocess_jobid} for experiment {expid}')
        
        ######################################################
        # Wait for NEMO job to finish
        ######################################################

        polling2.poll(lambda: get_slurm_job_status(nemo_jobid) not in ['RUNNING', 'PENDING', 'NOT_FOUND'], step=1, poll_forever=True)
        
        # Check if atmosphere job has failed; if so raise an error and cancel postprocessing job
        atmosphere_job_status = get_slurm_job_status(atmosphere_jobid)
        if atmosphere_job_status != 'COMPLETED':
            logger.info(f'Terminating postprocessing job {postprocess_jobid} as atmosphere job has failed')
            subprocess.run(['scancel', postprocess_jobid])
            
            # if atmosphere_job_status == 'RUNNING':
            #     logger.info(f'Atmosphere job {atmosphere_jobid} is still running, cancelling it now')
            #     subprocess.run(['scancel', atmosphere_jobid])
            # else:
            #     raise RuntimeError(f'Atmosphere job {atmosphere_jobid} ended with status {atmosphere_job_status}')
        
        
        ######################################################
        # Copy NEMO output files to results directory
        ######################################################
            
        logger.info('Copying NEMO files across to results directory')
        nemo_results_dir = os.path.join(results_dir, f'nemo_output_{atmosphere_fullname}')
        os.makedirs(nemo_results_dir, exist_ok=True)
        
        nemo_output_files = glob(os.path.join(rundir, 'nemo_ocean_output_*.nc')) + glob(os.path.join(rundir, 'lim_output_icemod_*.nc'))
        for file in nemo_output_files:
            ds = xr.open_dataset(file)
            ds = ds.astype(np.float32)
            
            if args.save_to_zarr:
                # Convert to zarr format
                
                zarr_fp = os.path.join(nemo_results_dir, file.split('/')[-1].replace('.nc', '.zarr'))
                ds.to_zarr(zarr_fp)
                ds.close()
            else:
                fp = os.path.join(nemo_results_dir, file.split('/')[-1])
                if args.compression_level != 0:
                    comp = dict(zlib=True, complevel=args.compression_level)
                    encoding = {var: comp for var in ds.data_vars if not var in ['time_bnds', 'time_centered_bounds', 'time_counter_bounds']}
                    ds.to_netcdf(fp, encoding=encoding)
                    ds.close()
                else:
                    ds.to_netcdf(fp)
                    ds.close()

        os.remove(tmp_yaml_filename)
        

        # Set variables for next iteration
        previous_expid = expid
        previous_start_date = start_date
        previous_end_date = end_date
        previous_rundir = rundir
        previous_router = router_dir
        previous_num_days_in_leg = num_days_in_leg
        
    
    # After final leg, copy all restart files to a restart directory
    restart_dir = os.path.join(results_dir, 'final_restart_files')
    os.makedirs(restart_dir, exist_ok=True)
    final_nemo_restart_files = glob(os.path.join(previous_rundir, f'{previous_expid}_*_restart_oce_*.nc')) + glob(os.path.join(previous_rundir, f'{previous_expid}_*_restart_ice_*.nc'))
    logger.info(f'Copying {len(final_nemo_restart_files)} NEMO restart files to final restart directory')  
    
    for file in final_nemo_restart_files:
        shutil.copy(file, restart_dir)
        
    final_atm_restart_fp = os.path.join(previous_rundir, f'restart/restart_{atmosphere_source}.nc')
    logger.info(f"Moving final atmosphere restart file from previous run {final_atm_restart_fp} to {restart_dir}")
    shutil.copy(final_atm_restart_fp, restart_dir)
    
    # Copy all configs to a configs directory
    config_dir = os.path.join(results_dir, 'configs')
    os.makedirs(config_dir, exist_ok=True)
    
    config_files = list(set(glob(os.path.join(previous_rundir, "{*.yaml,*.yml,nam*,*.xml}"))))
    for file in config_files:
        shutil.copy(file, config_dir)
    
