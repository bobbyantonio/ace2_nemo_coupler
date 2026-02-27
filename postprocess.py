#!/usr/bin/env python3
import os
import gc
import logging
import time
import polling2
import f90nml
from tqdm import tqdm
from collections import OrderedDict
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import xarray as xr
from argparse import ArgumentParser

var_frequency_lookup = {'gencast': {'D': {'vars': [
                                    "2m_temperature",
                                    "mean_sea_level_pressure",
                                    "10m_v_component_of_wind",
                                    "10m_u_component_of_wind",
                                    "total_precipitation_12hr",
                                    "geopotential",
                                    ], 'levels': [500]},
                                    'MS': {'vars':'all', 'levels': 'all'}},
                        'ace2': {'D': {'vars': [
                                    'TMP2m', 
                                    'LHTFLsfc',
                                    'SHTFLsfc'
                                    ], 'levels': 'all'},
                                'MS': {'vars':'all', 'levels': 'all'}},
                       'atm2oce': {'D': {'vars': [
                                                'solid_precipitation',
                                                'liquid_precipitation'
                                                ]}, 
                                   'MS': {'vars': ['mean_surface_sensible_heat_flux',
                                                'mean_surface_latent_heat_flux',
                                                'evaporation',
                                                'instantaneous_eastward_turbulent_surface_stress',
                                                'instantaneous_northward_turbulent_surface_stress',
                                                'evaporation_ice',
                                                'momentum_flux_over_ice_x',
                                                'momentum_flux_over_ice_y',
                                                'solar_flux_over_ice',
                                                'sensible_heat_flux_ice',
                                                'latent_heat_flux_ice',
                                                'net_long_wave_radiation_flux_ice',
                                                'total_non_solar_flux_ice',
                                                'solid_precipitation',
                                                'liquid_precipitation',
                                                'latent_heat_of_vaporization',
                                                'mean_surface_upward_long_wave_radiation_flux',
                                                'mean_surface_downward_long_wave_radiation_flux',
                                                'mean_surface_downward_short_wave_radiation_flux',
                                                'mean_surface_upward_short_wave_radiation_flux']}},
                       'oce2atm': {'D': {'vars':['sea_surface_temperature',
                                                    ]}, 
                                   'MS': {'vars': ['sea_surface_temperature',
                                                    'sea_ice_temperature',
                                                    'ice_albedo',
                                                    'sea_ice_fraction',
                                                    'sea_ice_thickness',
                                                    'ocean_current_u',
                                                    'ocean_current_v',
                                                    'ice_velocity_u',
                                                    'ice_velocity_v']}}}

logger = logging.getLogger(__name__)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--results-data-directory', type=str, required=True)
    parser.add_argument('--model-directory', type=str, required=True)
    parser.add_argument('--atmosphere-source', type=str, choices=['era5', 'era5-calculated', 'gencast', 'ace2', 'ace2-calculated'], required=True)
    parser.add_argument('--router-data-directory', type=str, default=None)
    parser.add_argument('--ocean-source', type=str, choices=['era5', 'nemo'], required=True)
    parser.add_argument('--overwrite', action="store_true",
                        help="overwrite existing files")
    parser.add_argument('--coupling-timestep-secs', type=int, required=True,
                        help='Timestep of coupling in seconds')
    parser.add_argument('--run-in-background', action="store_true",
                        help="run in background mode")
    parser.add_argument('--save-to-zarr', action="store_true",
                        help="save output to zarr format instead of netcdf")
    parser.add_argument('--compression-level', type=int, default=0,
                        help="compression level for netcdf output (1-9), if set to 0 then no compression is applied")
    parser.add_argument('--debug', action="store_true",
                        help="activate debugging")                 
    args = parser.parse_args()
    
    logging.basicConfig(filename=os.path.join(args.model_directory, 'log', 'postprocess.log'), 
                        encoding='utf-8', level=logging.INFO,
                        format='%(asctime)s %(message)s')
    
    
    
    # Read NEMO namelist, to establish start date
    namelist_dict = f90nml.read(os.path.join(args.model_directory, 'namelist_cfg'))
    if namelist_dict == OrderedDict([]):
        # In some cases all of the config is written in the ref namelist
        namelist_dict = f90nml.read(os.path.join(args.model_directory, 'namelist_ref'))
        
    if 'rn_rdt' in namelist_dict['namdom']:
        rndt = namelist_dict['namdom']['rn_rdt']
    else:
        rndt = namelist_dict['namdom']['rn_Dt']

    itend = namelist_dict['namrun']['nn_itend']
    date0 = namelist_dict['namrun']['nn_date0']
    start_datetime = datetime.strptime(str(date0), '%Y%m%d')
    coupling_timestep_s = max(args.coupling_timestep_secs, 21600) # Only sample at 6 hour steps, not at higher frequency
    n_coupling_steps = int(itend * rndt / coupling_timestep_s)
    
    all_dts = [pd.Timestamp(start_datetime + timedelta(seconds = coupling_timestep_s * n)) for n in range(n_coupling_steps)]
    all_hour_diffs = [(dt -start_datetime).total_seconds() / 3600 for dt in all_dts]
    all_yms = [dt.strftime('%Y%m') for dt in all_dts]
    ym_set = set(all_yms)
    
    df = pd.DataFrame({'dts': all_dts, 'yms': all_yms, 'hour_diffs': all_hour_diffs})
    
    #######
    logger.info('Beginning postprocessing')
    
    os.makedirs(args.results_data_directory, exist_ok=True)
    
    data_list = {
        'atm2oce': {'prefix': 'atm2oce', 'suffix': f"_{args.atmosphere_source}_{args.ocean_source}", 'date_format': 'datetime'},
        'oce2atm': {'prefix': 'oce2atm', 'suffix': f"_{args.atmosphere_source}_{args.ocean_source}", 'date_format': 'hour'},
        args.atmosphere_source: {'prefix': args.atmosphere_source, 'suffix': '', 'date_format': 'hour' if args.atmosphere_source in ['ace2', 'ace2-calculated'] else 'datetime'}
    }
    
    for ym in tqdm(sorted(set(all_yms))):
        
        ym_df = df[df['yms'] == ym]
        
        if args.debug:
            ym_df = ym_df.head(10)
        
        ## Collect all data for the year-month

        for str_prefix, data_info in data_list.items():
            
            print('Running for ', str_prefix, ym, flush=True)
            
            monthly_ds = []
            for item in ym_df.itertuples():
                dt = item.dts
                hour_diff = item.hour_diffs

                if data_info['date_format'] == 'datetime':
                    date_str = dt.strftime('%Y%m%d-%H')
                else:
                    date_str = f"{int(hour_diff)}h"
                    
                if str_prefix == args.atmosphere_source and hour_diff == 0:
                    # skip iteration at hour 0 for atmosphere source, as this is not generated
                    continue
                    
                fp = os.path.join(args.router_data_directory, f"{data_info['prefix']}_{date_str}{data_info['suffix']}.nc")
                

                tmp_ds =  polling2.poll(lambda: xr.load_dataset(fp), 
                        ignore_exceptions=(IOError, ValueError, FileNotFoundError), 
                        poll_forever=True if (not args.debug and args.run_in_background) else False,
                        timeout=None if (not args.debug and args.run_in_background) else 1,
                        step=0.1,
                        log=logging.ERROR)

                tmp_ds = tmp_ds.assign_coords(time=np.array([dt]))
                
                if 'time' in tmp_ds.dims:
                    tmp_ds = tmp_ds.isel(time=0)
                monthly_ds.append(tmp_ds)
                
                tmp_ds.close()
                del tmp_ds
                
            monthly_ds = xr.concat(monthly_ds, dim='time', coords='minimal').sortby('time')

            ## Resample and save data
            for resampling_rule, resampling_dict in var_frequency_lookup[str_prefix].items():
                    
                # resample tmp_ds by month start
                output_fp = os.path.join(args.results_data_directory, f"{str_prefix}_{resampling_rule}_{args.atmosphere_source}_{args.ocean_source}_{ym}.nc")
                logger.info(f'Saving {str_prefix} data to {output_fp}')

                if resampling_dict['vars'] == 'all':
                    output_ds = monthly_ds.resample(time=resampling_rule).mean()
                else:
                    output_ds = monthly_ds[resampling_dict['vars']].resample(time=resampling_rule).mean()
                
                # Ensure all variables saved as float32 to save space
                output_ds = output_ds.astype(np.float32)
                
                if not args.debug:    
                    if os.path.exists(output_fp) and args.overwrite:

                        if args.overwrite:
                            os.remove(output_fp)
                            
                        else:
                            logger.info(f'{output_fp} already exists, skipping...')
                            continue
        
                    if args.save_to_zarr:
                        output_ds.to_zarr(output_fp.replace('.nc', '.zarr'))
                    else:
                        if args.compression_level > 0:
                            comp = dict(zlib=True, complevel=args.compression_level)
                            encoding = {var: comp for var in output_ds.data_vars}
                            output_ds.to_netcdf(output_fp, encoding=encoding)
                        else:
                            output_ds.to_netcdf(output_fp)
                    
                    output_ds.close()
                    del output_ds
            monthly_ds.close()
            del monthly_ds
            
            # Explicitly call garbage collector to free up memory
            gc.collect()
        
                
        logger.info('Postprocessing complete')
            
                