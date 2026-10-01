import os
import numpy as np
import xarray as xr
from tqdm import tqdm
from argparse import ArgumentParser



if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument('--output-dir', type=str, required=True,
                        help="Folder to save data to")
    parser.add_argument('--days', nargs='+', default=range(1,367),
                    help='Day of year values to collect data for. Defaults to all 366 values')
    parser.add_argument('--vars', nargs='+', default=None,
                    help='Specific variables to collect data for')  
    parser.add_argument('--resolution', type=float, default=1.0,)
    parser.add_argument('--force-overwrite', action='store_true',
                        help='Force overwrite of existing data')
    args = parser.parse_args()
    
    latitude_vals = np.arange(-90, 90 + args.resolution, args.resolution)
    longitude_vals = np.arange(0, 360, args.resolution)   
    
    ds = xr.open_zarr('gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_1440x721.zarr').sel(latitude=latitude_vals, longitude=longitude_vals)
        
    for var in args.vars:
        for doy in args.days:
            print(f'** Fetching data for var={var}, dayofyear={doy}', flush=True)

            output_subdir = os.path.join(args.output_dir, var)
            os.makedirs(output_subdir, exist_ok=True)
            output_fp = os.path.join(output_subdir, f"era5_clim_{var}_{doy}.nc")

            ds[var].sel(dayofyear=doy).to_netcdf(output_fp)