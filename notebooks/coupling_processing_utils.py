import os
import gc
from glob import glob
import datetime
from scipy.stats import t
import pandas as pd
import numpy as np
import xarray as xr
from scipy import signal
from scipy.stats import t

mean_areas = {'Global': {'min_lat': -90, 'max_lat': 90},
              'Northern Hemisphere': {'min_lat': 0, 'max_lat': 90},
                    'Southern Hemisphere': {'min_lat': -90, 'max_lat': 0},
                    'Tropics': {'min_lat': -20, 'max_lat': 20},
                    'Northern Extratropics': {'min_lat': 30, 'max_lat': 70},
                    'Southern Extratropics': {'min_lat': -70, 'max_lat': -30},
                    'North Atlantic': {'min_lat': 20, 'max_lat': 60, 'min_lon': 280, 'max_lon': 360},
                    'North Pacific': {'min_lat': 20, 'max_lat': 60, 'min_lon': 160, 'max_lon': 260},
                    'South Atlantic': {'min_lat': -60, 'max_lat': -20, 'min_lon': 300, 'max_lon': 360},
                    'South Pacific': {'min_lat': -60, 'max_lat': -20, 'min_lon': 160, 'max_lon': 260},
                    'Tropical Atlantic': {'min_lat': -20, 'max_lat': 20, 'min_lon': 300, 'max_lon': 360},
                    'Tropical Pacific': {'min_lat': -20, 'max_lat': 20, 'min_lon': 160, 'max_lon': 260}}

ace2_var_lookup = {'TMP2m': '2m_temperature',
                   'surface_temperature': 'surface_temperature', 
                   'PRATEsfc': 'total_precipitation',
                   'PRESsfc': 'surface_pressure',
                   'Q2m': '2m_specific_humidity',
                   'UGRD10m': '10m_u_component_of_wind',
                   'VGRD10m': '10m_v_component_of_wind'
                   }
for n in range(8):
    ace2_var_lookup[f'specific_total_water_{n}'] = f'specific_total_water_{n}'
    
GRAVITY = 9.80665  # m/s^2
AK = [0.0, 5119.90, 13881.3, 19343.5, 20087.1, 15596.7, 8880.45, 3057.27, 0.0]
BK = [0.0, 0.0, 0.00537781, 0.0597284, 0.203491, 0.438391, 0.680643, 0.873929, 1.0]

OLEVEL_VALUES = [2.6676816940307617, 9.822750091552734, 22.75761604309082, 41.180023193359375, 61.11283874511719, 108.03028106689453, 163.16445922851562, 244.890625, 370.6884765625, 565.2922973632812, 773.3682861328125, 1045.854248046875, 1387.376953125, 1795.6707763671875, 2429.025146484375, 3138.56494140625, 4093.15869140625, 5089.478515625, 5902.0576171875]

OLEVEL_BIN_EDGES = [0,100, 500, 1000, 2000,3500, 6000]

def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter
    
def calculate_anomalies(ds):

    time_vals = [pd.Timestamp(dt) for dt in sorted(ds['time'].values)]
    month_vals = ds['time.month'].values
    
    clim_ds = ds.groupby(['time.month', 'latitude', 'longitude']).mean().compute()
    clim_ds = xr.concat([clim_ds.sel(month=month_vals[n]).expand_dims(dim={'time': [time_vals[n]]}) for n in range(len(month_vals))], dim='time').compute()

    anom_ds = ds - clim_ds

    return anom_ds

    

def interface_pressure(surface_pressure: xr.DataArray) -> xr.DataArray:
        """
        Compute pressure at vertical layer interfaces.

        Args:
            surface_pressure: The surface pressure in units of Pa.

        Returns:
            A tensor of pressure at vertical layer interfaces. Will contain a new
            dimension at the end, representing the vertical.
        """
        return xr.concat(
            [ak + bk * surface_pressure for ak, bk in zip(AK, BK)],
            dim='level',
        )


def vertical_integral(
       integrand: xr.DataArray, 
       surface_pressure: xr.DataArray
    ) -> xr.DataArray:
        """
        Compute the mass-weighted vertical integral of the integrand.

        (1 / g) * ∫ x dp

        where
        - g = acceleration due to gravity
        - x = integrand
        - p = pressure level

        Args:
            surface_pressure: The surface pressure in units of Pa.
            integrand: A tensor whose last dimension is the vertical.

        Returns:
            A dataarray of same shape as integrand but without the last dimension.
        """
        if len(AK) != len(integrand['level']) + 1:
            raise ValueError(
                "The 'level' dimension of integrand must match the number of vertical "
                "layers in the hybrid sigma-pressure vertical coordinate."
            )
        interface_pressure_da = interface_pressure(surface_pressure)
        pressure_thickness_da = interface_pressure_da.diff(dim='level')
        return (integrand * pressure_thickness_da).sum(dim='level') / GRAVITY

def calculate_correlation(x: xr.DataArray,
                        y: xr.DataArray,
                        method: str) -> xr.Dataset:

    if 'lat' in x.dims:
        lat_str = 'lat'
        lon_str = 'lon'
    elif 'latitude' in x.dims:
        lat_str = 'latitude'
        lon_str = 'longitude'
    else:
        raise ValueError("Cannot automatically figure out lat and lon names")
        
    x = x.transpose('time', lat_str, lon_str)
    y = y.transpose('time', lat_str, lon_str)
    
    if method not in ['pearson']:
        raise ValueError(f'Unknown method {method}')

    # Adapted from
    # https://hrishichandanpurkar.blogspot.com/2017/09/vectorized-functions-for-correlation.html

    
    xmean = x.mean(axis=0, skipna=True)
    ymean = y.mean(axis=0, skipna=True)
    xstd  = x.std(axis=0, skipna=True, ddof=1)
    ystd  = y.std(axis=0, skipna=True, ddof=1)
    num_non_nans = (~np.isnan(x)).sum(axis=0)

    #4. Compute covariance along time axis
    cov   =  (num_non_nans / (num_non_nans-1)) * ((x - xmean)*(y - ymean)).mean(axis=0, skipna=True)
    cov   =  cov.where(num_non_nans>2) 

    #5. Compute correlation along time axis
    corr   = cov/(xstd*ystd)

    #6. Compute regression slope and intercept:
    slope     = cov/(xstd**2)
    intercept = ymean - xmean*slope  

    #7. Compute P-value and standard error
    #Compute t-statistics for the correlation (i.e. tests the hypothesis that cor/= 0 vs corr=0)
    tstats = np.abs(corr)*np.sqrt(num_non_nans-2)/np.sqrt(1-corr**2)
    stderr = slope/tstats

    pval   = t.sf(tstats, num_non_nans-2)
    pval   = xr.DataArray(pval, dims=corr.dims, coords=corr.coords)
    
    residual = (slope * x + intercept - y)**2
    residual = residual.transpose('time', lat_str, lon_str)
    sum_of_squares = (y - ymean)**2
    r2 = 1 - residual.sum('time')/sum_of_squares.sum('time')
    
    results_dict = {'slope': slope, 
                    'intercept': intercept, 
                    'corr': corr, 'cov': cov, 
                    'tstats': tstats, 
                    'stderr': stderr, 
                    'pval': pval, 
                    'r2': r2}

    results_ds = []
    for k, v in results_dict.items():
        v.name = k
        results_ds.append(v)
    results_ds = xr.merge(results_ds, compat='no_conflicts')

    return results_ds
    
def detrend_dataarray(da, time_dim):
    # Remove climate change signal from the data
    p = da.polyfit(time_dim, 1)
    fitted_vals = xr.polyval(da[time_dim], p.polyfit_coefficients).isel({time_dim: 0})
    
    detrended_data = da - fitted_vals
    return detrended_data, p


def convert_dts_to_first_of_month(ds):
    time_vals = [pd.Timestamp(item) for item in ds['time'].values]
    updated_dts = [datetime.datetime(dt.year, dt.month,1) for dt in time_vals]
    ds = ds.assign_coords({'time': updated_dts})

    return ds

def load_ds_subset(base_dir, glob_filename, vars_to_select, concat_dim='time', decode_times=True):
    fps = glob(os.path.join(base_dir, glob_filename))
    ds = []
    for fp in fps:
        tmp_ds = xr.open_dataset(fp, decode_times=decode_times)[vars_to_select]
  
        ds.append(tmp_ds)
        tmp_ds.close()
        del tmp_ds
        gc.collect()
        
    ds = xr.concat(ds, dim=concat_dim)

    return ds

def load_ece3_data(var, ece3_data_dir, years, ece3_experiment_id, level_values=None, groupby_bins=False):
    ece3_da = []
    for y in years:
        glob_str = os.path.join(ece3_data_dir, f'{var}_*mon_{ece3_experiment_id}_*_{y}01-{y}12.nc')
        fp = glob(glob_str)
        if len(fp) > 1:
            raise IOError(f'More than one file found for var={var}, year={y}')
        elif len(fp) == 0:
            raise IOError(f'No file found for var={var}, year={y}, glob_str={glob_str}')   
            
        fp = fp[0]
        tmp_da = xr.load_dataset(fp)[var]
        if 'lat' in tmp_da.coords:
            tmp_da = tmp_da.rename({'lat': 'latitude', 'lon': 'longitude'})

        if 'lev' in tmp_da.dims and level_values is not None:
            if groupby_bins:
                bin_edges = sorted(level_values)
                bin_labels = [f'{bin_edges[n]}-{bin_edges[n+1]}' for n in range(len(bin_edges)-1)]
                tmp_da = tmp_da.groupby_bins(group='lev', bins=bin_edges, right=True, labels=bin_labels).mean()
            else:
                tmp_da = tmp_da.sel(lev=level_values)
            
        ece3_da.append(tmp_da)
    ece3_da = xr.concat(ece3_da, dim='time')

    return ece3_da
    
def load_era5_monthly(var, era5_dir, years):
    era5_da = []
    for y in years:
        if os.path.isfile(os.path.join(era5_dir, 'surface', var, f'era5M_{var}_year{y}.nc')):
            tmp_da = xr.load_dataarray(os.path.join(era5_dir, 'surface', var, f'era5M_{var}_year{y}.nc'))
            tmp_da.name = var
            era5_da.append(tmp_da)
        else:
            raise ValueError(f"No file found for year={y}, var={var}")
    era5_da = xr.concat(era5_da, dim='valid_time')

    era5_da = era5_da.sortby('latitude').rename({'valid_time':'time'})
    era5_da = convert_dts_to_first_of_month(era5_da)
    return era5_da

def load_nemo_ds_subset(base_dir, 
                        glob_filename, 
                        vars_to_select, 
                        level_values=None, 
                        concat_dim='time',
                        groupby_bins=False,
                        decode_times=True):
    fps = glob(os.path.join(base_dir, glob_filename))
    ds = []
    for fp in sorted(fps):

        date_str = fp.split('/')[-1].split('_')[-1][:6]
        dt = pd.Timestamp(date_str + '01')
        
        tmp_ds = xr.open_dataset(fp, decode_times=decode_times)[vars_to_select]
        tmp_ds = tmp_ds.assign_coords({'time_counter': [dt]})
        tmp_ds = tmp_ds.rename({'time_counter': 'time'}).drop_vars('time_centered')

        if 'olevel' in tmp_ds.dims and level_values is not None:
            if groupby_bins:
                bin_edges = sorted(level_values)
                bin_labels = [f'{bin_edges[n]}-{bin_edges[n+1]}' for n in range(len(bin_edges)-1)]
                tmp_ds = tmp_ds.groupby_bins(group='olevel', bins=bin_edges, right=True, labels=bin_labels).mean()
            else:
                tmp_ds = tmp_ds.sel(olevel=level_values)
        ds.append(tmp_ds)
        tmp_ds.close()
        del tmp_ds
        gc.collect()
        
    ds = xr.concat(ds, dim=concat_dim)

    return ds

def load_oras5_single_level(var, oras5_dir, years):
    
    oras5_da = []
    for y in years:
        fps = glob(os.path.join(oras5_dir, 'single_level', var, f'oras5_{var}_{y}*.nc'))
        if len(fps)>0:
            for fp in fps:
                tmp_da = xr.load_dataarray(fp)
                oras5_da.append(tmp_da)
    oras5_da = xr.concat(oras5_da, dim='time_counter')

    oras5_da.name = var
    oras5_da = oras5_da.rename({'nav_lat': 'latitude', 
                                'nav_lon': 'longitude',
                                'time_counter': 'time'})
    oras5_da['latitude'] = np.round(oras5_da['latitude'], 3)

    return oras5_da


def calculate_lagged_correlations(ds, 
                                  lag_var1, 
                                  lag_var2, 
                                  month_lag_max=1):
    

    # Calculate anomalies
    for var in [lag_var1, lag_var2]:
    
        ds[f'{var}_anom'] = calculate_anomalies(ds[var])

    time_vals = [pd.Timestamp(dt) for dt in ds['time'].values]
    month_vals = ds['time.month'].values
    results_dict = {}
    
    for month_lag in range(-1*month_lag_max, month_lag_max+1):
        
        if month_lag <=-1:
            lagged_dates = time_vals[:month_lag]
            target_dates = time_vals[np.abs(month_lag):]
        elif month_lag >=1:
            lagged_dates = time_vals[month_lag:]
            target_dates = time_vals[:-1*month_lag]
        elif month_lag ==0:
            lagged_dates = time_vals
            target_dates = time_vals
        else:
            raise ValueError('Bad value for month lag')

        assert len(lagged_dates) == len(target_dates)
            
        lag_da = ds[f'{lag_var2}_anom'].sel(time=lagged_dates)
        lag_da = lag_da.assign_coords({'time': target_dates})
    
        corr_results_ds = calculate_correlation(
                x=(lag_da.rename({'latitude': 'lat', 'longitude': 'lon'})),
                y=(ds[f'{lag_var1}_anom'].sel(time=target_dates).rename({'latitude': 'lat', 'longitude': 'lon'})),
                method='pearson')
        results_dict[month_lag] = corr_results_ds

    return results_dict

def calculate_en34(input_sst_da, 
                   remove_seasonal_cycle=True, 
                   rolling_window=None,
                  resolution=1.0):

    return calculate_nino_index(input_sst_da, 
                       remove_seasonal_cycle=remove_seasonal_cycle, 
                       rolling_window=rolling_window,
                         resolution=resolution,
                        nino_region=3.4)


def calculate_nino_index(input_sst_da, 
                           remove_seasonal_cycle=True, 
                           rolling_window=None,
                             resolution=1.0,
                            nino_region=3.4):

    sst_da = input_sst_da.copy()

    if nino_region == 3.4:
        min_lon=190
        max_lon =240
    elif nino_region == 3:
        min_lon=210
        max_lon =270
    elif nino_region == 4:
        min_lon = 160
        max_lon = 210
        
    sst_da = sst_da.sel(latitude=slice(-5,5 + resolution), longitude=slice(min_lon, max_lon+resolution))

    if remove_seasonal_cycle:
        sst_da['month'] = sst_da['time.month']
        month_vals = sst_da['month'].values
        time_vals = sst_da['time'].values
        
        clim_da = sst_da.groupby(['time.month', 'latitude', 'longitude']).mean().compute()
        clim_da = xr.concat([clim_da.sel(month=month_vals[n]).expand_dims(dim={'time': [time_vals[n]]}) for n in range(len(month_vals))], dim='time').compute()
        
        sst_anomaly_da = sst_da - clim_da
        en34_da = sst_anomaly_da.mean(['latitude', 'longitude']).sortby('time')
    else:
        en34_da = sst_da.mean(['latitude', 'longitude']).sortby('time')

    if rolling_window is not None:
        en34_da = en34_da.rolling(time=rolling_window, center=False, min_periods=rolling_window-1).mean().dropna("time")

    return en34_da

def calculate_en34_spectra(da, fs = 12):

    nperseg = np.min([40*12, len(da['time'].values)])
    
    nino34_series =  da.sortby('time')
    f, Pxx = signal.welch(nino34_series, fs=fs, nperseg=nperseg, detrend='linear')

    return f, Pxx
    

def calculate_linear_relationship(x,y):

    
    
    xmean = x.mean(axis=0, skipna=True)
    ymean = y.mean(axis=0, skipna=True)
    xstd  = x.std(axis=0, skipna=True, ddof=1)
    ystd  = y.std(axis=0, skipna=True, ddof=1)
    num_non_nans = (~np.isnan(x)).sum(axis=0)
    
    #4. Compute covariance along time axis
    cov   =  (num_non_nans / (num_non_nans-1)) * ((x - xmean)*(y - ymean)).mean(axis=0, skipna=True)
    cov   =  cov.where(num_non_nans>2) 
    
    #5. Compute correlation along time axis
    corr   = cov/(xstd*ystd)
    
    #6. Compute regression slope and intercept:
    slope     = cov/(xstd**2)
    intercept = ymean - xmean*slope
    
    #7. Compute P-value and standard error
    #Compute t-statistics for the correlation (i.e. tests the hypothesis that cor/= 0 vs corr=0)
    tstats = np.abs(corr)*np.sqrt(num_non_nans-2)/np.sqrt(1-corr**2)
    stderr = slope/tstats
    
    pval   = t.sf(tstats, num_non_nans-2)
    pval_da   = xr.DataArray(pval, dims=corr.dims, coords=corr.coords)
    pvalue_not_significant = pval_da >= 0.025

    corr.name = 'correlation'
    cov.name = 'covariance'
    pval_da.name = 'pvalue'
    slope.name = 'slope'
    intercept.name = 'intercept'
    output_ds = xr.merge([corr, cov, slope, pval_da, intercept], compat='no_conflicts')

    return output_ds


def bjerknes_feedback_analysis(ds):
    
    enso_vars_ds = ds[['sea_surface_temperature', 
                       'mean_surface_latent_heat_flux',
                       'total_precipitation_daily', 
                       'surface_pressure', 
                       '10m_u_component_of_wind']].sel(longitude=slice(130, 250), latitude=slice(-15,15)).transpose('time', 'latitude', 'longitude').copy()
    
    anomaly_ds = calculate_anomalies(enso_vars_ds)

    for var in ['sea_surface_temperature', 'total_precipitation_daily', 'surface_pressure', 'mean_surface_latent_heat_flux']:
    
        anomaly_ds[f'{var}_gradient'] = anomaly_ds[var].sel(longitude=slice(220, 250), 
                                                            latitude=slice(-5,5)).mean(['longitude', 'latitude']) - anomaly_ds[var].sel(longitude=slice(130, 160), latitude=slice(-5,5)).mean(['longitude', 'latitude'])
        anomaly_ds[f'{var}_gradient'] = anomaly_ds[f'{var}_gradient'] / ( ( 235 - 145) * 111.32 * 1000) # Result is in K/m

    anomaly_ds['10m_u_component_of_wind_area_avg'] = anomaly_ds['10m_u_component_of_wind'].sel(latitude=slice(-5,5)).mean(['longitude', 'latitude'])
    
    ###########
    results_dict = {}
    for comparison_vars in [
                            ['sea_surface_temperature_gradient', '10m_u_component_of_wind'],
                            ['sea_surface_temperature_gradient', 'total_precipitation_daily_gradient'],
                            ['sea_surface_temperature_gradient', 'total_precipitation_daily'],
                            ['sea_surface_temperature_gradient', 'mean_surface_latent_heat_flux_gradient'],
                            ['sea_surface_temperature_gradient', 'mean_surface_latent_heat_flux'],
                            ['total_precipitation_daily_gradient', 'surface_pressure_gradient'],
                            ['surface_pressure_gradient', '10m_u_component_of_wind_area_avg'],
                            ['surface_pressure_gradient', '10m_u_component_of_wind'],
                            ['surface_pressure_gradient', 'total_precipitation_daily'],
                            ['surface_pressure_gradient', 'mean_surface_latent_heat_flux']
                           ]:
    
        cvar1 = comparison_vars[0]
        cvar2 = comparison_vars[1]
    
        results_dict[f'{cvar1}__{cvar2}'] = calculate_linear_relationship(anomaly_ds[cvar1], anomaly_ds[cvar2])
        results_dict[f'{cvar2}__{cvar1}'] = calculate_linear_relationship(anomaly_ds[cvar2], anomaly_ds[cvar1])
        
    return results_dict, anomaly_ds