#!/usr/bin/env python3
import os, sys
import time
import datetime
import f90nml
import polling2
import pickle
import AirSeaFluxCode
import numpy as np
import pandas as pd

from codetiming import Timer
from collections import OrderedDict
import xarray as xr
import xarray_regrid
from unittest.mock import MagicMock, Mock
from argparse import ArgumentParser
from pickle import UnpicklingError
from scipy.ndimage import uniform_filter

import logging
logger = logging.getLogger(__name__)
# logging.basicConfig(format='%(asctime)s %(message)s')

#TODO: incorporate gustiness contribution in momentum fluxes

ERA5_DIR = '/ec/res4/hpcperm/ecme4254/era5'

FIRST_POLL_TIMEOUT = 20 * 60  # 20 minutes
POLLING_TIMEOUT = 60 * 10  # 10 minutes

stefan_boltzmann = 5.67e-8
air_density = 1.22
specific_heat_capacity_air = 1005.0  # J/(kg*K)
C_ice = 1.4e-3  # Heat transfer coefficient for ice, assumed constant
Ls = 2.839e6  # Latent heat of sublimation for ice, J/kg
ocean_albedo = 0.066


era5_var_lookup = {'A_Evap_total': 'evaporation', 
                   'A_Evap_ice': 'evaporation_ice',
                   'A_Qns_ice': 'total_non_solar_flux_ice',
                   'A_TauX_ice': 'momentum_flux_over_ice_x',
                   'A_TauY_ice': 'momentum_flux_over_ice_y',
                   'A_TauX_oce': 'instantaneous_eastward_turbulent_surface_stress',
                   'A_TauY_oce': 'instantaneous_northward_turbulent_surface_stress',
                   'A_Qs_ice': 'solar_flux_over_ice',
                   'A_Qs_oce': 'mean_surface_net_short_wave_radiation_flux'}

zero_vars = ['A_dQns_dT']

ATM2OCE_VARS = [
                # 'A_EvapMPre',
                'A_Evap_total',
                'A_Evap_ice',
                'A_Precip_liquid',
                'A_Precip_solid', 
                'A_Qns_ice' ,
                'A_Qns_oce',
                'A_Qs_oce' ,
                'A_Qs_ice' ,
                'A_TauX_ice',
                'A_TauX_oce',
                'A_TauY_ice',
                'A_TauY_oce',
                'A_dQns_dT']

OCE2ATM_VARS = ['A_SST', 
                'A_Ice_temp',
                'A_Ice_albedo',
                'A_Ice_frac', 
                'A_Ice_thickness',
                'A_Snow_thickness',
                'A_OceCurrent_u',
                'A_OceCurrent_v',
                'A_IceVelocity_u',
                'A_IceVelocity_v',
                ]

def latent_heat_flux_over_ice(atmosphere_ds, ice_ds):
    
    """
    Latent heat flux; note that it is defined as negative when ice is sublimated into the air (because heat is taken from the ice to 
    sublimate). Or in other words positive when latent heat is transferred into the ice.
    """
    
    latent_heat_flux = air_density * Ls * C_ice * atmosphere_ds['relative_wind_speed_ice'] *  (atmosphere_ds['specific_humidity_surface'] -  11637800 * np.exp( -5897.8 / ice_ds['sea_ice_temperature'] ) / air_density  )
    latent_heat_flux = xr.where(latent_heat_flux > 0, 0, latent_heat_flux)  # Set positive fluxes to zero (in line with NEMO 3.6)
    latent_heat_flux.name = 'latent_heat_flux_ice'
    
    return latent_heat_flux

def momentum_flux_over_ice(ds):
    
    taux = air_density * C_ice * ds['relative_wind_speed_ice'] * ds['relative_wind_speed_ice_u']
    tauy = air_density * C_ice * ds['relative_wind_speed_ice'] * ds['relative_wind_speed_ice_v']
    
    taux.name = 'momentum_flux_x_ice'
    tauy.name = 'momentum_flux_y_ice'
    
    return taux, tauy

def sensible_heat_flux_over_ice(atmosphere_ds, ice_ds):
    # Sensible heat flux; note that it is defined as positive when heat is transferred from the air to the ice
    sensible_heat_flux = air_density * specific_heat_capacity_air * C_ice * atmosphere_ds['relative_wind_speed_ice'] * (atmosphere_ds['2m_temperature'] - ice_ds['sea_ice_temperature'])
    
    return sensible_heat_flux

def net_long_wave_flux_over_ice(atmosphere_ds, ice_ds) -> xr.DataArray:
    
    q_long_wave = 0.95 + ( atmosphere_ds['mean_surface_downward_long_wave_radiation_flux'] - stefan_boltzmann * ice_ds['sea_ice_temperature']**4 )
    q_long_wave.name = 'net_long_wave_radiation_flux_ice'

    return q_long_wave

def solar_flux_over_ice(atmosphere_ds, ice_ds) -> xr.DataArray:
    
    total_solar_flux = (1 - ice_ds['ice_albedo']) * atmosphere_ds['mean_surface_downward_short_wave_radiation_flux']
    total_solar_flux.name = 'short_wave_radiation_flux_ice'
    return total_solar_flux

def solar_flux_over_ocean(atmosphere_ds, ice_ds) -> xr.DataArray:
    
    total_solar_flux = (1 - ocean_albedo) * atmosphere_ds['mean_surface_downward_short_wave_radiation_flux']
    
    # To be on the safe side, we also adjust solar flux over ice points
    ice_mask = ice_ds['sea_ice_fraction'] > 0.1
    total_solar_flux = xr.where(ice_mask, solar_flux_over_ice(atmosphere_ds, ice_ds), total_solar_flux)
    total_solar_flux.name = 'short_wave_radiation_flux_ocean'
    
    return total_solar_flux


def non_solar_fluxes_ice(atmosphere_ds, ice_ds, source, clim_ds=None) -> xr.Dataset:
    
    # Net long wave radiation fluxes
    q_long_wave = net_long_wave_flux_over_ice(atmosphere_ds, ice_ds)

    # Turbulent fluxes
    if source in ['gencast']:
        sensible_heat_flux = clim_ds['mean_surface_sensible_heat_flux_clim']
    else:
        sensible_heat_flux = sensible_heat_flux_over_ice(atmosphere_ds, ice_ds)
    sensible_heat_flux.name = 'sensible_heat_flux_ice'

    # Latent heat flux; note that it is defined as positive when water vapor is transferred from the air to the ice
    if source in ['gencast']:
        latent_heat_flux = clim_ds['mean_surface_latent_heat_flux_clim']
    else:
        latent_heat_flux = latent_heat_flux_over_ice(atmosphere_ds, ice_ds)
    latent_heat_flux.name = 'latent_heat_flux_ice'
    
    return xr.merge([
        q_long_wave,
        sensible_heat_flux,
        latent_heat_flux])


def get_era5_ocean_data(dt: datetime.datetime,
                        data_dir: str,
                        atmosphere_grid: str) -> xr.Dataset:
    """
    Get ERA5 ocean data for a given datetime.
    """
    
    
    ocean_ds = []
    for era5_var in ['sea_surface_temperature',
                     'skin_temperature',
                     'sea_ice_cover',
                     'forecast_albedo']:
        tmp_da = xr.load_dataarray(os.path.join(data_dir, 'surface', era5_var, f"era5_{era5_var}_{dt.strftime('%Y%m%d')}.nc")).sel(time=dt)
        tmp_da.name = era5_var
        ocean_ds.append(tmp_da)
    
    for var in ['sea_ice_thickness', 
                'ocean_current_u',
                'ocean_current_v',
                'ice_velocity_u',
                'ice_velocity_v',
                'snow_depth']:
        tmp_da = xr.zeros_like(ocean_ds[0])
        tmp_da.name = var
        ocean_ds.append(tmp_da)
    
    ocean_ds = xr.merge(ocean_ds).rename({'skin_temperature': 'sea_ice_temperature',
                                      'sea_ice_cover': 'sea_ice_fraction',
                                      'forecast_albedo': 'ice_albedo'})
    
    if atmosphere_grid is not None:
        # regrid
        ocean_ds = ocean_ds.regrid.linear(atmosphere_grid)
        
    return ocean_ds

def fluxes_to_oasis_structure(flux_ds: xr.Dataset,
                              ocean_ds: xr.Dataset,
                              latitude_vals: list,
                              longitude_vals: list) -> xr.Dataset:
    
    for var, flux_var in era5_var_lookup.items():
        
        if flux_var in ['evaporation', 'evaporation_ice']:
            # From NEMO documentation: "a positive E implies a freshwater loss for the ocean"
            # From ERA5 documentation: "negative values indicate evaporation and positive values indicate condensation"
            # So we need to reverse the sign of evaporation and evaporation_ice
            flux_ds[var] = -1 * flux_ds[flux_var]
            
        else:
            flux_ds[var] = flux_ds[flux_var]
        
    flux_ds['A_Qns_oce'] = (flux_ds['mean_surface_sensible_heat_flux'] + flux_ds['mean_surface_latent_heat_flux'] + flux_ds['mean_surface_net_long_wave_radiation_flux'])
    flux_ds['A_Qs_ice'] = flux_ds['solar_flux_over_ice']
    sea_ice_frac = ocean_ds['sea_ice_fraction'].fillna(0.0)
    
    # Interpolate between sea points and sea-ice points based on sea ice fraction
    for var in ['A_Qns_oce', 'A_Qs_oce', 'A_TauX_oce', 'A_TauY_oce']:
        flux_ds[var] = sea_ice_frac * flux_ds[var.replace('oce', 'ice')] + (1 - sea_ice_frac) * flux_ds[var]

    flux_ds['A_Evap_total'] = sea_ice_frac * flux_ds['A_Evap_ice'] + (1 - sea_ice_frac) * flux_ds['A_Evap_total']
    
    # Convert precip from m/hour to kg/m^2/s
    flux_ds['A_Precip_liquid'] = flux_ds['total_precipitation'] * 1000 / 3600  
    flux_ds['A_Precip_solid'] = flux_ds['solid_precipitation'] * 1000 / 3600 
    
    flux_ds['A_EvapMPre'] = flux_ds['A_Evap_total'] - flux_ds['total_precipitation']
    
    for var in zero_vars:
        flux_ds[var] = xr.zeros_like(flux_ds['A_Qns_oce'])
        flux_ds[var].name = var
    
    return flux_ds.sel(
                    latitude=latitude_vals, 
                    longitude=longitude_vals
                )


def interpolate_surface_specific_humidity(ds: xr.Dataset):
    """
    Interpolate specific humidity to the surface using geopotential height.
    
    Args:
        ds (xr.Dataset): Dataset containing specific humidity and geopotential height.
    Returns:
        xr.DataArray: Specific humidity at the surface.
    """
    pls = sorted(ds['level'].values)[-2:]

    surface_da = ds['specific_humidity'].sel(level=pls[1]) + (ds['specific_humidity'].sel(level=pls[1]) - ds['specific_humidity'].sel(level=pls[0])) * (0 - ds['geopotential'].sel(level=pls[1])) / (ds['geopotential'].sel(level=pls[1]) - ds['geopotential'].sel(level=pls[0]))

    # Ensure there are no negative values (although even in ERA5 data there are negatives of the order 1e-6)
    surface_da = np.clip(surface_da, a_max=None, a_min=1e-6)
    return surface_da



class FluxCalculator:
    
    def __init__(self,
                 atmosphere_source: str,
                 era5_directory: str,
                 atmosphere_directory: str,
                 start_datetime: datetime.datetime,
                 coupling_timestep_hrs: int,
                 atmospheric_timestep_hrs: int,
                 climatology_ds: xr.Dataset,
                 ocean_source: str,
                 latitude_vals: list,
                 longitude_vals: list,
                 start_from_era5: bool=False):
        
        self.start_datetime = start_datetime
        self.coupling_timestep_hrs = coupling_timestep_hrs
        self.coupling_timestep_s = 3600 * coupling_timestep_hrs
        self.atmospheric_timestep_hrs = atmospheric_timestep_hrs
        self.atmospheric_timestep_s = 3600 * atmospheric_timestep_hrs
        self.era5_directory = era5_directory
        
        self.atmosphere_directory = atmosphere_directory
        self.atmosphere_source = atmosphere_source
        
        self.climatology_ds = climatology_ds
        
        self.ocean_source = ocean_source
        self.latitude_vals = latitude_vals
        self.longitude_vals = longitude_vals
        
        self.start_from_era5 = start_from_era5
        
        self.flux_ds_upper = None
        self.flux_ds_lower = None
        
        self.poll_counter = 0
        
        # Dummy data array with correct lat/lon coords
        self.base_dataarray = xr.DataArray(
                name='temp',
                data=np.zeros((len(longitude_vals), len(latitude_vals))),
                dims=["longitude", "latitude"],
                coords=dict(
                    longitude=longitude_vals,
                    latitude=latitude_vals
                )
            )

        if self.climatology_ds is not None:
            self.climatology_ds = self.climatology_ds.regrid.linear(self.base_dataarray)

    def current_step_climatology(self, dt: datetime.datetime):
        return self.climatology_ds.interp(dayofyear=dt.dayofyear, hour=dt.hour, method='linear').drop_vars(['dayofyear', 'hour'])

    def __call__(self, 
                 dt: datetime.datetime,
                 ocean_ds: xr.Dataset,
                 test_mode: bool=False) -> xr.Dataset:

            
        if test_mode:
            flux_ds = []
            
            for var in ATM2OCE_VARS:
                flux_ds.append(xr.ones_like(self.base_dataarray).rename(var)*1e-9)
            
            flux_ds = xr.merge(flux_ds)
            
        elif self.atmospheric_timestep_hrs == self.coupling_timestep_hrs:
            
            if dt == self.start_datetime and self.start_from_era5:
                logger.debug(f'Using ERA5 initial conditions for {dt}')
                flux_ds = self.calculate_oasis_fluxes(dt, 
                            ocean_ds,
                            self.era5_directory,
                            atmosphere_source='era5')
            else:
                flux_ds = self.calculate_oasis_fluxes(dt, 
                            ocean_ds,
                            atmosphere_directory,
                            atmosphere_source=self.atmosphere_source)
        else:
                    
            #TODO: make this based on seconds passed, rather than the hour of day, to make it more general
            # Although perhaps it is fine if it's using the nearest hour

            lower_bound_dt = pd.Timestamp(datetime.datetime(dt.year, dt.month, dt.day, dt.hour - (dt.hour % self.atmospheric_timestep_hrs)))
            upper_bound_dt = pd.Timestamp(lower_bound_dt + datetime.timedelta(hours=self.atmospheric_timestep_hrs))

            coeff_upper = (self.atmospheric_timestep_s - np.abs((upper_bound_dt - dt).total_seconds())) / atmospheric_timestep_s
            coeff_lower = (self.atmospheric_timestep_s - np.abs((lower_bound_dt - dt).total_seconds())) / atmospheric_timestep_s
            
            if ((dt - self.start_datetime).seconds % self.atmospheric_timestep_s == 0) or ((dt - self.start_datetime).seconds == self.coupling_timestep_s):
                # If we are at the boundary of an atmospheric timestep, we need to recalculate both upper and lower fluxes
                # We also recalculate both fluxes at the first coupling timestep, since the initial coupling calculation uses ERA5 ocean
                self.flux_ds_lower = self.flux_ds_upper = None

            if self.flux_ds_lower is None or self.flux_ds_upper is None:
                # For first timestep, calculate upper and lower fluxes
                if lower_bound_dt == self.start_datetime:
                    # Use ERA5 initial conditions if it is the very first timestep
                    self.flux_ds_lower = self.calculate_oasis_fluxes(lower_bound_dt, 
                            ocean_ds,
                            self.era5_directory,
                            atmosphere_source='era5')
                else:
                    self.flux_ds_lower = self.calculate_oasis_fluxes(lower_bound_dt, 
                        ocean_ds,
                        self.atmosphere_directory,
                        atmosphere_source=self.atmosphere_source)


                if dt == self.start_datetime:
                    self.flux_ds_upper = xr.zeros_like(self.flux_ds_lower)
                else:
                    self.flux_ds_upper = self.calculate_oasis_fluxes(upper_bound_dt, 
                        ocean_ds,
                        self.atmosphere_directory,
                        self.atmosphere_source)

            logger.debug(f'Interpolating fluxes for {dt} with timestep {atmospheric_timestep_s} s')
            
            # Interpolate fluxes to the atmospheric timestep
            flux_ds = (coeff_upper * self.flux_ds_upper + coeff_lower * self.flux_ds_lower)
            
        return flux_ds

    def calculate_oasis_fluxes(self,
                               dt: datetime.datetime,
                                ocean_ds: xr.Dataset,
                                data_dir: str,
                                atmosphere_source: str) -> xr.Dataset:
        """
        Calculate all fluxes for a given datetime.
        """
        
        flux_ds = self.get_fluxes(dt, 
                            ocean_ds, 
                            data_dir, 
                            atmosphere_source)

        # Convert to OASIS structure
        oasis_flux_ds = fluxes_to_oasis_structure(flux_ds, ocean_ds, self.latitude_vals, self.longitude_vals)
        
        # Mask out land points and interpolate by longitude to avoid large gradients near the coastline
        land_mask = np.isnan(ocean_ds['sea_surface_temperature'])
        ice_mask = ocean_ds['sea_ice_fraction'] > 0.1
        filtered_land_mask = land_mask.copy()
        filtered_land_mask.values = uniform_filter(land_mask.values.astype(np.float32), size=3)
        
        # Remove fluxes from coastal ice areas, since they cause problems
        oasis_flux_ds = xr.where(land_mask, 0.0, oasis_flux_ds)
        oasis_flux_ds = xr.where(ice_mask, xr.where(filtered_land_mask>0, 0.0, oasis_flux_ds), oasis_flux_ds)
        
        # Cap the non-solar fluxes, since they teend to produce extreme values that cause problems with sea ice
        # and sea surface height (also typically near the coast, think there can be problems caused by differences
        # in land-sea mask)
        # oasis_flux_ds['A_Qns_oce'] = xr.where(ice_mask, oasis_flux_ds['A_Qns_oce'].clip(-400,400), oasis_flux_ds['A_Qns_oce'])
        # oasis_flux_ds['A_Qns_ice'] = xr.where(ice_mask, oasis_flux_ds['A_Qns_ice'].clip(-800,800), oasis_flux_ds['A_Qns_oce'])
        
        # Also extend coastal masking to Antarctica and Arctic circle, even if there isn't sea ice there.
        # Since otherwise there are extreme fluxes that cause problems
        south_pole_mask = oasis_flux_ds['latitude'] < -60
        arctic_circle_mask = oasis_flux_ds['latitude'] > 66
        oasis_flux_ds = xr.where(np.logical_and(filtered_land_mask>0, south_pole_mask), 0.0, oasis_flux_ds)
        oasis_flux_ds = xr.where(np.logical_and(filtered_land_mask>0, arctic_circle_mask), 0.0, oasis_flux_ds)
        
        # Important to have no null values
        # Note that sometimes there are null values remaining over Antarctica, hence we fill those with the mean.
        oasis_flux_ds = oasis_flux_ds.fillna(0.0)
        # oasis_flux_ds = oasis_flux_ds.interpolate_na(dim='longitude', method='linear', fill_value="extrapolate").fillna(oasis_flux_ds.mean())

        return oasis_flux_ds
    
    def get_fluxes(self,
                   dt: datetime.date, 
                    ocean_ds: xr.Dataset,
                    data_dir: str,
                    atmosphere_source: str) -> xr.Dataset:
        
        """
        Get fluxes for a given datetime.
        
        Note that the flux sign conventions follow the ECMWF conventions
        """
        
        # Atmosphere dataset, containing variables that may be provided by an atmosphere model or ERA5
        atmosphere_ds = self.create_atmosphere_ds(dt, 
                                                data_dir, 
                                                ocean_ds,
                                                atmosphere_source)

        sea_mask = ~np.isnan(ocean_ds['sea_surface_temperature'])
        
        if atmosphere_source == 'era5':
            # Flux variables taken directly from ERA5
            flux_ds =[]
            for era5_var in ['mean_surface_sensible_heat_flux', 
                            'mean_surface_latent_heat_flux', 
                            'mean_surface_net_long_wave_radiation_flux', 
                            'evaporation', 
                            'instantaneous_eastward_turbulent_surface_stress', 
                            'instantaneous_northward_turbulent_surface_stress', 
                            'mean_surface_net_short_wave_radiation_flux',
                            ]:
                # Gather averages over the coupling timestep
                tmp_da = xr.load_dataarray(os.path.join(data_dir, 'surface', era5_var, f"era5_{era5_var}_{dt.strftime('%Y%m%d')}.nc")).sel(time=dt)
                tmp_da.name = era5_var
                
                if era5_var == 'evaporation':
                    # Convert to kg/m^2/s from m/hour, by multiplying by 1000 (kg/m^3) and dividing by 3600 (s/hour)
                    tmp_da = tmp_da * 1000 / 3600   

                flux_ds.append(tmp_da)
            flux_ds = xr.merge(flux_ds)
            
            if 'latitude' in flux_ds.coords:
                flux_ds = flux_ds.regrid.linear(self.base_dataarray)

            # For ERA5, evaporation over ice already calculated properly
            flux_ds['evaporation_ice'] = flux_ds['evaporation'].copy()
            
            flux_ds['momentum_flux_over_ice_x'] = flux_ds['instantaneous_eastward_turbulent_surface_stress'].copy()
            flux_ds['momentum_flux_over_ice_y'] = flux_ds['instantaneous_northward_turbulent_surface_stress'].copy()
            
            flux_ds['solar_flux_over_ice'] = flux_ds['mean_surface_net_short_wave_radiation_flux'].copy()

            flux_ds['sensible_heat_flux_ice'] = flux_ds['mean_surface_sensible_heat_flux'].copy()
            flux_ds['latent_heat_flux_ice'] = flux_ds['mean_surface_latent_heat_flux'].copy()
            
            flux_ds['net_long_wave_radiation_flux_ice'] = flux_ds['mean_surface_net_long_wave_radiation_flux'].copy()
            
            flux_ds = xr.merge([flux_ds, atmosphere_ds])
        
        elif atmosphere_source in ['gencast', 'era5-calculated']:

            flux_ds = self.calculate_fluxes(atmosphere_ds,
                                                ocean_ds,
                                                max_iterations=50)
            
            flux_ds = flux_ds[['mean_surface_sensible_heat_flux',
                            'mean_surface_latent_heat_flux',
                            'mean_surface_net_long_wave_radiation_flux',
                            'evaporation',
                            'instantaneous_eastward_turbulent_surface_stress',
                            'instantaneous_northward_turbulent_surface_stress'
                            ]]
            
            # TODO: see if we can do better than this
            flux_ds = flux_ds.fillna(flux_ds.mean())
            
            flux_ds['mean_surface_net_short_wave_radiation_flux'] = solar_flux_over_ocean(atmosphere_ds, ocean_ds)
        
            # Calculated fluxes over ice
            non_solar_flux_ds = non_solar_fluxes_ice(atmosphere_ds, ocean_ds, self.current_step_climatology(dt), source=atmosphere_source)
            
            # latent heat flux is negative when ice is sublimated into the air, but ERA5 convention is that "negative values indicate evaporation and positive values indicate condensation". So we keep the ERA5 convention here to be consistent with ERA5 calculations
            flux_ds['evaporation_ice'] = non_solar_flux_ds['latent_heat_flux_ice'] / Ls
            
            flux_ds['solar_flux_over_ice'] = solar_flux_over_ice(atmosphere_ds, ocean_ds)  
            
            flux_ds['momentum_flux_over_ice_x'], flux_ds['momentum_flux_over_ice_y'] = momentum_flux_over_ice(atmosphere_ds)
            
            flux_ds = xr.merge([flux_ds, non_solar_flux_ds, atmosphere_ds])      

        elif atmosphere_source in ['ace2', 'ace2-calculated']:
            # Unfortunately we still need to calculate momentum fluxes, as these aren't provided by ACE2
            calculated_flux_ds = self.calculate_fluxes(atmosphere_ds,
                                                ocean_ds,
                                                max_iterations=50)
            
            if atmosphere_source == 'ace2':
                flux_ds = xr.merge([calculated_flux_ds[['instantaneous_eastward_turbulent_surface_stress', 
                                                        'instantaneous_northward_turbulent_surface_stress', 
                                                        'latent_heat_of_vaporization']], atmosphere_ds])  
                
                # ACE2 sign convention for these fluxes is opposite to ECMWF convention of positive downward 
                flux_ds['mean_surface_latent_heat_flux'] = -1 * flux_ds['mean_surface_latent_heat_flux']
                flux_ds['mean_surface_sensible_heat_flux'] = -1 * flux_ds['mean_surface_sensible_heat_flux']
                
                # Since latent heat of vaporization is not provided by ACE2, we need to use these formulae
                # for evaporation over ice
                flux_ds['evaporation'] = flux_ds['mean_surface_latent_heat_flux'] / (flux_ds['latent_heat_of_vaporization'])
                
                ## Replacing ACE2 fluxes over ice with calculated fluxes over ice 
                non_solar_flux_ds = non_solar_fluxes_ice(atmosphere_ds, ocean_ds, clim_ds=None, source=atmosphere_source)
                # flux_ds = xr.merge([flux_ds, non_solar_flux_ds])
                                
                flux_ds['evaporation_ice'] = non_solar_flux_ds['latent_heat_flux_ice'] / Ls
                # flux_ds['evaporation_ice'] = flux_ds['mean_surface_latent_heat_flux'] / Ls
                
                # Following the ECMWF convention of positive downwards
                flux_ds['mean_surface_net_long_wave_radiation_flux'] = flux_ds['mean_surface_downward_long_wave_radiation_flux'] - flux_ds['mean_surface_upward_long_wave_radiation_flux']
                flux_ds['mean_surface_net_short_wave_radiation_flux'] = flux_ds['mean_surface_downward_short_wave_radiation_flux'] - flux_ds['mean_surface_upward_short_wave_radiation_flux']
                
                # Since ACE2 has ice in the model, we assume these fluxes are correct over ice as well.
                flux_ds['net_long_wave_radiation_flux_ice'] = flux_ds['mean_surface_net_long_wave_radiation_flux'].copy()
                flux_ds['solar_flux_over_ice']  = flux_ds['mean_surface_net_short_wave_radiation_flux'].copy()
                     
                # flux_ds['sensible_heat_flux_ice'] = flux_ds['mean_surface_sensible_heat_flux'].copy()
                # flux_ds['latent_heat_flux_ice'] = flux_ds['mean_surface_latent_heat_flux'].copy()
                flux_ds['sensible_heat_flux_ice'] = non_solar_flux_ds['sensible_heat_flux_ice']
                flux_ds['latent_heat_flux_ice'] = non_solar_flux_ds['latent_heat_flux_ice']
                
                flux_ds['momentum_flux_over_ice_x'], flux_ds['momentum_flux_over_ice_y'] = momentum_flux_over_ice(atmosphere_ds)
            else:
                flux_vars = ['mean_surface_sensible_heat_flux',
                            'mean_surface_latent_heat_flux',
                            'mean_surface_net_long_wave_radiation_flux',
                            'evaporation',
                            'instantaneous_eastward_turbulent_surface_stress',
                            'instantaneous_northward_turbulent_surface_stress'
                            ]
                flux_ds = calculated_flux_ds[flux_vars]
            
                # TODO: see if we can do better than this
                flux_ds = calculated_flux_ds.fillna(calculated_flux_ds.mean())
                
                flux_ds['mean_surface_net_short_wave_radiation_flux'] = solar_flux_over_ocean(atmosphere_ds, ocean_ds)
            
                # Calculated fluxes over ice
                non_solar_flux_ds = non_solar_fluxes_ice(atmosphere_ds, ocean_ds, clim_ds=None, source=atmosphere_source)
                
                # latent heat flux is negative when ice is sublimated into the air, but ERA5 convention is that "negative values indicate evaporation and positive values indicate condensation". So we keep the ERA5 convention here to be consistent with ERA5 calculations
                flux_ds['evaporation_ice'] = non_solar_flux_ds['latent_heat_flux_ice'] / Ls
                
                flux_ds['solar_flux_over_ice'] = solar_flux_over_ice(atmosphere_ds, ocean_ds)  
                
                flux_ds['momentum_flux_over_ice_x'], flux_ds['momentum_flux_over_ice_y'] = momentum_flux_over_ice(atmosphere_ds)

                flux_ds = xr.merge([flux_ds, non_solar_flux_ds, atmosphere_ds[[v for v in atmosphere_ds.data_vars if v not in flux_vars]]])

        flux_ds['total_non_solar_flux_ice'] = flux_ds['net_long_wave_radiation_flux_ice'] + flux_ds['sensible_heat_flux_ice'] + flux_ds['latent_heat_flux_ice']

        # Infer solid precipitation, based on observation that fraction of solid precipitation is typically 1 over ocean points when 2mt <= 273K
        cool_mask = atmosphere_ds['2m_temperature'] <= 273
        
        cool_sea_mask = np.logical_and(cool_mask, sea_mask)
        flux_ds['solid_precipitation'] = xr.where(cool_sea_mask, atmosphere_ds['total_precipitation'], 0)
        flux_ds['liquid_precipitation'] = xr.where(~cool_sea_mask, atmosphere_ds['total_precipitation'], 0)   

        
        return flux_ds
    
    def create_atmosphere_ds(self,
                             dt: datetime.datetime, 
                             data_dir: str,
                             ocean_ds: xr.Dataset,
                             atmosphere_source: str) -> xr.Dataset:
    
        if atmosphere_source == 'era5':
            ds = self.get_atmospheric_fields_era5(dt, data_dir, calculated_fluxes=False)
            
            # Need to do this to accomodate ACE2 grid
            ds = ds.regrid.linear(self.base_dataarray)
        elif atmosphere_source == 'era5-calculated':
            ds = self.get_atmospheric_fields_era5(dt, data_dir, calculated_fluxes=True)
        
            # Need to do this to accomodate ACE2 grid
            ds = ds.regrid.linear(self.base_dataarray)                        
        elif atmosphere_source == 'gencast':
            ds = self.get_atmospheric_fields_gencast(dt, data_dir)
        elif atmosphere_source in ['ace2', 'ace2-calculated']:
            ds = self.get_atmospheric_fields_ace2(dt, data_dir)
        
        ds = ds.sel(latitude=self.latitude_vals, longitude=self.longitude_vals)
        
        if atmosphere_source in ['era5-calculated', 'gencast']:
            ds['specific_humidity_surface'] = interpolate_surface_specific_humidity(ds)
        
        if atmosphere_source in ['era5-calculated', 'gencast', 'ace2']:
            ds['wind_speed'] = np.sqrt(ds['10m_u_component_of_wind']**2 + ds['10m_v_component_of_wind']**2)
            ds['relative_wind_speed_u'] = ds['10m_u_component_of_wind'] - ocean_ds['ocean_current_u']
            ds['relative_wind_speed_v'] = ds['10m_v_component_of_wind'] - ocean_ds['ocean_current_v']
            ds['relative_wind_speed_ice_u'] = ds['10m_u_component_of_wind'] - ocean_ds['ice_velocity_u']
            ds['relative_wind_speed_ice_v'] = ds['10m_v_component_of_wind'] - ocean_ds['ice_velocity_v']

            ds['relative_wind_speed'] = np.sqrt((ds['relative_wind_speed_u'])**2 + (ds['relative_wind_speed_v'])**2)
            ds['relative_wind_speed_ice'] = np.sqrt((ds['relative_wind_speed_ice_u'])**2 + (ds['relative_wind_speed_ice_v'])**2)
        
        return ds
    
    def get_atmospheric_fields_era5(self,
                                    dt: datetime.datetime,
                                    data_dir: str,
                                    calculated_fluxes: bool) -> xr.Dataset:
        if calculated_fluxes:
            era5_vars = ['10m_u_component_of_wind',
                        '10m_v_component_of_wind',
                        'mean_surface_downward_long_wave_radiation_flux',
                        'mean_surface_downward_short_wave_radiation_flux',
                        'mean_surface_net_short_wave_radiation_flux',
                        'mean_sea_level_pressure',
                        '2m_temperature',
                        'total_precipitation']
        else:
            era5_vars = ['2m_temperature', 'total_precipitation']
        
        surface_ds = []
        for era5_var in era5_vars:
            tmp_da = xr.load_dataarray(os.path.join(data_dir, 'surface', era5_var, f"era5_{era5_var}_{dt.strftime('%Y%m%d')}.nc")).sel(time=dt)
            tmp_da.name = era5_var
            surface_ds.append(tmp_da)
        surface_ds = xr.merge(surface_ds)
    
        # Convert precip to kg/m^2/s flux, by multiplying by density of water (1000 kg/m^3) and dividing by 3600 seconds in an hour
        surface_ds['total_precipitation'] = surface_ds['total_precipitation'] * 1000 / (3600)
        
        if calculated_fluxes:
            plevel_ds = []
            for era5_var in ['specific_humidity',
                            'geopotential']:

                fps = [os.path.join(data_dir, 'plevels', era5_var, f'{pl}hPa', f"era5_{era5_var}_{dt.strftime('%Y%m%d')}.nc") for pl in [1000, 975]]
                plevel_ds.append(xr.open_mfdataset(fps, combine='nested', preprocess = lambda x: x.sel(time=dt), concat_dim='pressure_level'))
            plevel_ds = xr.merge(plevel_ds).rename({'z': 'geopotential',
                                                    'q': 'specific_humidity',
                                                    'pressure_level': 'level'})
            
            return xr.merge([surface_ds, plevel_ds])
        else:
            return surface_ds

    def get_atmospheric_fields_gencast(self,
                                       dt: datetime.datetime,
                                       data_dir: str) -> xr.Dataset:
        
        ds = polling2.poll(lambda: xr.load_dataset(os.path.join(data_dir, f"gencast_{dt.strftime('%Y%m%d-%H')}.nc")), 
                        ignore_exceptions=(IOError, ValueError, FileNotFoundError), 
                        timeout=FIRST_POLL_TIMEOUT if self.poll_counter == 0 else POLLING_TIMEOUT,
                        step=0.1,
                        log=logging.ERROR).isel(time=0)
        self.poll_counter += 1

        if 'batch' in ds.coords:
            ds = ds.isel(batch=0)
        
        if 'sample' in ds.coords:
            ds = ds.isel(sample=0)
            
        ds = ds.rename({'lat': 'latitude',
                        'lon': 'longitude'})
        
        # Get radiation data from climatology
        ds = xr.merge([ds, self.current_step_climatology(dt)[['mean_surface_downward_long_wave_radiation_flux', 'mean_surface_downward_short_wave_radiation_flux']]])

        aggregated_precip_fields = [f for f in ds.data_vars if f.startswith('total_precipitation_')]
        
        assert len(aggregated_precip_fields) == 1, "There should be exactly one aggregated precipitation field"
        
        precip_agg_interval = int(aggregated_precip_fields[0].split('_')[-1].replace('hr', ''))
        ds = ds.rename({aggregated_precip_fields[0]: 'total_precipitation'})
        
        # Convert precipitation to kg/m^2/s
        ds['total_precipitation'] = ds['total_precipitation'] * 1000 / (3600 * precip_agg_interval)  # Convert from mm/hour to kg/m^2/s

        return ds


    def get_atmospheric_fields_ace2(self,
                                    dt: datetime.datetime,
                                    data_dir: str) -> xr.Dataset:

        hour_interval = int((dt - self.start_datetime).total_seconds() / 3600)

        logger.debug(f"Polling ACE data in {os.path.join(data_dir, f'ace2_{hour_interval}h.nc')}")
        ds = polling2.poll(lambda: xr.load_dataset(os.path.join(data_dir, f"ace2_{hour_interval}h.nc")), 
                        ignore_exceptions=(IOError, ValueError, FileNotFoundError), 
                        timeout=FIRST_POLL_TIMEOUT if self.poll_counter == 0 else POLLING_TIMEOUT,
                        step=0.1,
                        log=logging.ERROR)
        self.poll_counter += 1
        
        # If we ingest a restart file, it may have time as a variable
        if 'time' in ds.data_vars:
            ds = ds.drop_vars('time')

        if 'time' in ds.dims:
            ds = ds.assign_coords(time=[dt])
        else:
            ds = ds.expand_dims({'time': [dt]})

        ds = ds.isel(time=0)
        
        # precipitation is already in kg/m^2/s
        ds = ds.rename({'PRATEsfc': 'total_precipitation', 
                        'LHTFLsfc': 'mean_surface_latent_heat_flux', 
                        'SHTFLsfc': 'mean_surface_sensible_heat_flux',
                        'DLWRFsfc': 'mean_surface_downward_long_wave_radiation_flux', 
                        'DSWRFsfc': 'mean_surface_downward_short_wave_radiation_flux',
                        'ULWRFsfc': 'mean_surface_upward_long_wave_radiation_flux', 
                        'USWRFsfc': 'mean_surface_upward_short_wave_radiation_flux',
                        'UGRD10m': '10m_u_component_of_wind',
                        'VGRD10m': '10m_v_component_of_wind',
                        'TMP2m': '2m_temperature',
                        'Q2m': 'specific_humidity_surface',
                        'PRESsfc': 'mean_sea_level_pressure'})
        return ds
    
    def calculate_fluxes(self,
                         atmosphere_ds: xr.Dataset,
                        sea_surface_ds: xr.DataArray,
                        max_iterations:int=10):
        """Convert atmosphere and sea surface data into fluxes

        Note we have the sea mask input as dataarray and dataframe, to reduce the number of times needed to convert from dataset to dataframe.
        And since we may be using skin temperature, we can't always infer it from the sea_temperature_da
        Args:
            atmosphere_ds (xr.Dataset): Output of atmosphere model
            sea_surface_ds (xr.DataArray): Sea surface data
            max_iterations (int, optional): Maximum number of iterations for flux calculation. Defaults to 10.

        Returns:
            xr.Dataset: Dataset containing the calculated fluxes
        """
        flux_input_variables = ['2m_temperature',
                            'mean_sea_level_pressure',  # Assumed to be in Pa
                            'relative_wind_speed_u',
                            'relative_wind_speed_v',
                            'mean_surface_downward_short_wave_radiation_flux',
                            'mean_surface_downward_long_wave_radiation_flux',
                            'specific_humidity_surface'] # Required for calculating surface specific humidity
        
        ds = atmosphere_ds[flux_input_variables].copy()
        
        ds['mean_sea_level_pressure'] = ds['mean_sea_level_pressure'] / 100 # Convert to hPA
        ds['specific_humidity_surface'] = ds['specific_humidity_surface'] * 1000 # Convert to g/kg
            
        # Required since the SST ds time gets passed through
        sea_surface_ds['time'] = atmosphere_ds['time']
        sea_surface_ds['sea_mask'] = ~np.isnan(sea_surface_ds['sea_surface_temperature'])
        sea_surface_df = sea_surface_ds.to_dataframe().reset_index()

        ds['wind_speed'] = np.sqrt(ds['relative_wind_speed_u']**2 + ds['relative_wind_speed_v']**2)   

        if 'level' in ds.dims:
            ds = ds.drop_dims('level')
        df = ds.to_dataframe().reset_index()

        flux_df = df[sea_surface_df['sea_mask']].reset_index()
        
        sea_surface_df = sea_surface_df[sea_surface_df['sea_mask']]

        out_vars = ("tau", "sensible", "latent", "cd", "cp", "ct", "cq", "rho", 'dter', 'dqer', 'dtwl', 'rh', 'lv', 'qsea', 'usr', 'Rnl', 'Rs')

        res_ssst = AirSeaFluxCode.AirSeaFluxCode(spd=flux_df['wind_speed'].to_numpy(),
                            T=flux_df['2m_temperature'].to_numpy(),
                            SST=sea_surface_df['sea_surface_temperature'].to_numpy(), # Using SST with cswl adjustment since harder to skin temp from NEMO
                            SST_fl="bulk",
                            meth="ecmwf",
                            lat=flux_df['latitude'].to_numpy(),
                            hin=np.array([10, 2]),
                            hum=('q', flux_df['specific_humidity_surface'].to_numpy()),
                            hout=10,
                            maxiter=max_iterations,
                            P=flux_df['mean_sea_level_pressure'].to_numpy(),
                            cskin=1,
                            Rs=flux_df['mean_surface_downward_short_wave_radiation_flux'].to_numpy(),
                            Rl=flux_df['mean_surface_downward_long_wave_radiation_flux'].to_numpy(),
                            tol=['all', 0.01, 0.01, 1e-05, 1e-3, 0.1, 0.1],
                            L="tsrv",
                            out=0,
                            wl=1,
                            out_var=out_vars)

        res_ssst_df = pd.concat([flux_df[['latitude', 'longitude']], res_ssst], axis=1)
        full_ssst_df = df[['latitude', 'longitude']].merge(res_ssst_df, on=['latitude', 'longitude'], how='left')
        res_ssst_ds = full_ssst_df.set_index(['latitude', 'longitude']).to_xarray()

        # Rename in line with ERA5 variables
        res_ssst_ds = res_ssst_ds.rename({'sensible': 'mean_surface_sensible_heat_flux',
                                        'latent': 'mean_surface_latent_heat_flux',
                                        'Rnl': 'mean_surface_net_long_wave_radiation_flux',
                                        'rho': 'air_density',
                                        'lv': 'latent_heat_of_vaporization',})
        
        # Calculate TauX and TauY
        #TODO: include gustiness contribution
        res_ssst_ds['instantaneous_eastward_turbulent_surface_stress'] = res_ssst_ds['air_density'] * res_ssst_ds['cd'] * ds['wind_speed'] * ds['relative_wind_speed_u']
        res_ssst_ds['instantaneous_northward_turbulent_surface_stress'] = res_ssst_ds['air_density'] * res_ssst_ds['cd'] * ds['wind_speed'] * ds['relative_wind_speed_v']
        
        # Calculate evaporation
        # Following ERA5 convention, evaporation is defined such that negative values indicate evaporation. Since latent heating is negative when evaporation occurs, the signs are the same.
        # Note that ERA5 defines this as metres of water equivalent, whereas here it is in kg/m^2/s
        res_ssst_ds['evaporation'] = res_ssst_ds['mean_surface_latent_heat_flux'] / (res_ssst_ds['latent_heat_of_vaporization']) 
        
        return res_ssst_ds
            

if __name__ == "__main__":

    
    parser = ArgumentParser()
    parser.add_argument('--model-directory', type=str,
                        help='Run directory of ocean model', required=True)
    parser.add_argument('--router-data-directory', type=str, required=True)
    parser.add_argument('--atmosphere-source', type=str, choices=['era5', 'era5-calculated', 'gencast', 'ace2', 'ace2-calculated'], required=True)
    parser.add_argument('--atmosphere-gridfile', type=str,default=None)
    parser.add_argument('--climatology-directory', type=str, default=None)
    parser.add_argument('--era5-directory', type=str,required=True)
    parser.add_argument('--atmospheric-timestep-hrs', type=float, required=True,
                        help='Timestep of atmospheric model in hours')
    parser.add_argument('--coupling-timestep-secs', type=int, required=True,
                        help='Timestep of coupling in seconds')
    parser.add_argument('--ocean-source', type=str, choices=['era5', 'nemo'], required=True)
    parser.add_argument('--deactivated-fluxes', nargs='+', default=None,
                        help='List of fluxes to deactivate. freshwater, momentum, heat(sensible and latent heat fluxes)')
    parser.add_argument('--start-from-era5', action="store_true",
                        help="Whether to use ERA5 ocean data for initial conditions")
    parser.add_argument('--debug', action="store_true",
                        help="activate debugging")
    parser.add_argument('--test-mode', action="store_true",
                        help="activate test mode, where forcing fluxes are replaced by constant shapes")                   
    args = parser.parse_args()
    
    # if args.debug:
    #     # Use ERA5 ocean data for debugging
    #     args.ocean_source = 'era5'
        
    if args.atmosphere_source == 'gencast' and args.climatology_directory is None:
        raise ValueError("Climatology directory must be provided when using gencast source")
    
    print('Setting up logging', flush=True)
    os.makedirs(os.path.join(args.model_directory, 'log'), exist_ok=True)

    # log_level = logging.DEBUG if (args.debug or args.atmosphere_source == 'era5') else logging.INFO
    logging.basicConfig(filename=os.path.join(args.model_directory, 'log', 'router.log'), 
                        encoding='utf-8', level=logging.INFO,
                        format='%(asctime)s %(message)s')

    
    args.atmospheric_timestep_hrs = int(args.atmospheric_timestep_hrs)
    atmospheric_timestep_s = 3600 * args.atmospheric_timestep_hrs
    coupling_timestep_s = args.coupling_timestep_secs # Coupling timstep in seconds, doesn't have to match the ML timestep, but they need to be multiples of each other
    coupling_timestep_hrs = coupling_timestep_s / 3600
    num_atmosphere_steps_per_coupling_step = atmospheric_timestep_s // coupling_timestep_s
    
    assert atmospheric_timestep_s % coupling_timestep_s == 0, "Atmospheric timestep must be a multiple of coupling timestep"

    grid = xr.load_dataset(args.atmosphere_gridfile)
    lat_points = grid['latitude'].values
    lon_points = grid['longitude'].values
    n_lat_points = len(lat_points)
    n_lon_points = len(lon_points)
    n_points = n_lat_points*n_lon_points
    
    logger.info('Reading namelist')
    namelist_dict = f90nml.read(os.path.join(args.model_directory, 'namelist_cfg'))
    
    if namelist_dict == OrderedDict([]):
        # In some cases all of the config is written in the ref namelist
        namelist_dict = f90nml.read(os.path.join(args.model_directory, 'namelist_ref'))

    it000 = namelist_dict['namrun']['nn_it000']
    
    if 'rn_rdt' in namelist_dict['namdom']:
        rndt = namelist_dict['namdom']['rn_rdt']
    else:
        rndt = namelist_dict['namdom']['rn_Dt']

    itend = namelist_dict['namrun']['nn_itend'] # Total number of time steps that NEMO will run for
    date0 = namelist_dict['namrun']['nn_date0']
    sn_rcv_qsr = namelist_dict['namsbc_cpl']['sn_rcv_qsr']
    n_coupling_steps = int(itend * rndt / coupling_timestep_s)
    n_atmosphere_steps = int(itend * rndt / atmospheric_timestep_s)

    logger.info(f'Number of coupling steps: {n_coupling_steps}')
    logger.info(f'Number of atmosphere steps: {n_atmosphere_steps}')

    if args.debug:
        n_coupling_steps = 5  # For debugging, just run for 5 coupling steps

    if args.atmosphere_source in ['era5', 'era5-calculated']:
        
        atmosphere_directory = args.era5_directory
    else:
        atmosphere_directory = args.router_data_directory
        if args.ocean_source != 'era5':
            os.makedirs(args.router_data_directory, exist_ok=True)
    
    start_datetime = datetime.datetime.strptime(str(date0), '%Y%m%d')
    all_datetimes = [pd.Timestamp(start_datetime + datetime.timedelta(seconds=coupling_timestep_s * n)) for n in range(n_coupling_steps)]

    
    logger.info('Starting Graphcast OASIS component')

    # Mocking for debugging
    if (args.ocean_source == 'era5' and args.atmosphere_source=='era5') or args.debug:
        pyoasis = Mock()
        OASIS= Mock()
        OASIS.OUT = 'out'
        OASIS.OUT = 'in'
        mock_component = MagicMock(return_value=None)
        mock_component.enddef = MagicMock(return_value=None)
        pyoasis.Component = MagicMock(return_value=mock_component)

        mock_var = MagicMock(return_value=None)
        mock_var.get = MagicMock(return_value=None)
        mock_var.put = MagicMock(return_value=None)

        pyoasis.Var = MagicMock(return_value=mock_var)
        pyoasis.Component.enddef = MagicMock(return_value=None)
        pyoasis.SerialPartition = MagicMock(return_value='mock_partition')
    else:
        # Since pyoasis can't be installed without Oasis being compiled, for test sytems we mock the import
        import pyoasis
        from pyoasis import OASIS


    # Initialize OASIS
    logger.info('Initializing OASIS')
    comp = pyoasis.Component(args.atmosphere_source)
    logger.info(comp)

    logger.info('Initialising partition')

    partition = pyoasis.BoxPartition(0, n_lon_points, n_lat_points, n_lon_points)
    logger.info(partition)

    send_variables = {}
    for var in ATM2OCE_VARS:
        send_variables[var] = pyoasis.Var(var, partition, OASIS.OUT)
        logger.debug(send_variables[var])

    recv_variables = {}
    for var in OCE2ATM_VARS:
        recv_variables[var] = pyoasis.Var(var, partition, OASIS.IN)
        logger.debug(recv_variables[var])

    logger.debug('End of definition')
    comp.enddef()
    
    if args.atmosphere_source in ['gencast']:

        logger.info("initialising climatology data")

        hour_vals = [td.hour for td in all_datetimes]
        day_of_year_vals = sorted(set([td.dayofyear for td in all_datetimes]))
        
        # Latent and sensible heat fluxes, for the fluxes over ice (from Weatherbench2)
        sensible_heat_flux_fps = [os.path.join(args.climatology_directory, 'mean_surface_sensible_heat_flux', f"era5_clim_mean_surface_sensible_heat_flux_{doy}.nc") for doy in day_of_year_vals]
        msshf_da = xr.open_mfdataset(sensible_heat_flux_fps, combine='nested', concat_dim='dayofyear')['mean_surface_sensible_heat_flux'].compute()
        msshf_da.name = 'mean_surface_sensible_heat_flux_clim'

        latent_heat_flux_fps = [os.path.join(args.climatology_directory, 'mean_surface_latent_heat_flux', f"era5_clim_mean_surface_latent_heat_flux_{doy}.nc") for doy in day_of_year_vals]
        mslhf_da = xr.open_mfdataset(latent_heat_flux_fps, combine='nested', concat_dim='dayofyear')['mean_surface_latent_heat_flux'].compute()
        mslhf_da.name = 'mean_surface_latent_heat_flux_clim'

        # Climatology data for long wave and short wave downward radiation, calculated from ERA5 data
        full_lw_clim_da = xr.load_dataarray(os.path.join(args.climatology_directory, f"mean_mean_surface_downward_long_wave_radiation_flux_1989-01-01__2009-12-31{'_debug' if args.debug else ''}.nc")).sel(dayofyear=day_of_year_vals)
        full_sw_clim_da = xr.load_dataarray(os.path.join(args.climatology_directory, f"mean_mean_surface_downward_short_wave_radiation_flux_1989-01-01__2009-12-31{'_debug' if args.debug else ''}.nc")).sel(dayofyear=day_of_year_vals)

        # lw_clim_da = xr.concat([full_lw_clim_da.sel(dayofyear=day_of_year_vals[n], hour=hour_vals[n]).expand_dims(dim={'time': [all_datetimes[n]]}) for n in range(len(all_datetimes))], dim='time').drop_vars(['hour', 'dayofyear'])
        # sw_clim_da = xr.concat([full_sw_clim_da.sel(dayofyear=day_of_year_vals[n], hour=hour_vals[n]).expand_dims(dim={'time': [all_datetimes[n]]}) for n in range(len(all_datetimes))], dim='time').drop_vars(['hour', 'dayofyear'])
        
        
        clim_ds = xr.merge([full_lw_clim_da, full_sw_clim_da, msshf_da, mslhf_da])
    
    else:
        clim_ds = None
        
    
    flux_calculator = FluxCalculator(
                    atmosphere_source=args.atmosphere_source,
                    era5_directory=args.era5_directory,
                    atmosphere_directory=args.router_data_directory,
                    start_datetime=start_datetime,
                    coupling_timestep_hrs=coupling_timestep_hrs,
                    atmospheric_timestep_hrs=args.atmospheric_timestep_hrs,
                    climatology_ds=clim_ds,
                    ocean_source=args.ocean_source,
                    latitude_vals=lat_points,
                    longitude_vals=lon_points,
                    start_from_era5=args.start_from_era5
                )

    for n in range(n_coupling_steps + 1):

        dt = pd.Timestamp(start_datetime + datetime.timedelta(seconds = coupling_timestep_s * n))

        logger.info(10*'*')
        logger.info(f"Processing timestep {n}, datetime={dt.strftime('%Y-%m-%d %H:%M:%S')}")

        logger.info(10*'*')
        logger.info('Getting received fields')
        start = time.time()

        da_list = []
        for varname, var in recv_variables.items():
            recv_data = pyoasis.asarray(np.zeros((n_lon_points, n_lat_points)))
            var.get(n * coupling_timestep_s, recv_data)
            
            
            da = xr.DataArray(
                name=varname,
                data=recv_data,
                dims=["longitude", "latitude"],
                coords=dict(
                    latitude=lat_points,
                    longitude=lon_points,
                ),
                attrs=dict(
                    description="Variable.",
                    units="",
                ),
            )
            da_list.append(da)

            logger.info(f'Max val for {varname}: {recv_data.max()}')
        
        logger.info(f'getting received fields took {time.time() - start}s')
        
        if n==0:
            restart_ocean_fp = os.path.join(args.router_data_directory, f"restart_oce2atm_0h_{args.atmosphere_source}_{args.ocean_source}{'_debug' if args.debug else ''}.nc")
            
            if os.path.exists(restart_ocean_fp):
                ocean_ds = xr.load_dataset(restart_ocean_fp).sel(latitude=lat_points, longitude=lon_points)
            else:
                ocean_ds = get_era5_ocean_data(dt, 
                                           args.era5_directory, 
                                           atmosphere_grid=grid).sel(latitude=lat_points, longitude=lon_points)
                
        elif args.ocean_source == 'era5' or args.debug:
            # Note that NEMO just gives 0s in the first timestep, so we just use ERA5
            #TODO: create interpolated SST from NEMO restart files
            logger.debug('Getting ERA5 ocean data')
            ocean_ds = get_era5_ocean_data(dt, 
                                           args.era5_directory, 
                                           atmosphere_grid=grid).sel(latitude=lat_points, longitude=lon_points)
        else:
            ocean_ds = xr.merge(da_list).rename({'A_SST': 'sea_surface_temperature',
                    'A_Ice_temp': 'sea_ice_temperature',
                    'A_Ice_albedo': 'ice_albedo',
                    'A_Ice_frac': 'sea_ice_fraction',
                    'A_Ice_thickness': 'sea_ice_thickness',
                    'A_Snow_thickness': 'sea_ice_snow_thickness',
                    'A_OceCurrent_u': 'ocean_current_u',
                    'A_OceCurrent_v': 'ocean_current_v',
                    'A_IceVelocity_u': 'ice_velocity_u',
                    'A_IceVelocity_v': 'ice_velocity_v'}) 
        
        # Remove 0 values, as this confuses AirSeaFluxCode
        ocean_ds['sea_surface_temperature'] = xr.where(ocean_ds['sea_surface_temperature'] == 0, np.nan, ocean_ds['sea_surface_temperature'])
        
        if args.atmosphere_source in ['ace2', 'ace2-calculated']:
            date_str = f'{int((coupling_timestep_s * n) / 3600)}h'  # ACE2 files are named by hours since start
        else:
            date_str = dt.strftime('%Y%m%d-%H')
        output_fp = os.path.join(args.router_data_directory, f"oce2atm_{date_str}_{args.atmosphere_source}_{args.ocean_source}{'_debug' if args.debug else ''}.nc")

        if not args.test_mode:
            if 'time' not in ocean_ds.dims:
                ocean_ds.expand_dims({'time': [dt]}).to_netcdf(output_fp)
            else:
                ocean_ds.to_netcdf(output_fp)
                ocean_ds = ocean_ds.isel(time=0)

        if n == n_coupling_steps:
            # No need to calculate fluxes for the last timestep, since they won't be used
            break
        
        logger.info(10*'*')
        logger.info('Sending fields')
        start = time.time()
        send_data_ds = []
        
        flux_ds = flux_calculator(dt, ocean_ds, test_mode=args.test_mode)

        for varname, var in send_variables.items():

            send_data_da = flux_ds[varname]
            
            if args.deactivated_fluxes is not None:
                if ('momentum' in args.deactivated_fluxes and varname in ['A_TauX_ice', 'A_TauX_oce', 'A_TauY_ice', 'A_TauY_oce']) or \
                    ('heat' in args.deactivated_fluxes and varname in ['A_Qns_ice', 'A_Qns_oce', 'A_Qs_oce', 'A_Qs_ice', 'A_dQns_dT']) or \
                        ('freshwater' in args.deactivated_fluxes and varname in ['A_Evap_ice', 'A_Evap_total', 'A_Precip_liquid', 'A_Precip_solid']) or \
                            ('freshwater_ice' in args.deactivated_fluxes and varname in ['A_Evap_ice', 'A_Precip_liquid', 'A_Precip_solid']) or \
                                ('heat_ice' in args.deactivated_fluxes and varname in ['A_Qns_ice', 'A_Qs_ice']) or 'all' in args.deactivated_fluxes:
                    send_data_da = 1e-6 * send_data_da
                    logger.info(f'Deactivated {varname}')
                    
            # Coordinates must be ordered correctly, otherwise the data will be mangled
            send_data_da = send_data_da.transpose('longitude', 'latitude')

            logger.info(f'Length of {varname}: {send_data_da.values.size}')
            logger.info(f'Max val for {varname}: {send_data_da.max().item()}')
            
            send_data = pyoasis.asarray(send_data_da.values)
            var.put(n * coupling_timestep_s, send_data)
            logger.info(f'Sent {varname}')
            
        logger.info(f'sending fields took {time.time() - start}s')
        if not args.test_mode:
            output_vars = list(flux_ds.data_vars)
            flux_ds[output_vars].assign_coords(time=[dt]).to_netcdf(os.path.join(args.router_data_directory, f"atm2oce_{dt.strftime('%Y%m%d-%H')}_{args.atmosphere_source}_{args.ocean_source}{'_debug' if args.debug else ''}.nc"))
        logger.debug(f'Wrote file')
        
        time.sleep(0.05)
        
    logger.info(f'Finished processing {n_coupling_steps} timesteps')
    logger.info(f'Chacking NEMO timestep has hit {itend}')
    
    # Wait until nemo time step hits the correct point
    def _nemo_timestep(model_directory, itend):
        with open(os.path.join(model_directory, 'time.step'), 'r') as f:
            ts = f.read().strip()
        if ts == str(itend):
            return True
        return False
    
    polling2.poll(lambda: _nemo_timestep(args.model_directory, itend), 
                        ignore_exceptions=(IOError, ValueError, FileNotFoundError), 
                        timeout=5*60,
                        step=0.1,
                        log=logging.ERROR)
    
    time.sleep(30)  # Wait for a bit before finalising