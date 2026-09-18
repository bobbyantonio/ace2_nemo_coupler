# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: ece4
#     language: python
#     name: ece4
# ---

# %%
# #!/usr/bin/env python3
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

        oasis_flux_ds = xr.where(land_mask, 0.0, oasis_flux_ds)
        
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


# %%

# %%
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib import pyplot as plt
from matplotlib import colorbar, colors, gridspec

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature, auto_scaler, AdaptiveScaler

def plot_grid_shared_axes(da_grid, 
                          num_rows, 
                          num_cols, 
                          cbar_label,
                          titles_grid,
                          vmax, 
                          vmin,
                          width_height_ratio = [8,6],
                          shrink_factor=0.7, 
                          central_longitude=180, 
                          wspace=0.001,
                          cbar_height_ratio=0.02,
                          cmap='RdBu_r', 
                          mask=None):
   
    fig = plt.figure(constrained_layout=True, figsize=(shrink_factor*width_height_ratio[0]*2, shrink_factor*width_height_ratio[1]))

    gs = gridspec.GridSpec(num_rows + 1, num_cols, figure=fig, 
                        width_ratios=[1]* num_cols,
                        height_ratios=[1] * num_rows + [0.02],
                           wspace=wspace) 
    plot_axs = [[fig.add_subplot(gs[m, n], projection = ccrs.PlateCarree(central_longitude=central_longitude)) for n in range(num_cols)] for m in range(num_rows)]


    for row in range(num_rows):
        for col in range(num_cols):
            
            plot_da = da_grid[row][col]
            if mask is not None:
                plot_da = xr.where(mask, plot_da, np.nan)
            im = plot_da.plot(ax=plot_axs[row][col], 
                              vmax=vmax, vmin=vmin, 
                              cmap=cmap, 
                              add_colorbar=False, rasterized=True,
                              transform=ccrs.PlateCarree())

            if row == num_rows - 1:
                plot_axs[row][col].set_xticks(np.arange(-180,181,60), crs=ccrs.PlateCarree())
                lon_formatter = cticker.LongitudeFormatter()
                plot_axs[row][col].xaxis.set_major_formatter(lon_formatter)
                plot_axs[row][col].set_xlabel('Longitude')

            if col == 0:
                plot_axs[row][col].set_yticks(np.arange(-90,91,30), crs=ccrs.PlateCarree())
                lat_formatter = cticker.LatitudeFormatter()
                plot_axs[row][col].yaxis.set_major_formatter(lat_formatter)
                plot_axs[row][col].set_ylabel('Latitude')

            plot_axs[row][col].set_title(titles_grid[row][col])

    cbar_ax = fig.add_subplot(gs[row+1, :])
    cbar = plt.colorbar(im, cax=cbar_ax, label=cbar_label, orientation='horizontal')
    cbar.ax.tick_params(labelsize=10)

    return fig, plot_axs

# %%
ace2_grid = xr.open_dataset("/home/ecme4254/hpcperm/ml_model_data/ace2/grid.nc")

# %%
import datetime

base_dir = '/home/ecme4254/scratch/run_dir/n3.6_ace2_1951_control_compressed_19510101-19610101_m0'
start_datetime=datetime.datetime(1951,1,1)
coupling_timestep_hrs=6
coupling_timestep_s=6*3600

flux_calculator = FluxCalculator(
                    atmosphere_source='ace2',
                    era5_directory='/home/ecme4254/scratch/era5',
                    atmosphere_directory=base_dir,
                    start_datetime=start_datetime,
                    coupling_timestep_hrs=coupling_timestep_hrs,
                    atmospheric_timestep_hrs=6,
                    climatology_ds=None,
                    ocean_source='nemo',
                    latitude_vals=ace2_grid['latitude'].values,
                    longitude_vals=ace2_grid['longitude'].values
                )

# %%
n=1
dt = pd.Timestamp(start_datetime + datetime.timedelta(seconds = coupling_timestep_s * n))
date_str = f'{int((coupling_timestep_s * n) / 3600)}h'
atmosphere_source='ace2'

ocean_ds = xr.open_dataset(os.path.join(base_dir, 'router', f"oce2atm_{date_str}_ace2_nemo.nc"))

atmosphere_ds = flux_calculator.create_atmosphere_ds(dt, 
                             os.path.join(base_dir, 'router'),
                             ocean_ds,
                             'ace2')

# %%
era5_ds = []
for var in ['2m_temperature', 'sea_ice_cover', 'mean_surface_latent_heat_flux', 'mean_surface_sensible_heat_flux', 'skin_temperature', 'mean_surface_net_short_wave_radiation_flux']:
    era5_da = xr.load_dataarray(f'/home/ecme4254/hpcperm/era5/surface/{var}/era5_{var}_19510101.nc')
    era5_da.name = var
    era5_ds.append(era5_da)
era5_ds = xr.merge(era5_ds)
era5_ds = era5_ds.regrid.linear(atmosphere_ds)

era5_ds = era5_ds.rename({'sea_ice_cover': 'sea_ice_fraction'})

# %%
# Empirical estimation of bulk transfer coefficient
deltemp = air_density * specific_heat_capacity_air*atmosphere_ds['relative_wind_speed_ice'].isel(time=0)*(era5_ds['2m_temperature'].isel(time=0) - era5_ds['skin_temperature'].isel(time=1))
y = era5_ds['mean_surface_sensible_heat_flux'].isel(time=1)
deltemp_ice = xr.where(ice_mask, deltemp, np.nan).values.flatten()
y_ice = xr.where(ice_mask, y, np.nan).values.flatten()
plt.scatter(deltemp_ice[~np.isnan(deltemp_ice)], y_ice[~np.isnan(deltemp_ice)])

# %%

# %%
# Empirical relationship of ERA5 SHF with CORE SHF
y = era5_ds['mean_surface_sensible_heat_flux'].isel(time=1).values.flatten()
# y_ice = xr.where(ice_mask, y, np.nan).values.flatten()

x = non_solar_flux_ds['sensible_heat_flux_ice'].values.flatten()

plt.scatter(x[~np.isnan(x)], y[~np.isnan(x)])

# %%
delta_x = 2e5
delta_y = 400
cice_empirical = delta_y/delta_x

# %%
cice_empirical

# %%
air_density * specific_heat_capacity_air * C_ice * atmosphere_ds['relative_wind_speed_ice']

# %%
calculated_flux_ds = flux_calculator.calculate_fluxes(atmosphere_ds,
                                                ocean_ds,
                                                max_iterations=50)
            
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

# %%
atmosphere_ds['mean_surface_net_short_wave_radiation_flux'] = atmosphere_ds['mean_surface_downward_short_wave_radiation_flux'] - atmosphere_ds['mean_surface_upward_short_wave_radiation_flux']

# %%
ranges = {'latent_heat_flux_ice':[-400,400], 'sensible_heat_flux_ice': [-300,300],
         'mean_surface_sensible_heat_flux': [-600,600],
         'mean_surface_latent_heat_flux': [-800,800],
          'mean_surface_net_long_wave_radiation_flux': [-150,150],
          'mean_surface_net_short_wave_radiation_flux': [0,1000],
          'instantaneous_eastward_turbulent_surface_stress': [-1, 1],
          'instantaneous_northward_turbulent_surface_stress': [-1,1],
         'evaporation': [-0.0003, 0.0003]
           }

# %%
sea_mask = ~np.isnan(ocean_ds['sea_surface_temperature'].isel(time=0))
ice_mask = ocean_ds['sea_ice_fraction'].isel(time=0) > 0.05

# %%
import matplotlib.pyplot as plt
import cartopy.mpl.ticker as cticker

plot_vars = ['mean_surface_latent_heat_flux', 'mean_surface_sensible_heat_flux'] 
             # 'mean_surface_net_short_wave_radiation_flux']
nrows = len(plot_vars)
ncols=2


                                  
for n, v in enumerate(plot_vars):

    da_dict = {'ACE2': xr.where(ice_mask,atmosphere_ds[v],np.nan),
               # 'ERA5': xr.where(sea_mask,era5_ds[v].isel(time=1),np.nan),
               'Flux': xr.where(ice_mask,core_ice_flux_ds[v].isel(time=0),np.nan),
                  # 'ECE3': -1*ece3_ds[v].rename({'lat':'latitude', 'lon': 'longitude'}).isel(time=0)
              }


    da_dict = {k: v.transpose('latitude', 'longitude') for k,v in da_dict.items()}
        

    fig, axs = plot_grid_shared_axes(da_grid= [list(da_dict.values())], 
                              num_rows=1, 
                              num_cols=len(da_dict.keys()), 
                              titles_grid=[[item.replace('ace2', 'ACE2').replace('era5', 'ERA5') for item in da_dict.keys()]],
                              cbar_label=v,
                              vmin=-200, 
                              vmax=200,
                              shrink_factor=0.7, 
                              central_longitude=180, 
                              cmap='RdBu_r', 
                              mask=None)

# %%
import matplotlib.pyplot as plt
import cartopy.mpl.ticker as cticker

plot_vars = ['mean_surface_downward_short_wave_radiation_flux', 'mean_surface_upward_short_wave_radiation_flux',
            'mean_surface_downward_long_wave_radiation_flux', 'mean_surface_upward_long_wave_radiation_flux']
nrows = len(plot_vars)
ncols=2


                                  
for n, v in enumerate(plot_vars):

    da_dict = {'ACE2': xr.where(sea_mask,atmosphere_ds[v],np.nan),
                  'ECE3': xr.where(sea_mask,ece3_ds[v].isel(time=0),np.nan)
              }


    da_dict = {k: v.transpose('latitude', 'longitude') for k,v in da_dict.items()}
        

    fig, axs = plot_grid_shared_axes(da_grid= [list(da_dict.values())], 
                              num_rows=1, 
                              num_cols=len(da_dict.keys()), 
                              titles_grid=[[item.replace('ace2', 'ACE2').replace('era5', 'ERA5') for item in da_dict.keys()]],
                              cbar_label=v,
                              vmin=None, 
                              vmax=None,
                              shrink_factor=0.7, 
                              central_longitude=180, 
                              cmap='RdBu_r', 
                              mask=None)

# %%
# Compare to restart fluxes

# %%

# %%

ece3_ds = xr.merge([xr.load_dataset(f'/home/ecme4254/scratch/ece3_cmip6_data_download/{s}/{s}_Amon_EC-Earth3P_control-1950_r1i1p2f1_gr_195101-195112.nc') for s in ['hfss', 'hfls','rsus','rsds', 'rlus', 'rlds']])
ece3_ds = ece3_ds.rename({'hfss': 'mean_surface_sensible_heat_flux', 
                          'lat': 'latitude', 'lon': 'longitude',
                          'hfls': 'mean_surface_latent_heat_flux',
                           'rsus': 'mean_surface_downward_short_wave_radiation_flux',
                           'rsds': 'mean_surface_upward_short_wave_radiation_flux',
                         'rlus': 'mean_surface_downward_long_wave_radiation_flux',
                           'rlds': 'mean_surface_upward_long_wave_radiation_flux'})

ece3_ocean_ds = xr.merge([xr.load_dataset(f'/home/ecme4254/scratch/ece3_cmip6_data_download/{s}/{s}_SImon_EC-Earth3P_control-1950_r1i1p2f1_gn_195101-195112.nc') for s in ['siconc']])
ece3_ocean_ds['siconc'] = ece3_ocean_ds['siconc']/100.0
ece3_ocean_ds = ece3_ocean_ds.rename({'siconc': 'sea_ice_fraction', 
                          })


# %%
import xesmf as xe
regridder_ece3_ocean = xe.Regridder(ece3_ocean_ds['sea_ice_fraction'].isel(time=0), 
                         atmosphere_ds.isel(time=0)['2m_temperature'], 
                         'bilinear',
                         ignore_degenerate=True, 
                         reuse_weights=False, 
                         periodic=True, 
                         filename='weights_ece3.nc')

regridder_ece3_atm = xe.Regridder(ece3_ds['mean_surface_sensible_heat_flux'].isel(time=0), 
                         atmosphere_ds.isel(time=0)['2m_temperature'], 
                         'bilinear',
                         ignore_degenerate=True, 
                         reuse_weights=False, 
                         periodic=True, 
                         filename='weights_ece3.nc')

# %%
# Ocean variables require regridding
ece3_ocean_ds = regridder_ece3_ocean(ece3_ocean_ds)
ece3_ds = regridder_ece3_atm(ece3_ds)

ece3_ds = xr.merge([ece3_ds, ece3_ocean_ds])
# ece3_ds = convert_dts_to_first_of_month(ece3_ds)

# ece3_ds = ece3_ds.rename(ece3_var_lookup)

# %%
# Load restarts
restart_ice = xr.load_dataset('/perm/ecme4254/ece3data/nemo/restart/ORCA1/19510101/restart_ice.nc')

# %%
non_solar_flux_ds = non_solar_fluxes_ice(atmosphere_ds, ocean_ds, clim_ds=None, source='ace2')


# %%
quantile_vals = np.linspace(0,1,1000)

tmp_ice_mask = ocean_ds['sea_ice_fraction'].isel(time=0) > 0.1
tmp_era5_ice_mask = era5_ds.isel(time=1)['sea_ice_fraction'] > 0.1

both_ice_mask = np.logical_and(tmp_era5_ice_mask, tmp_ice_mask)

for v in ['sensible', 'latent']:
    
    core_vals = xr.where(both_ice_mask, non_solar_flux_ds[f'{v}_heat_flux_ice'].isel(time=0), np.nan).values.flatten()
    era5_vals = xr.where(both_ice_mask, era5_ds.isel(time=1)[f'mean_surface_{v}_heat_flux'], np.nan).values.flatten()
    
    non_null_idxs = ~np.isnan(core_vals)
    core_vals = core_vals[non_null_idxs]

    era5_vals = era5_vals[non_null_idxs]

    output_fp = f'/home/ecme4254/hpcperm/ml_model_data/quantile_mapping/{v}_heat_flux_quantiles_era5.pkl'
    print(output_fp)
    with open(output_fp, 'wb+') as ofh:
        pickle.dump({'core': [np.quantile(core_vals, q) for q in quantile_vals], 
                     'era5':[np.quantile(era5_vals, q) for q in quantile_vals]}, ofh)
    


# %%
shf_da = non_solar_flux_ds['sensible_heat_flux_ice'].copy()
sensible_heat_flux_quantiles = pickle.load(open(f'/home/ecme4254/hpcperm/ml_model_data/quantile_mapping/sensible_heat_flux_quantiles_era5.pkl', 'rb'))
shf_da.values = np.interp(non_solar_flux_ds['sensible_heat_flux_ice'].values, 
                                                              sensible_heat_flux_quantiles['core'], 
                                                              sensible_heat_flux_quantiles['era5'])

# %%
np.max(sensible_heat_flux_quantiles['core'])

# %%
numerator = (sensible_heat_flux_quantiles['era5'][-1] - sensible_heat_flux_quantiles['era5'][-4])
denominator = (sensible_heat_flux_quantiles['core'][-1] - sensible_heat_flux_quantiles['core'][-4])
positive_uplift_gradient = numerator / denominator

numerator = (sensible_heat_flux_quantiles['era5'][4] - sensible_heat_flux_quantiles['era5'][0])
denominator = (sensible_heat_flux_quantiles['core'][4] - sensible_heat_flux_quantiles['core'][0])
negative_uplift_gradient = numerator / denominator

shf_da = xr.where(non_solar_flux_ds['sensible_heat_flux_ice'] > np.max(sensible_heat_flux_quantiles['core']), positive_uplift_gradient*non_solar_flux_ds['sensible_heat_flux_ice'], shf_da)
shf_da = xr.where(non_solar_flux_ds['sensible_heat_flux_ice'] < np.min(sensible_heat_flux_quantiles['core']), negative_uplift_gradient*non_solar_flux_ds['sensible_heat_flux_ice'], shf_da)


# %%
negative_uplift_gradient

# %%
positive_uplift_gradient

# %%
np.max(non_solar_flux_ds['sensible_heat_flux_ice'].values)

# %%
shf_da = xr.where(non_solar_flux_ds['sensible_heat_flux_ice'] > np.max(sensible_heat_flux_quantiles['core']), positive_uplift_gradient*non_solar_flux_ds['sensible_heat_flux_ice'], shf_da)
shf_da = xr.where(non_solar_flux_ds['sensible_heat_flux_ice'] < np.min(sensible_heat_flux_quantiles['core']), negative_uplift_gradient*non_solar_flux_ds['sensible_heat_flux_ice'], shf_da)



# %%
xr.where(non_solar_flux_ds['sensible_heat_flux_ice'] > np.max(sensible_heat_flux_quantiles['core']), positive_uplift_gradient* non_solar_flux_ds['sensible_heat_flux_ice'], np.nan )




# %%

flux_type='sensible'

loaded_quantiles = pickle.load(open(f'/home/ecme4254/hpcperm/ml_model_data/quantile_mapping/{flux_type}_heat_flux_quantiles_era5.pkl', 'rb'))
fig, ax = plt.subplots(1,1)
ax.scatter(loaded_quantiles['core'], loaded_quantiles['era5'])


# %%
# core_ice_flux_ds = xr.merge([solar_flux_over_ice(atmosphere_ds, ocean_ds), non_solar_flux_ds['sensible_heat_flux_ice'], non_solar_flux_ds['latent_heat_flux_ice']])
quantile_vals = np.linspace(0,1,1000)

# ace2_vals = non_solar_flux_ds['sensible_heat_flux_ice'].values.flatten()
core_val_list = {'sensible': [], 'latent': []}
ece3_val_list  = {'sensible': [], 'latent': []}
for month in range(1,13):

    atm2oce_ds = xr.load_dataset(f"/home/ecme4254/hpcperm/model_runs/n3.6_ace2_1951_control_compressed_19510101-20210101_m2/atm2oce_MS_ace2_nemo_1951{month:02d}.nc").isel(time=0)
    oce2atm_ds = xr.load_dataset(f"/home/ecme4254/hpcperm/model_runs/n3.6_ace2_1951_control_compressed_19510101-20210101_m2/oce2atm_MS_ace2_nemo_1951{month:02d}.nc").isel(time=0)
    tmp_ice_mask = oce2atm_ds['sea_ice_fraction'] > 0.1
    tmp_ece3_ice_mask = ece3_ds.isel(time=month-1)['sea_ice_fraction'] > 0.1
    
    both_ice_mask = np.logical_and(tmp_ece3_ice_mask, tmp_ice_mask)

    for v in ['sensible', 'latent']:
        
        core_vals = xr.where(both_ice_mask, atm2oce_ds[f'{v}_heat_flux_ice'], np.nan).values.flatten()
        ece3_vals = xr.where(both_ice_mask, ece3_ds.isel(time=month-1)[f'mean_surface_{v}_heat_flux'], np.nan).values.flatten()
        
        non_null_idxs = ~np.isnan(core_vals)
        core_vals = core_vals[non_null_idxs]

        # Note we have to reverse signs of the ECE fluxes since they are defined in the opposite direction
        ece3_vals = (-1)*ece3_vals[non_null_idxs]

        core_val_list[v] += list(core_vals)
        ece3_val_list[v] += list(ece3_vals)

        output_fp = f'/home/ecme4254/hpcperm/ml_model_data/quantile_mapping/{v}_heat_flux_quantiles_month{month:02d}.pkl'
        print(output_fp)
        with open(output_fp, 'wb+') as ofh:
            pickle.dump({'core': [np.quantile(core_vals, q) for q in quantile_vals], 
                         'ece3':[np.quantile(ece3_vals, q) for q in quantile_vals]}, ofh)
        
core_quantiles = {k: [np.quantile(core_val_list[k], q) for q in quantile_vals] for k in ['sensible', 'latent']}
ece3_quantiles = {k: [np.quantile(ece3_val_list[k], q) for q in quantile_vals] for k in ['sensible', 'latent']}

for k in ['sensible', 'latent']:
    output_fp = f'/home/ecme4254/hpcperm/ml_model_data/quantile_mapping/{k}_heat_flux_quantiles.pkl'
    print(output_fp)
    with open(output_fp, 'wb+') as ofh:
        pickle.dump({'core': core_quantiles[k], 'ece3': ece3_quantiles[k]}, ofh)


# %%

flux_type='sensible'

loaded_quantiles = pickle.load(open(f'/home/ecme4254/hpcperm/ml_model_data/quantile_mapping/{flux_type}_heat_flux_quantiles_month01.pkl', 'rb'))
fig, ax = plt.subplots(1,1)
ax.scatter(loaded_quantiles['core'], loaded_quantiles['ece3'])

loaded_quantiles = pickle.load(open(f'/home/ecme4254/hpcperm/ml_model_data/quantile_mapping/{flux_type}_heat_flux_quantiles.pkl', 'rb'))
fig, ax = plt.subplots(1,1)
ax.scatter(loaded_quantiles['core'], loaded_quantiles['ece3'])

# %%
fig, ax = plt.subplots(1,1)

ax.scatter(core_quantiles['latent'], ece3_quantiles['latent'])

# %%
# ace2_vals = non_solar_flux_ds['sensible_heat_flux_ice'].values.flatten()
core_vals = xr.where(ice_mask, core_ice_flux_ds['short_wave_radiation_flux_ice'], np.nan).values.flatten()
ece3_vals = xr.where(ice_mask, ece3_ds['mean_surface_latent_heat_flux'], np.nan).isel(time=0).values.flatten()

non_null_idxs = ~np.isnan(core_vals)
core_vals = core_vals[non_null_idxs]
era5_vals = era5_vals[non_null_idxs]
ece3_vals = (-1)*ece3_vals[non_null_idxs]

core_quantiles = [np.quantile(core_vals, q) for q in quantile_vals]
ece3_quantiles = [np.quantile(ece3_vals, q) for q in quantile_vals]
    
# non_solar_flux_ds['sensible_heat_flux_ice_corrected'] = non_solar_flux_ds['sensible_heat_flux_ice'].copy()
core_ice_flux_ds['latent_heat_flux_ice'].values = np.interp(non_solar_flux_ds['latent_heat_flux_ice'].values, core_quantiles, ece3_quantiles, right=np.nan)

core_ice_flux_ds = core_ice_flux_ds.rename({'latent_heat_flux_ice': 'mean_surface_latent_heat_flux'})

fig, ax = plt.subplots(1,1)

ax.scatter(core_quantiles, ece3_quantiles)

# %%
# ace2_series = pd.Series(ace2_vals)
core_series = pd.Series(core_vals)
era5_series = pd.Series(era5_vals)
ece3_series = pd.Series(ece3_vals)

# %%





# %%
quantile_vals = np.linspace(0,1,1000)

core_quantiles = [np.quantile(core_vals, q) for q in quantile_vals]
ece3_quantiles = [np.quantile(ece3_vals, q) for q in quantile_vals]

fig, ax = plt.subplots(1,1)

ax.scatter(core_quantiles, ece3_quantiles)

# %%
core_ice_flux_ds = xr.merge([solar_flux_over_ice(atmosphere_ds, ocean_ds), non_solar_flux_ds['sensible_heat_flux_ice'], non_solar_flux_ds['latent_heat_flux_ice']])

# non_solar_flux_ds['sensible_heat_flux_ice_corrected'] = non_solar_flux_ds['sensible_heat_flux_ice'].copy()
core_ice_flux_ds['sensible_heat_flux_ice'].values = np.interp(non_solar_flux_ds['sensible_heat_flux_ice'].values, core_quantiles, ece3_quantiles, right=np.nan)

core_ice_flux_ds = core_ice_flux_ds.rename({'sensible_heat_flux_ice': 'mean_surface_sensible_heat_flux'})


# %%
import cartopy.mpl.ticker as cticker

plot_vars = [ 'mean_surface_sensible_heat_flux', 'mean_surface_latent_heat_flux']
nrows = len(plot_vars)
ncols=2


                                  
for n, v in enumerate(plot_vars):

    da_dict = {'ACE2': xr.where(sea_mask,atmosphere_ds[v],np.nan),
               'ERA5': xr.where(sea_mask,era5_ds[v].isel(time=1),np.nan),
               'Flux': xr.where(sea_mask,core_ice_flux_ds[v].isel(time=0),np.nan),
                  'ECE3': (-1)*ece3_ds[v].isel(time=0)
              }


    da_dict = {k: v.transpose('latitude', 'longitude') for k,v in da_dict.items()}
        

    fig, axs = plot_grid_shared_axes(da_grid= [list(da_dict.values())], 
                              num_rows=1, 
                              num_cols=len(da_dict.keys()), 
                              titles_grid=[[item.replace('ace2', 'ACE2').replace('era5', 'ERA5') for item in da_dict.keys()]],
                              cbar_label=v,
                              vmin=-200, 
                              vmax=200,
                              shrink_factor=0.7, 
                              central_longitude=180, 
                              cmap='RdBu_r', 
                              mask=None)

# %%
fig, ax = plt.subplots(1,1)

# ace2_im = ((ace2_series)).plot.kde(linewidth=2, ax=ax, label='ACE2')
core_im = ((core_series)/2).plot.kde(linewidth=2, ax=ax, label='CORE')
((era5_series)).plot.kde(linewidth=2, ax=ax, label='ERA5')
# ((ece3_series)).plot.kde(linewidth=2, ax=ax, label='ECE3')

# ax.set_yscale('log')
ax.legend()

# %%
fig, ax = plt.subplots(1,1)

ace2_im = (np.abs(ace2_series)).plot.kde(linewidth=2, ax=ax, label='ACE2')
# core_im = ((core_series)/2).plot.kde(linewidth=2, ax=ax, label='CORE')
(np.abs(era5_series)).plot.kde(linewidth=2, ax=ax, label='ERA5')
(np.abs(ece3_series)).plot.kde(linewidth=2, ax=ax, label='ECE3')

# ax.set_yscale('log')
ax.set_xscale('log')

ax.legend()


# %%
def sensible_heat_flux_over_ice(atmosphere_ds, ice_ds):
    # Sensible heat flux; note that it is defined as positive when heat is transferred from the air to the ice
    sensible_heat_flux = air_density * specific_heat_capacity_air * C_ice * atmosphere_ds['relative_wind_speed_ice'] * (atmosphere_ds['2m_temperature'] - ice_ds['sea_ice_temperature'])
    
    return sensible_heat_flux


# %%
sensible_heat_flux = air_density * specific_heat_capacity_air * C_ice * atmosphere_ds['relative_wind_speed_ice'] * (atmosphere_ds['2m_temperature'] - ocean_ds['sea_ice_temperature'])
sensible_heat_flux_surf = air_density * specific_heat_capacity_air * C_ice * atmosphere_ds['relative_wind_speed_ice'] * (atmosphere_ds['surface_temperature'] - ocean_ds['sea_ice_temperature'])


# %%
fig, ax = plt.subplots(1,3, figsize=(3*8,6))

xr.where(ice_mask, sensible_heat_flux.isel(time=0), np.nan).plot(vmax=300, vmin=-300, cmap='RdBu_r', x='longitude', y='latitude', ax=ax[0])
xr.where(ice_mask, sensible_heat_flux_surf.isel(time=0), np.nan).plot(vmax=300, vmin=-300, cmap='RdBu_r', x='longitude', y='latitude', ax=ax[1])
xr.where(ice_mask, (sensible_heat_flux_surf - sensible_heat_flux_surf).isel(time=0), np.nan).plot(cmap='RdBu_r', x='longitude', y='latitude', ax=ax[2])

# %%
ocean_ds['sea_ice_temperature'].isel(time=0).plot(vmin=220, vmax=273, x='longitude', y='latitude')

# %%
era5_ds['2m_temperature'].isel(time=0).plot()

# %%
(atmosphere_ds['2m_temperature'] - era5_ds['skin_temperature'].isel(time=0)).plot()

# %%
ncols = 3
fig, ax = plt.subplots(1,ncols, figsize=(8*ncols,6))

xr.where(ice_mask, atmosphere_ds['2m_temperature']- ocean_ds['sea_ice_temperature'].isel(time=0), np.nan).plot( cmap='RdBu_r', x='longitude', y='latitude', ax=ax[0])
xr.where(ice_mask, sensible_heat_flux, np.nan).plot(vmax=100, vmin=-100, cmap='RdBu_r', x='longitude', y='latitude', ax=ax[1])
xr.where(ice_mask, flux_ds['sensible_heat_flux_ice'], np.nan).plot(vmax=100, vmin=-100,cmap='RdBu_r', x='longitude', y='latitude', ax=ax[2])

ax[0].set_title('T2m - Ice temp')
ax[1].set_title('CORE sensible heat flux')
ax[2].set_title('ACE2 heat flux')

# %%
ncols = 3
fig, ax = plt.subplots(1,ncols, figsize=(8*ncols,6))

xr.where(ice_mask, atmosphere_ds['2m_temperature']- ocean_ds['sea_ice_temperature'].isel(time=0), np.nan).plot( cmap='RdBu_r', x='longitude', y='latitude', ax=ax[0])
xr.where(ice_mask, sensible_heat_flux, np.nan).plot(vmax=100, vmin=-100, cmap='RdBu_r', x='longitude', y='latitude', ax=ax[1])
xr.where(ice_mask, flux_ds['sensible_heat_flux_ice'], np.nan).plot(vmax=100, vmin=-100,cmap='RdBu_r', x='longitude', y='latitude', ax=ax[2])

ax[0].set_title('T2m - Ice temp')
ax[1].set_title('CORE sensible heat flux')
ax[2].set_title('ACE2 heat flux')

# %%
xr.where(ice_mask, (atmosphere_ds['surface_temperature'] - atmosphere_ds['2m_temperature'])/atmosphere_ds['surface_temperature'], np.nan).plot(x='longitude', y='latitude',)

# %%
## Look at ice restart files

# %%
ice_ds = xr.load_dataset('/home/ecme4254/perm/ece3data/nemo/restart/ORCA1/19510101/restart_ice.nc')

# %%
ice_ds['temp_ice_avg'] = 0.2* ( ice_ds['tempt_il1_htc1'] + ice_ds['tempt_il1_htc2'] + ice_ds['tempt_il1_htc3'] + ice_ds['tempt_il1_htc4'] + ice_ds['tempt_il1_htc5'])

# %%
ice_ds

# %%
avg_vals  = []
for i in range(5):
    avg_vals.append((i, ice_ds[f't_su_htc{i+1}'].mean()))

# %%
fig, ax = plt.subplots(1,1)

ax.scatter([item[0] for item in avg_vals], [item[1] for item in avg_vals])
ax.set_xlabel('Ice depth label (larger is deeper)')
ax.set_ylabel('Ice temperature [K]')

# %%
ice_ds[f't_su_htc1'].isel(t=0) .plot()

# %%
oce2atm_new = xr.load_dataset('/home/ecme4254/scratch/run_dir/n3.6_ace2_1951_multiIceCat_19510101-19560101_m0/router/oce2atm_6h_ace2_nemo.nc')

# %%
oce2atm_old = xr.load_dataset('/home/ecme4254/scratch/run_dir/n3.6_ace2_1951_control_compressed_19510101-19610101_m0/router/oce2atm_6h_ace2_nemo.nc')

# %%
(oce2atm_new['sea_ice_temperature'] - oce2atm_old['sea_ice_temperature']).isel(time=0).plot()

# %%
