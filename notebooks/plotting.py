# %%
import os
import copy
import string
import pickle
from tqdm import tqdm

# %%


import matplotlib as mpl
import numpy as np
import xarray as xr
from typing import Iterable
from scipy.stats import pearsonr
from matplotlib import pyplot as plt
from matplotlib import colorbar, colors, gridspec

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature, auto_scaler, AdaptiveScaler
import cartopy.mpl.ticker as cticker
from shapely.geometry import Polygon, LineString

# %%
path = os.path.dirname(os.path.abspath(__file__))

# %%
palette="YlGnBu"
default_linewidth = 0.4
alpha = 0.8
spacing = 10

# %%

# %%
step_size = 0.001
range_dict = {0: {'start': 0.1, 'stop': 1, 'interval': 0.1, 'marker': '+', 'marker_size': 32},
              1: {'start': 1, 'stop': 10, 'interval': 1, 'marker': '+', 'marker_size': 256},
              2: {'start': 10, 'stop': 80, 'interval':1, 'marker': '+', 'marker_size': 256},
              3: {'start': 80, 'stop': 99.1, 'interval': 1, 'marker': '+', 'marker_size': 256},
              4: {'start': 99.1, 'stop': 99.91, 'interval': 0.1, 'marker': '+', 'marker_size': 128},
              5: {'start': 99.9, 'stop': 99.99, 'interval': 0.01, 'marker': '+', 'marker_size': 32 },
              6: {'start': 99.99, 'stop': 99.999, 'interval': 0.001, 'marker': '+', 'marker_size': 10},
              7: {'start': 99.999, 'stop': 99.9999, 'interval': 0.0001, 'marker': '+', 'marker_size': 10},
              8: {'start': 99.9999, 'stop': 99.99999, 'interval': 0.00001, 'marker': '+', 'marker_size': 10}}

# %%
percentiles_list= [np.arange(item['start'], item['stop'], item['interval']) for item in range_dict.values()]
percentiles=np.concatenate(percentiles_list)
quantile_locs = [item / 100.0 for item in percentiles]


# %%
def get_geoaxes(*args, **kwargs):
    fig, ax = plt.subplots(*args, 
                        subplot_kw={'projection' : ccrs.PlateCarree()},
                        **kwargs)
    
    return fig, ax


def plot_maps_shared_colorbar(da_grid, 
                          cbar_label,
                          titles_grid,
                          vmax, 
                          vmin,
                          projection=ccrs.PlateCarree(central_longitude=180),
                          width_height_ratio = [8,6],
                          shrink_factor=0.7, 
                          wspace=0.001,
                          cbar_height_ratio=0.02,
                          cmap='RdBu_r', 
                          mask=None,
                          lon_ticks=np.arange(-180,181,60),
                          lat_ticks=np.arange(-90,91,30)):

    num_rows = len(da_grid)
    num_cols = len(da_grid[0])

    if vmax is None or vmin is None:
        raise ValueError("vmax and vmin must not be None, otherwise plots will be inconsistent")
    fig = plt.figure(constrained_layout=True, figsize=(shrink_factor*width_height_ratio[0]*num_cols, shrink_factor*width_height_ratio[1]*num_rows))

    gs = gridspec.GridSpec(num_rows + 1, num_cols, figure=fig, 
                        width_ratios=[1]* num_cols,
                        height_ratios=[1] * num_rows + [0.02],
                           wspace=wspace) 
    plot_axs = [[fig.add_subplot(gs[m, n], projection = projection) for n in range(num_cols)] for m in range(num_rows)]


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

            try:
                if row == num_rows - 1:
                    plot_axs[row][col].set_xticks(lon_ticks, crs=ccrs.PlateCarree())
                    lon_formatter = cticker.LongitudeFormatter()
                    plot_axs[row][col].xaxis.set_major_formatter(lon_formatter)
                    plot_axs[row][col].set_xlabel('Longitude')
    
                if col == 0:
                    plot_axs[row][col].set_yticks(lat_ticks, crs=ccrs.PlateCarree())
                    lat_formatter = cticker.LatitudeFormatter()
                    plot_axs[row][col].yaxis.set_major_formatter(lat_formatter)
                    plot_axs[row][col].set_ylabel('Latitude')
            except RuntimeError:
                pass

            plot_axs[row][col].set_title(titles_grid[row][col])

    cbar_ax = fig.add_subplot(gs[row+1, :])
    cbar = plt.colorbar(im, cax=cbar_ax, label=cbar_label, orientation='horizontal')
    cbar.ax.tick_params(labelsize=10)

    return fig, plot_axs

def plot_imshow_shared_axes(da_grid, 
                          num_rows, 
                          num_cols, 
                          cbar_label,
                          titles_grid,
                          width_height_ratio = [8,6],
                          shrink_factor=0.7, 
                          wspace=0.001,
                          cbar_height_ratio=0.02,
                          cmap='RdBu_r', 
                          mask=None,
                           **plot_kwargs):
   
    fig = plt.figure(constrained_layout=True, figsize=(shrink_factor*width_height_ratio[0]*2, shrink_factor*width_height_ratio[1]))

    gs = gridspec.GridSpec(num_rows + 1, num_cols, figure=fig, 
                        width_ratios=[1]* num_cols,
                        height_ratios=[1] * num_rows + [0.02],
                           wspace=wspace) 
    plot_axs = [[fig.add_subplot(gs[m, n]) for n in range(num_cols)] for m in range(num_rows)]


    for row in range(num_rows):
        for col in range(num_cols):
            
            plot_da = da_grid[row][col]
            if mask is not None:
                plot_da = xr.where(mask, plot_da, np.nan)
            im = plot_da.plot(ax=plot_axs[row][col], 
                              cmap=cmap, 
                              add_colorbar=False, rasterized=True,
                              **plot_kwargs)

            plot_axs[row][col].set_title(titles_grid[row][col])

    cbar_ax = fig.add_subplot(gs[row+1, :])
    cbar = plt.colorbar(im, cax=cbar_ax, label=cbar_label, orientation='horizontal')
    cbar.ax.tick_params(labelsize=10)

    return fig, plot_axs

def plot_map_grid_cbar_by_row(da_grid,
                                cbar_labels,
                                titles_grid ,
                                vmax_vals,
                                vmin_vals,
                                  projection,
                                  cmaps,
                                width_height_ratio = [8,6],
                                shrink_factor= 1,
                                wspace=0.001,
                                cbar_height_ratio=0.02,
                              lat_ticks=None,
                              lon_ticks=None
                                ):

    num_rows = len(da_grid)
    num_cols = len(da_grid[0])
    
    fig = plt.figure(constrained_layout=True, figsize=(num_cols*shrink_factor*width_height_ratio[0], num_rows*shrink_factor*width_height_ratio[1]))
    
    gs = gridspec.GridSpec(num_rows * 2, 2*num_cols, figure=fig, 
                        width_ratios=[1]* 2*num_cols,
                        height_ratios=[1, 0.02] * num_rows,
                           wspace=wspace) 
    plot_axs = [[fig.add_subplot(gs[2*m, 2*n:2*n+2], projection = projection) for n in range(num_cols)] for m in range(num_rows)]
    
    
    for row in range(num_rows):
        for col in range(num_cols):
            
            plot_da = da_grid[row][col]

            im = plot_da.plot(ax=plot_axs[row][col], 
                              vmax=vmax_vals[row], 
                              vmin=vmin_vals[row], 
                              cmap=cmaps[row], 
                              add_colorbar=False, rasterized=True,
                              transform=ccrs.PlateCarree())
    
            try:
                if lon_ticks is not None:
                    plot_axs[row][col].set_xticks(lon_ticks, crs=ccrs.PlateCarree())
                    lon_formatter = cticker.LongitudeFormatter()
                    plot_axs[row][col].xaxis.set_major_formatter(lon_formatter)
                    plot_axs[row][col].set_xlabel('Longitude')
    
                if col == 0 and lat_ticks is not None:
                    plot_axs[row][col].set_yticks(lat_ticks, crs=ccrs.PlateCarree())
                    lat_formatter = cticker.LatitudeFormatter()
                    plot_axs[row][col].yaxis.set_major_formatter(lat_formatter)
                    plot_axs[row][col].set_ylabel('Latitude')
            except RuntimeError:
                pass
    
            plot_axs[row][col].set_title(titles_grid[row][col])
            plot_axs[row][col].coastlines()
    
        cbar_ax = fig.add_subplot(gs[2*row+1, 1:-1])
        cbar = plt.colorbar(im, cax=cbar_ax, label=cbar_labels[row], orientation='horizontal')
        cbar.ax.tick_params(labelsize=10)
    return fig, plot_axs

def plot_map_grid_cbar_by_column(da_grid,
                                cbar_labels,
                                titles_grid ,
                                vmax_vals,
                                vmin_vals,
                                  projection,
                                  cmaps,
                                width_height_ratio = [8,6],
                                shrink_factor= 1,
                                wspace=0.001,
                                cbar_height_ratio=0.02,
                              lat_ticks=None,
                              lon_ticks=None
                                ):

    num_rows = len(da_grid)
    num_cols = len(da_grid[0])
    
    fig = plt.figure(constrained_layout=True, figsize=(num_cols*shrink_factor*width_height_ratio[0], num_rows*shrink_factor*width_height_ratio[1]))
    
    gs = gridspec.GridSpec(num_rows + 1, num_cols, figure=fig, 
                          width_ratios=[1]* num_cols,
                        height_ratios=[1] * (num_rows) + [cbar_height_ratio],
                           wspace=wspace) 
    plot_axs = [[fig.add_subplot(gs[m, n], projection = projection) for n in range(num_cols)] for m in range(num_rows)]
    
    for col in range(num_cols):
        for row in range(num_rows):
        
            
            plot_da = da_grid[row][col]

            im = plot_da.plot(ax=plot_axs[row][col], 
                              vmax=vmax_vals[col], 
                              vmin=vmin_vals[col], 
                              cmap=cmaps[col], 
                              add_colorbar=False, rasterized=True,
                              transform=ccrs.PlateCarree())
    
            try:
                if lon_ticks is not None:
                    plot_axs[row][col].set_xticks(lon_ticks, crs=ccrs.PlateCarree())
                    lon_formatter = cticker.LongitudeFormatter()
                    plot_axs[row][col].xaxis.set_major_formatter(lon_formatter)
                    plot_axs[row][col].set_xlabel('Longitude')
    
                if col == 0 and lat_ticks is not None:
                    plot_axs[row][col].set_yticks(lat_ticks, crs=ccrs.PlateCarree())
                    lat_formatter = cticker.LatitudeFormatter()
                    plot_axs[row][col].yaxis.set_major_formatter(lat_formatter)
                    plot_axs[row][col].set_ylabel('Latitude')
            except RuntimeError:
                pass
    
            plot_axs[row][col].set_title(titles_grid[row][col])
            plot_axs[row][col].coastlines()
            
        cbar_ax = fig.add_subplot(gs[row+1, col])
        cbar = plt.colorbar(im, cax=cbar_ax, label=cbar_labels[col], orientation='horizontal')
        cbar.ax.tick_params(labelsize=10)
    return fig, plot_axs
    
def plot_map_grid_no_shared_colorbar(da_grid,
                                     ncols,
                                nrows,
                                vmax_grid,
                                vmin_grid,
                                title_grid,
                                cbar_label_grid,
                                cmap_grid,
                                projection = ccrs.PlateCarree(central_longitude=180),
                                cbar_loc = 'bottom',
                                cbar_frac = 0.08,
                                cbar_shrink=0.7,
                                width_height_ratio = [8,6],
                                  shrink_factor=0.7, 
                                  wspace=0.001,
                                  cbar_height_ratio=0.02
                                ):
   

    fig = plt.figure(constrained_layout=True, figsize=(shrink_factor*width_height_ratio[0]*ncols, shrink_factor*width_height_ratio[1]*nrows))
    gs = gridspec.GridSpec(2*nrows, ncols, figure=fig, 
                        width_ratios=[1]* ncols,
                        height_ratios=[1, cbar_height_ratio] * nrows,
                           wspace=wspace) 

    plot_axs = []
    for row in range(nrows):
        for col in range(ncols):

            plot_ax = fig.add_subplot(gs[2*row, col], projection = projection)
        
            im = da_grid[row][col].plot(ax=plot_ax, 
                         vmin=vmin_grid[row][col], 
                         vmax=vmax_grid[row][col], 
                         cmap=cmap_grid[row][col], 
                         transform=ccrs.PlateCarree(), 
                         rasterized=True,
                         add_colorbar=False)
    
            
            cbar_ax = fig.add_subplot(gs[2*row+1, col])
            plt.colorbar(im, 
                         cax=cbar_ax, 
                         label=cbar_label_grid[row][col], 
                         fraction=cbar_frac, 
                         location=cbar_loc,
                        shrink=cbar_shrink)
        
            plot_ax.set_xlabel('Longitude')
            plot_ax.set_ylabel('Latitude')
            plot_ax.set_title(title_grid[row][col])
            plot_ax.coastlines()
    
            plot_axs.append(plot_axs)
    return fig, plot_axs