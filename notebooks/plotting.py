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


def plot_map_grid(da_grid,
                 cbar_labels,
                 titles_grid,
                 vmax_vals,
                 vmin_vals,
                 projection,
                 cmaps,
                 column_groups,
                 width_height_ratio=[8, 6],
                 shrink_factor=1,
                 wspace=0.001,
                 cbar_height_ratio=0.02,
                 lat_ticks=None,
                 lon_ticks=None,
                 cbar_max_width=1.0):
    """Plot a map grid with one colorbar per row and column group.

    column_groups contains contiguous, zero-based column indices, for example
    [[0, 1], [2]] to share colorbars across columns 0–1 and column 2.
    cbar_max_width is the maximum colorbar width in subplot-width units.
    """
    num_rows = len(da_grid)
    if num_rows == 0 or not da_grid[0]:
        raise ValueError("da_grid must contain at least one row and column")

    num_cols = len(da_grid[0])
    if any(len(row) != num_cols for row in da_grid):
        raise ValueError("All rows in da_grid must have the same number of columns")
    if cbar_max_width <= 0:
        raise ValueError("cbar_max_width must be greater than zero")

    groups = [list(group) for group in column_groups]
    flattened = [col for group in groups for col in group]
    if (not groups
            or any(not group for group in groups)
            or sorted(flattened) != list(range(num_cols))):
        raise ValueError("column_groups must partition all columns exactly once")
    for group in groups:
        if group != list(range(min(group), max(group) + 1)):
            raise ValueError("Each column group must contain contiguous columns")

    fig = plt.figure(
        constrained_layout=True,
        figsize=(
            num_cols * shrink_factor * width_height_ratio[0],
            num_rows * shrink_factor * width_height_ratio[1],
        ),
    )
    gs = gridspec.GridSpec(
        num_rows * 2,
        num_cols,
        figure=fig,
        width_ratios=[1] * num_cols,
        height_ratios=[1, cbar_height_ratio] * num_rows,
        wspace=wspace,
    )

    plot_axs = [
        [fig.add_subplot(gs[2 * row, col], projection=projection)
         for col in range(num_cols)]
        for row in range(num_rows)
    ]
    colorbar_axes = []
    row_images = []

    for row in range(num_rows):
        images = []
        for col in range(num_cols):
            ax = plot_axs[row][col]
            im = da_grid[row][col].plot(
                ax=ax,
                vmax=vmax_vals[row],
                vmin=vmin_vals[row],
                cmap=cmaps[row],
                add_colorbar=False,
                rasterized=True,
                transform=ccrs.PlateCarree(),
            )
            images.append(im)

            try:
                if lon_ticks is not None:
                    ax.set_xticks(lon_ticks, crs=ccrs.PlateCarree())
                    ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
                    ax.set_xlabel("Longitude")
                if col == 0 and lat_ticks is not None:
                    ax.set_yticks(lat_ticks, crs=ccrs.PlateCarree())
                    ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())
                    ax.set_ylabel("Latitude")
            except RuntimeError:
                pass

            ax.set_title(titles_grid[row][col])
            ax.coastlines()

        row_images.append(images)
        for group in groups:
            colorbar_ax = fig.add_subplot(
                gs[2 * row + 1, min(group):max(group) + 1]
            )
            colorbar_axes.append((colorbar_ax, row, group))

    # Measure the laid-out subplot widths, then cap and center each colorbar.
    fig.canvas.draw()
    subplot_width = plot_axs[0][0].get_position().width
    fig.set_constrained_layout(False)

    for colorbar_ax, row, group in colorbar_axes:
        im = row_images[row][group[0]]
        plt.colorbar(
            im,
            cax=colorbar_ax,
            label=cbar_labels[row],
            orientation="horizontal",
        )
        colorbar_ax.tick_params(labelsize=10)

        bbox = colorbar_ax.get_position()
        width = min(bbox.width, cbar_max_width * subplot_width)
        colorbar_ax.set_position([
            bbox.x0 + (bbox.width - width) / 2,
            bbox.y0,
            width,
            bbox.height,
        ])

    return fig, plot_axs


def plot_map_grid(da_grid,
                 cbar_labels,
                 titles_grid,
                 vmax_vals,
                 vmin_vals,
                 projection,
                 cmaps,
                 column_groups,
                 width_height_ratio=[8, 6],
                 shrink_factor=1,
                 wspace=0.001,
                 cbar_height_ratio=0.02,
                 lat_ticks=None,
                 lon_ticks=None,
                 cbar_max_width=1.0,
                 bottom_row_colorbars=False):
    """Plot a map grid with colorbars shared by specified column groups.

    column_groups contains contiguous, zero-based column indices, for example
    [[0, 1], [2]] to share colorbars across columns 0–1 and column 2.
    cbar_max_width is the maximum colorbar width in subplot-width units.
    If bottom_row_colorbars is True, show colorbars only below the last row.
    """
    num_rows = len(da_grid)
    if num_rows == 0 or not da_grid[0]:
        raise ValueError("da_grid must contain at least one row and column")

    num_cols = len(da_grid[0])
    if any(len(row) != num_cols for row in da_grid):
        raise ValueError("All rows in da_grid must have the same number of columns")
    if cbar_max_width <= 0:
        raise ValueError("cbar_max_width must be greater than zero")

    groups = [list(group) for group in column_groups]
    flattened = [col for group in groups for col in group]
    if (not groups
            or any(not group for group in groups)
            or sorted(flattened) != list(range(num_cols))):
        raise ValueError("column_groups must partition all columns exactly once")
    for group in groups:
        if group != list(range(min(group), max(group) + 1)):
            raise ValueError("Each column group must contain contiguous columns")

    if bottom_row_colorbars:
        grid_rows = num_rows + 1
        height_ratios = [1] * num_rows + [cbar_height_ratio]
    else:
        grid_rows = num_rows * 2
        height_ratios = [1, cbar_height_ratio] * num_rows

    fig = plt.figure(
        constrained_layout=True,
        figsize=(
            num_cols * shrink_factor * width_height_ratio[0],
            num_rows * shrink_factor * width_height_ratio[1],
        ),
    )
    gs = gridspec.GridSpec(
        grid_rows,
        num_cols,
        figure=fig,
        width_ratios=[1] * num_cols,
        height_ratios=height_ratios,
        wspace=wspace,
    )

    plot_axs = [
        [
            fig.add_subplot(
                gs[row if bottom_row_colorbars else 2 * row, col],
                projection=projection,
            )
            for col in range(num_cols)
        ]
        for row in range(num_rows)
    ]

    colorbar_axes = []
    row_images = []

    for row in range(num_rows):
        images = []
        for col in range(num_cols):
            ax = plot_axs[row][col]
            im = da_grid[row][col].plot(
                ax=ax,
                vmax=vmax_vals[row],
                vmin=vmin_vals[row],
                cmap=cmaps[row],
                add_colorbar=False,
                rasterized=True,
                transform=ccrs.PlateCarree(),
            )
            images.append(im)

            try:
                if lon_ticks is not None:
                    ax.set_xticks(lon_ticks, crs=ccrs.PlateCarree())
                    ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
                    ax.set_xlabel("Longitude")
                if col == 0 and lat_ticks is not None:
                    ax.set_yticks(lat_ticks, crs=ccrs.PlateCarree())
                    ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())
                    ax.set_ylabel("Latitude")
            except RuntimeError:
                pass

            ax.set_title(titles_grid[row][col])
            ax.coastlines()

        row_images.append(images)

        if not bottom_row_colorbars or row == num_rows - 1:
            cbar_row = num_rows if bottom_row_colorbars else 2 * row + 1
            for group in groups:
                colorbar_ax = fig.add_subplot(
                    gs[cbar_row, min(group):max(group) + 1]
                )
                colorbar_axes.append((colorbar_ax, row, group))

    # Measure subplot widths, then cap and center each colorbar.
    fig.canvas.draw()
    subplot_width = plot_axs[0][0].get_position().width
    fig.set_constrained_layout(False)

    for colorbar_ax, row, group in colorbar_axes:
        plt.colorbar(
            row_images[row][group[0]],
            cax=colorbar_ax,
            label=cbar_labels[row],
            orientation="horizontal",
        )
        colorbar_ax.tick_params(labelsize=10)

        bbox = colorbar_ax.get_position()
        width = min(bbox.width, cbar_max_width * subplot_width)
        colorbar_ax.set_position([
            bbox.x0 + (bbox.width - width) / 2,
            bbox.y0,
            width,
            bbox.height,
        ])

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