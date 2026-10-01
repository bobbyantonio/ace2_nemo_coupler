# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import os
import numpy as np
import xarray as xr
import xesmf as xe

# %%
input_grid = 'ORCA025'
output_grid = 'ORCA1'
init_date='20101101'
esm_component = 'oce'
input_base_folder = '/home/ecme4254/hpcperm/ece3data/nemo/restart/'

input_folder = os.path.join(input_base_folder, input_grid, init_date)
output_folder =  os.path.join(input_base_folder, output_grid, init_date)

output_grid_ds = xr.load_dataset(f'/home/ecme4254/hpcperm/ece3data/nemo/domain/{output_grid}/domain_cfg.nc').isel(t=0)
output_grid_ds = output_grid_ds.assign_coords(lon=output_grid_ds['nav_lon'], lat=output_grid_ds['nav_lat'])

# %%
output_mask_ds = xr.load_dataset(f'/home/ecme4254/hpcperm/ece3data/nemo/domain/{output_grid}/maskutil.nc')

# %%
input_oce_fp = f'/hpcperm/ecme4254/ece4data/nemo/restart/{input_grid}/{init_date}/restart_oce.nc'
input_ice_fp = f'/hpcperm/ecme4254/ece4data/nemo/restart/{input_grid}/{init_date}/restart_ice.nc'

input_oce_ds = xr.load_dataset(input_oce_fp)
input_oce_ds = input_oce_ds.assign_coords(lon=input_oce_ds['nav_lon'], lat=input_oce_ds['nav_lat'])

input_ice_ds = xr.load_dataset(input_ice_fp)
input_ice_ds = input_ice_ds.assign_coords(lon=input_ice_ds['nav_lon'], lat=input_ice_ds['nav_lat'])

regridder = xe.Regridder(input_oce_ds, output_grid_ds, "nearest_s2d", unmapped_to_nan=True, ignore_degenerate=True)
output_oce_ds = regridder(input_oce_ds)
output_ice_ds = regridder(input_ice_ds)

# %%
# Compare with another restart file
example_oce_ds = xr.load_dataset('/home/ecme4254/hpcperm/ece3data/nemo/restart/ORCA1/20000101/restart_oce.nc')
example_ice_ds = xr.load_dataset('/home/ecme4254/hpcperm/ece3data/nemo/restart/ORCA1/20000101/restart_ice.nc')

# %%
input_ice_ds

# %%
example_ice_ds

# %%
name_mapping = {
'avm': 'avm_k',
'avmu': None, # Get from regridding avm
'avmv': None, # Get from regridding avm
'avt': 'avt_k',
'dissl':'dissl',
'emp_b': 'emp_b',
'en': 'en',
'fraqsr_1lev': 'fraqsr_1lev',
'fse3t_b': 'e3t_b',
'fse3t_n': 'e3t_n',
'hdivb': None, # Get from calculating directly, do we need to add noise as well?
'hdivn': None, # Get from calculating directly
'nav_lat':'nav_lat',
'nav_lev': None, # From domain cfg
'nav_lon': 'nav_lon',
'ndastp': 'ndastp',
'qns_b': 'qns_b',
'qsr_hc_b': 'qsr_hc_b',
'rhop': 'rhop',
'rnf_b': 'rnf_b',
'rnf_hc_b': 'rnf_hc_b',
'rnf_sc_b': 'rnf_sc_b',
'rotb': None, # Get from calculating directly?
'rotn': None, # Get from calculating directly?
'sb': None, # Set equal to now?
'sbc_hc_b': 'sbc_hc_b',
'sbc_sc_b': 'sbc_sc_b',
'sfx_b': 'sfx_b',
'sn': 'sn',
'sshb': None, # Set equal to now?
'sshn': 'sshn',
'tb': None, # Set equal to now?
'tn': 'tn',
'ub': None, # Set equal to now?
'ub2_b': None, # Set equal to now?
'un': 'un',
'utau_b': 'utau_b',
'vb': None, # Set equal to now?
'vb2_b': None, # Set equal to now?
'vn': 'vn',
'vtau_b': 'vtau_b'}

output_ds = regridder(input_ds)
output_ds = output_ds.rename_dims({'time_counter': 't', 'nav_lev': 'z'})
# Have to reinstate float variables after regridding
for v in ['adatrj', 'kt', 'ndastp', 'rdt']:
    output_ds[v] = example_ds[v]

existing_mappings = {v: k for k, v in name_mapping.items() if v is not None }
output_ds = output_ds.rename(existing_mappings)[[k for k, v in name_mapping.items() if v is not None]]

# Set before fields to same as now (since 4.2 doesn'thave before fields in the restart)
for var in ['ssh', 's', 't', 'u', 'v']:
    output_ds[f'{var}b'] = output_ds[f'{var}n']
    
output_ds['avmu'] = output_ds['avm']
output_ds['avmv'] = output_ds['avm']

# For half-step fluxes (only at surface), set these to what they are in the example dataset
for v in ['ub2_b', 'vb2_b']:
    output_ds[v] = example_ds[v]
# rotb, rotn, hdivb, hdivn

# %%
output_ds = output_ds.drop_indexes(['nav_lev', 'time_counter']).reset_coords()

# %%

# %% [markdown]
# # From NEMO3.6 docs / code:
# adatrj: date in days since the beginning of the run 
# avm: vertical eddy viscosity
# avmu: vertical viscosity coef at uw-pts [m2/s]. Just a regridding of avm to U and V grid?
# avmv: vertical viscosity coef at vw-pts [m2/s]
# avt: vertical diffusivity coefficients
# dissl: mixing lenght of dissipation
# emp_b: 'before' freshwater flux
# en: turbulent kinetic energy
# fraqsr_1lev:  fraction of solar net radiation absorbed in the first ocean level
# fse3t_b: Horizontal scale factor interpolations before
# fse3t_n: Vertical scale factor interpolations now
# hdivb: horizontal divergence before (of veolicity I think?)
# hdivn: horizontal divergence now (of veolicity I think?)
#
#  hdivn(ji,jj,jk) = &
#                   ( e2u(ji,jj)*e3u_n(ji,jj,jk) * un(ji,jj,jk) - e2u(ji-1,jj)*e3u_n(ji-1,jj,jk) * un(ji-1,jj,jk) &
#                    + e1v(ji,jj)*e3v_n(ji,jj,jk) * vn(ji,jj,jk) - e1v(ji,jj-1)*e3v_n(ji,jj-1,jk) * vn(ji,jj-1,jk) ) &
#                   / ( e1t(ji,jj) * e2t(ji,jj) * e3t_n(ji,jj,jk) )
#
# where 
# e2u etc taken from domain config file
#
# kt: time step number
# 'nav_lat':
# 'nav_lev':
# 'nav_lon':
# 'ndastp': from namelist I think? Something to do with time
# 'qns_b': sea heat flux, non solar, before
# 'qsr_hc_b': heat content trend due to qsr flux
# 'rdt': time step in secs
# 'rdttra1': surface tracer time step
# 'rhop': density
# 'rnf_b': runoff before
# 'rnf_hc_b': before heat content of runoff
# 'rnf_sc_b': before salinity content of runoff
# 'rotb': relative vorticity  before 
# 'rotn': relative vorticity  now
#
# IN NEMO/OPA_SRC/DYN/divcur.f90
#
# rotn(ji,jj,jk) = (  zwv(ji+1,jj  ) - zwv(ji,jj)      &
#                   &              - zwu(ji  ,jj+1) + zwu(ji,jj)  ) * fmask(ji,jj,jk) / ( e1f(ji,jj)*e2f(ji,jj) )
#
# where 
# zwu(ji,jj) = e1u(ji,jj) * un(ji,jj,jk)
# zwv(ji,jj) = e2v(ji,jj) * vn(ji,jj,jk)
#
# plus accounting for periodicity of the grid, and some values set to zero. 
#
#
# 'sb': Salinity before
# 'sbc_hc_b': before heat content sbc trend
# 'sbc_sc_b': before salt content sbc trend
# 'sfx_b': salt flux before
# 'sn': Salinity now
# 'sshb': SSH before
# 'sshn': SSH now
# 'tb': temp before
# 'time_counter'
# 'tn': temp now
# 'ub': U before
# 'ub2_b': Half step fluxes
# 'un': U now
# 'utau_b': sea surface i-stress (ocean referential)
# 'vb': V before
# 'vb2_b': Half step fluxes
# 'vn': V now
# 'vtau_b': sea surface i-stress (ocean referential)
#

# %% [markdown]
# ## From NEMO 4.2 docs
#
# Excluding items that have the same name in 3.0.6
#
# 'e3t_b': T-cell thickness ? Scale factor interpolations?  Think this maps to fse3t_b
# 'e3t_ini': scale factor interpolations; think this maps to ...
# 'e3t_n': Think this maps to fse3t_n
# 'frc_s': global forcing trends (salinity?)
# 'frc_t': global forcing trends (temp?)
# 'frc_v': global forcing trends (v?)
# 'fwb_flx_m': ?? fwb meansfreshwater budget I think?
# 'fwb_flx_mb': ??
# 'fwb_flx_mn': ??
# 'fwb_flx_v': ??
# 'fwb_flx_vb': ??
# 'fwb_flx_vn': ??
# 'fwb_hbp0':
# 'fwb_hbp00':
# 'fwb_hst0':
# 'fwb_hst00': 
# 'fwb_kt':
# 'fwb_nr_ssh':
# 'fwb_om0':
# 'fwb_om00':
# 'fwb_sl0':
# 'fwb_sl00':
# 'fwb_ssh0':
# 'fwb_ssh00':
# 'fwf_isf_b': freshwater fluxes?
# 'hc_loc_ini': initial heat content
# 'isf_hc_b': ice shelf variables? heat content?
# 'isf_sc_b': ice shelf variables? salinity content?
# 'nn_fwb': Namelist parametre for freshwater budget
# 'ntime': some kind of time step index I think
# 'sc_loc_ini': Initial salt content
# 'ssh_ini': SSH initial
# 'surf_ini': Initial 'surface'. Is this SST then?

# %%
# Mapping from NEMO 4.0.6 data
{'adatrj': 'adatrj',
'avm': 'avm_k',
'avmu': None, # Get from regridding avm
'avmv': None, # Get from regridding avm
'avt': 'avt_k',
'dissl':'dissl',
'emp_b': 'emp_b',
'en': 'en',
'fraqsr_1lev': 'fraqsr_1lev',
'fse3t_b': 'e3t_b',
'fse3t_n': 'e3t_n',
'hdivb': None, # Get from calculating directly, do we need to add noise as well?
'hdivn': None, # Get from calculating directly
'kt': 'kt',
'nav_lat':'nav_lat',
'nav_lev': None, # From domain cfg
'nav_lon': 'nav_lon',
'ndastp': 'ndastp',
'qns_b': 'qns_b',
'qsr_hc_b': 'qsr_hc_b',
'rdt': 'rdt',
'rdttra1': 'rdt',
'rhop': 'rhop',
'rnf_b': 'rnf_b',
'rnf_hc_b': 'rnf_hc_b',
'rnf_sc_b': 'rnf_sc_b',
'rotb': None, # Get from calculating directly?
'rotn': None, # Get from calculating directly?
'sb': None, # Set equal to now?
'sbc_hc_b': 'sbc_hc_b',
'sbc_sc_b': 'sbc_sc_b',
'sfx_b': 'sfx_b',
'sn': 'sn',
'sshb': None, # Set equal to now?
'sshn': 'sshn',
'tb': None, # Set equal to now?
'time_counter'
'tn': 'tn',
'ub': None, # Set equal to now?
'ub2_b': None, # Set equal to now?
'un': 'un',
'utau_b': 'utau_b',
'vb': None, # Set equal to now?
'vb2_b': None, # Set equal to now?
'vn': 'vn'
'vtau_b': 'vtau_b'}

# %%
output_domain_cfg = xr.load_dataset(f'/home/ecme4254/hpcperm/ece3data/nemo/domain/{output_grid}/domain_cfg.nc')

# %%
# From nemo-3.6/CONFIG/ORCA1L75_LIM3/BLD/ppsrc/nemo/par_oce.f90

# !!----------------------------------------------------------------------
#    !! Domain decomposition
#    !!----------------------------------------------------------------------
#    !! if we dont use massively parallel computer (parameters jpni=jpnj=1) so jpiglo=jpi and jpjglo=jpj
#    INTEGER, PUBLIC :: jpni !: number of processors following i
#    INTEGER, PUBLIC :: jpnj !: number of processors following j
#    INTEGER, PUBLIC :: jpnij !: nb of local domain = nb of processors ( <= jpni x jpnj )
#    INTEGER, PUBLIC, PARAMETER :: jpr2di = 0 !: number of columns for extra outer halo
#    INTEGER, PUBLIC, PARAMETER :: jpr2dj = 0 !: number of rows for extra outer halo
#    INTEGER, PUBLIC, PARAMETER :: jpreci = 1 !: number of columns for overlap
#    INTEGER, PUBLIC, PARAMETER :: jprecj = 1 !: number of rows for overlap
# INTEGER, PUBLIC :: jpi ! = ( jpiglo-2*jpreci + (jpni-1) ) / jpni + 2*jpreci !: first dimension
#  INTEGER, PUBLIC :: jpj ! = ( jpjglo-2*jprecj + (jpnj-1) ) / jpnj + 2*jprecj !: second dimension
#  INTEGER, PUBLIC :: jpk ! = jpkdta
#  INTEGER, PUBLIC :: jpim1 ! = jpi-1 !: inner domain indices
#  INTEGER, PUBLIC :: jpjm1 ! = jpj-1 !: - - -
#  INTEGER, PUBLIC :: jpkm1 ! = jpk-1 !: - - -
#  INTEGER, PUBLIC :: jpij ! = jpi*jpj !: jpi x jpj

# %%
# From nemo-3.6/CONFIG/ORCA1L75_LIM3/BLD/inc/vectopt_loop_substitute.h90
# Doesn't look like it is set in our case

#if defined key_vectopt_loop
#  define   fs_2       1
#  define   fs_jpim1   jpi
#else
#  define   fs_2       2
#  define   fs_jpim1   jpim1
#endif

# %%
jpni = 1
jpnj = 1
jpreci = 1
jprecj = 1
jpiglo = output_domain_cfg['jpiglo'].item()
jpjglo = output_domain_cfg['jpjglo'].item()
jpkglo = output_domain_cfg['jpkglo'].item()

jpi  = jpiglo
jpj = jpjglo
jpkdata = jpkglo
jpk = jpkdata

jpim1 = jpi-1 # Inner domain indices
jpjm1 = jpj-1
jpkm1 = jpk-1

fs_2  = 2
fs_jpim1  = jpim1

# %%
output_mask_ds['tmaskutil'].isel(time_counter=0).plot(vmin=-1, vmax=1)

# %%

# %%
# Code adapted from nemp-3.6/CONFIG/WORK/divcur.f90

e2u = output_domain_cfg['e2u']
e3u_n = output_domain_cfg['e3u_0']
e1v = output_domain_cfg['e1v']
e3v_n = output_domain_cfg['e3v_0']
e1t  = output_domain_cfg['e1t']
e2t  = output_domain_cfg['e2t']
e3t_n  = output_domain_cfg['e3t_0']
e1f = output_domain_cfg['e1f']
e2f = output_domain_cfg['e2f']
nperio = output_domain_cfg['nperio'].item()
vn = output_ds['vn'].values()
un = output_ds['un'].values()
tn output_ds['tn'].values()


nbondi = 
nlci = 
nlcj = 
tmask = 

mbathy = output_domain_cfg['bottom_level']


# %%
def calculate_hdivn(jpkm1, jpjm1, jpi, jpj, fs_2, fs_jpim1, e2u, fse3u, un, e1v,fse3v, vn, e1t, e2t, fse3t, nbondi, nlci, nlcj, e1f, e2f, fmask):

    hdivn = np.zeros((jpim1,jpjm1,jk))
    rotn = np.zeros((jpim1,jpjm1,jk))
    zwu = np.zeros((jpim1,jpjm1))
    zwv = np.zeros((jpim1,jpjm1))
    
    for jk in range(jpkm1):  # jk = 0 to jpkm1-1
        # Time swap of div and rot arrays
    
        # Horizontal divergence
        for jj in range(1, jpjm1):  # jj = 1 to jpjm1-1
            for ji in range(fs_2-1, fs_jpim1):  # adjust fs_2 for 0-based
                hdivn[ji, jj, jk] = (
                    e2u[ji, jj] * fse3u[ji, jj, jk] * un[ji, jj, jk]
                    - e2u[ji-1, jj] * fse3u[ji-1, jj, jk] * un[ji-1, jj, jk]
                    + e1v[ji, jj] * fse3v[ji, jj, jk] * vn[ji, jj, jk]
                    - e1v[ji, jj-1] * fse3v[ji, jj-1, jk] * vn[ji, jj-1, jk]
                ) / (e1t[ji, jj] * e2t[ji, jj] * fse3t[ji, jj, jk])
    
        if not AGRIF_Root():
            if nbondi in (1, 2):
                hdivn[nlci-2, :, jk] = 0.0  # east
            if nbondi in (-1, 2):
                hdivn[1, :, jk] = 0.0       # west
            if nbondj in (1, 2):
                hdivn[:, nlcj-2, jk] = 0.0  # north
            if nbondj in (-1, 2):
                hdivn[:, 1, jk] = 0.0       # south
    
        # Contravariant velocity (inside model domain)
        for jj in range(jpj):  # jj = 0 to jpj-1
            for ji in range(jpi):  # ji = 0 to jpi-1
                zwu[ji, jj] = e1u[ji, jj] * un[ji, jj, jk]
                zwv[ji, jj] = e2v[ji, jj] * vn[ji, jj, jk]
    
        # East-West boundary conditions
        if nperio in (1, 4, 6):
            zwv[0, :] = zwv[jpi-2, :]
            zwv[-1, :] = zwv[jpi-3, :]
            zwv[jpi, :] = zwv[2, :]
            zwv[jpi+1, :] = zwv[3, :]
        else:
            zwv[0, :] = 0.0
            zwv[-1, :] = 0.0
            zwv[jpi, :] = 0.0
            zwv[jpi+1, :] = 0.0
    
        # North-South boundary conditions
        if nperio in (3, 4):
            zwu[jpi-1, jpj] = 0.0
            zwu[jpi-1, jpj+1] = 0.0
            for ji in range(jpi-1):
                iju = jpi - ji - 1
                zwu[ji, jpj] = -zwu[iju, jpj-3]
                zwu[ji, jpj+1] = -zwu[iju, jpj-4]
        elif nperio in (5, 6):
            zwu[jpi-1, jpj] = 0.0
            zwu[jpi-1, jpj+1] = 0.0
            for ji in range(jpi-1):
                iju = jpi - ji - 1
                zwu[ji, jpj-1] = -zwu[iju, jpj-2]
                zwu[ji, jpj] = -zwu[iju, jpj-3]
                zwu[ji, jpj+1] = -zwu[iju, jpj-4]
            for ji in range(-1, jpi+2):
                ijt = jpi - ji
                zwv[ji, jpj-1] = -zwv[ijt, jpj-3]
            for ji in range(jpi//2, jpi+2):
                ijt = jpi - ji
                zwv[ji, jpjm1-1] = -zwv[ijt, jpjm1-1]
        else:
            zwu[:, jpj] = 0.0
            zwu[:, jpj+1] = 0.0
    
        # Relative vorticity (vertical component of velocity curl)
        for jj in range(jpjm1):  # jj = 0 to jpjm1-1
            for ji in range(fs_jpim1):  # ji = 0 to fs_jpim1-1
                rotn[ji, jj, jk] = (
                    zwv[ji+1, jj] - zwv[ji, jj]
                    - zwu[ji, jj+1] + zwu[ji, jj]
                ) * fmask[ji, jj, jk] / (e1f[ji, jj] * e2f[ji, jj])
    
        # Second order accurate scheme along straight coast
        for jl in range(npcoa[0, jk]):
            ii = nicoa[jl, 0, jk]
            ij = njcoa[jl, 0, jk]
            rotn[ii, ij, jk] = 1.0 / (e1f[ii, ij] * e2f[ii, ij]) * (
                4.0 * zwv[ii+1, ij] - zwv[ii+2, ij] + 0.2 * zwv[ii+3, ij]
            )
        for jl in range(npcoa[1, jk]):
            ii = nicoa[jl, 1, jk]
            ij = njcoa[jl, 1, jk]
            rotn[ii, ij, jk] = 1.0 / (e1f[ii, ij] * e2f[ii, ij]) * (
                -4.0 * zwv[ii, ij] + zwv[ii-1, ij] - 0.2 * zwv[ii-2, ij]
            )
        for jl in range(npcoa[2, jk]):
            ii = nicoa[jl, 2, jk]
            ij = njcoa[jl, 2, jk]
            rotn[ii, ij, jk] = -1.0 / (e1f[ii, ij] * e2f[ii, ij]) * (
                4.0 * zwu[ii, ij+1] - zwu[ii, ij+2] + 0.2 * zwu[ii, ij+3]
            )
        for jl in range(npcoa[3, jk]):
            ii = nicoa[jl, 3, jk]
            ij = njcoa[jl, 3, jk]
            rotn[ii, ij, jk] = -1.0 / (e1f[ii, ij] * e2f[ii, ij]) * (
                -4.0 * zwu[ii, ij] + zwu[ii, ij-1] - 0.2 * zwu[ii, ij-2]
            )

    return hdivn, rotn


def dom_msk_nsa(jpk, jpjm1, jpim1, jpi, jpj, jpkm1, jperio,  tmask, fmask):
    # Assume all arrays (tmask, fmask, npcoa, nicoa, njcoa, icoord, etc.) are numpy arrays
    # and all variables (jpk, jpjm1, jpim1, jpi, jpj, jpkm1, etc.) are defined in the calling scope
     # ALLOCATE( npcoa(4,jpk), nicoa(2*(jpi+jpj),4,jpk), njcoa(2*(jpi+jpj),4,jpk), STAT=ierr(12) )

    npcoa = np.zeros((4,jpk))
    nicoa = np.zeros((2*(jpi+jpj),4,jpk))
    njcoa = np.zeros((2*(jpi+jpj),4,jpk))
    
    # Initialize
    for jk in range(jpk):
        for jl in range(4):
            npcoa[jl, jk] = 0
            for ji in range(2 * (jpi + jpj)):
                nicoa[ji, jl, jk] = 0
                njcoa[ji, jl, jk] = 0

    if jperio == 2:
        print(' ')
        print(' symetric boundary conditions need special')
        print(' treatment not implemented. we stop.')
        raise RuntimeError('Symmetric boundary conditions not implemented')

    # Convex corners
    for jk in range(jpkm1):
        for jj in range(jpjm1):
            for ji in range(jpim1):
                zaa = (tmask[ji, jj, jk] + tmask[ji, jj+1, jk] +
                       tmask[ji+1, jj, jk] + tmask[ji+1, jj+1, jk])
                if abs(zaa - 3.0) <= 0.1:
                    fmask[ji, jj, jk] = 1.0

    # North-south straight coast
    for jk in range(jpkm1):
        inw = 0
        ine = 0
        for jj in range(1, jpjm1):  # Fortran 2:jpjm1 -> Python 1:jpjm1-1
            for ji in range(1, jpim1):  # Fortran 2:jpim1 -> Python 1:jpim1-1
                zaa = tmask[ji+1, jj, jk] + tmask[ji+1, jj+1, jk]
                if abs(zaa - 2.0) <= 0.1 and fmask[ji, jj, jk] == 0.0:
                    nicoa[inw, 0, jk] = ji
                    njcoa[inw, 0, jk] = jj
                    inw += 1
                zaa = tmask[ji, jj, jk] + tmask[ji, jj+1, jk]
                if abs(zaa - 2.0) <= 0.1 and fmask[ji, jj, jk] == 0.0:
                    nicoa[ine, 1, jk] = ji
                    njcoa[ine, 1, jk] = jj
                    ine += 1
        npcoa[0, jk] = inw
        npcoa[1, jk] = ine

    # West-east straight coast
    for jk in range(jpkm1):
        ins = 0
        inn = 0
        for jj in range(1, jpjm1):
            for ji in range(1, jpim1):
                zaa = tmask[ji, jj+1, jk] + tmask[ji+1, jj+1, jk]
                if abs(zaa - 2.0) <= 0.1 and fmask[ji, jj, jk] == 0.0:
                    nicoa[ins, 2, jk] = ji
                    njcoa[ins, 2, jk] = jj
                    ins += 1
                zaa = tmask[ji+1, jj, jk] + tmask[ji, jj, jk]
                if abs(zaa - 2.0) <= 0.1 and fmask[ji, jj, jk] == 0.0:
                    nicoa[inn, 3, jk] = ji
                    njcoa[inn, 3, jk] = jj
                    inn += 1
        npcoa[2, jk] = ins
        npcoa[3, jk] = inn

    itest = 2 * (jpi + jpj)
    for jk in range(jpk):
        if (npcoa[0, jk] > itest or npcoa[1, jk] > itest or
            npcoa[2, jk] > itest or npcoa[3, jk] > itest):
            raise RuntimeError(f"Straight coast index arrays are too small at level {jk}")

    ierror = 0
    iind = 0
    ijnd = 0
    if nperio in (1, 4, 6):
        iind = 2
    if nperio in (3, 4, 5, 6):
        ijnd = 2

    for jk in range(jpk):
        for jl in range(npcoa[0, jk]):
            if nicoa[jl, 0, jk] + 3 > jpi + iind:
                icoord[ierror, 0] = nicoa[jl, 0, jk]
                # icoord[ierror, 1] = njcoa[jl, 0, jk]
                # icoord[ierror, 2] = jk
                raise RuntimeError('We stop 1...')
        for jl in range(npcoa[1, jk]):
            if nicoa[jl, 1, jk] - 2 < 1 - iind:
                # icoord[ierror, 0] = nicoa[jl, 1, jk]
                # icoord[ierror, 1] = njcoa[jl, 1, jk]
                # icoord[ierror, 2] = jk
                raise RuntimeError('We stop 2...')
        for jl in range(npcoa[2, jk]):
            if njcoa[jl, 2, jk] + 3 > jpj + ijnd:
                # icoord[ierror, 0] = nicoa[jl, 2, jk]
                # icoord[ierror, 1] = njcoa[jl, 2, jk]
                # icoord[ierror, 2] = jk
                raise RuntimeError('We stop 3...')
        for jl in range(npcoa[3, jk]):
            if njcoa[jl, 3, jk] - 2 < 1:
                # icoord[ierror, 0] = nicoa[jl, 3, jk]
                # icoord[ierror, 1] = njcoa[jl, 3, jk]
                # icoord[ierror, 2] = jk
                raise RuntimeError('We stop 4...')
    return npcoa, nicoa, njcoa

# End of dom_msk_nsa

def dom_msk(jpi, jpj, jpk, jpjm1, jpim1, mbathy, misfdep, rn_shlat):
    # nemo-3.6/CONFIG/ORCA1L75_LIM3/BLD/ppsrc/nemo/dommsk.f90

    # misfdep !: top first ocean level (ISF)
    
    # Assume all arrays (tmask, umask, vmask, fmask, bmask, etc.) are numpy arrays
    # and all variables (jpi, jpj, jpk, jpjm1, jpim1, etc.) are defined in the calling scope
    tmask = np.zeros((jpi,jpj,jpk))
    ssmask = np.zeros((jpi,jpj,jpk))
    
    # 1. Ocean/land mask at t-point (computed from mbathy)

    for jk in range(jpk):
        for jj in range(jpj):
            for ji in range(jpi):
                if float(mbathy[ji, jj] - (jk+1)) + 0.1 >= 0.0:
                    tmask[ji, jj, jk] = 1.0

    # (ISF) define barotropic mask and mask the ice shelf point
    ssmask[:, :] = tmask[:, :, 0]
    for jk in range(jpk):
        for jj in range(jpj):
            for ji in range(jpi):
                if float(misfdep[ji, jj] - (jk+1)) - 0.1 >= 0.0:
                    tmask[ji, jj, jk] = 0.0

    # 2. Ocean/land mask at u-, v-, and f-points (computed from tmask)
    for jk in range(jpk):
        for jj in range(jpjm1):
            for ji in range(jpim1):
                umask[ji, jj, jk] = tmask[ji, jj, jk] * tmask[ji+1, jj, jk]
                vmask[ji, jj, jk] = tmask[ji, jj, jk] * tmask[ji, jj+1, jk]
            for ji in range(jpim1):
                fmask[ji, jj, jk] = (tmask[ji, jj, jk] * tmask[ji+1, jj, jk] *
                                     tmask[ji, jj+1, jk] * tmask[ji+1, jj+1, jk])


    # Lateral boundary conditions on velocity (modify fmask)
    for jk in range(jpk):
        zwf = fmask[:, :, jk].copy()
        for jj in range(1, jpjm1):
            for ji in range(1, jpim1):
                if fmask[ji, jj, jk] == 0.0:
                    fmask[ji, jj, jk] = rn_shlat * min(
                        1.0, max(zwf[ji+1, jj], zwf[ji, jj+1], zwf[ji-1, jj], zwf[ji, jj-1])
                    )
        for jj in range(1, jpjm1):
            if fmask[0, jj, jk] == 0.0:
                fmask[0, jj, jk] = rn_shlat * min(
                    1.0, max(zwf[1, jj], zwf[0, jj+1], zwf[0, jj-1])
                )
            if fmask[jpi-1, jj, jk] == 0.0:
                fmask[jpi-1, jj, jk] = rn_shlat * min(
                    1.0, max(zwf[jpi-1, jj+1], zwf[jpim1-1, jj], zwf[jpi-1, jj-1])
                )
        for ji in range(1, jpim1):
            if fmask[ji, 0, jk] == 0.0:
                fmask[ji, 0, jk] = rn_shlat * min(
                    1.0, max(zwf[ji+1, 0], zwf[ji, 1], zwf[ji-1, 0])
                )
            if fmask[ji, jpj-1, jk] == 0.0:
                fmask[ji, jpj-1, jk] = rn_shlat * min(
                    1.0, max(zwf[ji+1, jpj-1], zwf[ji-1, jpj-1], zwf[ji, jpjm1-1])
                )
    return tmask, umask, vmask, fmask


# %%
output_dir = f'/hpcperm/ecme4254/ece3data/nemo/restart/{output_grid}/{init_date}'
os.makedirs(output_dir, exist_ok=True)

output_fp =  os.path.join(output_dir, f'restart_{esm_component}.nc')

if os.path.isfile(output_fp):
    raise ValueError('File already exists')

output_ds.to_netcdf(output_fp)

# %%
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

fig, ax = plt.subplots(1,2, figsize=(2*8,5))
input_ds['tn'].isel(time_counter=0).isel(nav_lev=0).plot.pcolormesh(ax=ax[0], x="x", y="y")

output_ds['tn'].isel(time_counter=0).isel(nav_lev=0).plot.pcolormesh(ax=ax[1], x="x", y="y")

# %%
