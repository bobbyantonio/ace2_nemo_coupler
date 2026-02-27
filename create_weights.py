#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import re
from netCDF4 import Dataset

#
# BEGIN USER MODIFICATIONS
#

# Directory with domccfg target file
DOMCFG_DIR="/ec/res4/hpcperm/ecme4254/ece4data/nemo/domain/ORCA025"

# Suffix of domcfg files
RAD='domain_cfg.nc'

# Directory with original forcing on native grid
FORCING_DIR='/ec/res4/hpcperm/ecme4254/ece4data/nemo/forcing'
INITIAL_CONDITION_DIR='/ec/res4/hpcperm/ecme4254/ece4data/nemo/initial'

GRID = 'ORCA025'
# Forcing file names, interpolation method (default bilin), and weigth file name (optional), lon(optional), lat(optional)  
FILES=[
['calving.nc'  ,'bilin','','','', 'forcing'],
['chlorophyll.nc'  ,'bilin','','','', 'forcing'],
['geothermal_heating.nc'  ,'bilin','','','', 'forcing'],
['Goutorbe_ghflux.nc'  ,'bilin','','','', 'forcing'],
['runoff_monthly_ORCA1.nc'               ,'bilin','','','', 'forcing'],
['runoff_monthly_ORCA025.nc'               ,'bilin','','','', 'forcing'],
['runoff-icb_DaiTrenberth_Depoorter_ORCA1_JD.nc'  ,'bilin','','','', 'forcing'],
['era5_instantaneous_eastward_turbulent_surface_stress.nc','bilin','','','', 'forcing'],
['era5_instantaneous_northward_turbulent_surface_stress.nc','bilin','','','', 'forcing'],
['era5_mean_surface_downward_short_wave_radiation_flux.nc', 'bilin', '', '', '', 'forcing'],
['era5_mean_surface_sensible_heat_flux.nc', 'bilin', '', '', '', 'forcing'],
['ncar_precip.15JUNE2009.nc', 'bilin', '', '', '', 'forcing'],
['ncar_rad.15JUNE2009.nc', 'bilin', '', '', '', 'forcing'],
['q_10.15JUNE2009.nc', 'bilin', '', '', '', 'forcing'],
['runoff.15JUNE2009.nc', 'bilin', '', '', '', 'forcing'],
['slp.15JUNE2009.nc', 'bilin', '', '', '', 'forcing'],
['t_10.15JUNE2009.nc', 'bilin', '', '', '', 'forcing'],
['u_10.15JUNE2009.nc', 'bicub', '', '', '', 'forcing'],
['v_10.15JUNE2009.nc', 'bicub', '', '', '', 'forcing'],
['oras5_potential_temperature_201510.nc'  ,'bilin','','','', 'initial']
]

#
# END USER MODIFICATIONS
#
mycmd="ls "+DOMCFG_DIR+"/"+RAD
returned_output = subprocess.check_output(mycmd, shell=True)
listcfg = (returned_output.decode("utf-8")).split()



for i in range(len(listcfg)):
	print ('Computing weights for cfg file %s :' % listcfg[i])
	print()

	for myfile in FILES:
		print('Input file is %s with %s interpolation' % (myfile[0],  myfile[1]))
		if len(myfile[2]) == 0 :
			wfile=str(i+1)+'_'+myfile[1]+'_'+myfile[0]
		else:
			wfile=str(i+1)+'_'+myfile[2]
		print('   Performing weights computation ...')
		if myfile[1]=='namelist_bicub':
			namelist='namelist_bicub'
		else :
			namelist='namelist_bilin'

		if myfile[5] == 'forcing':

			myfilename=os.path.join(FORCING_DIR, myfile[0])
		elif myfile[5] == 'initial':
			myfilename=os.path.join(INITIAL_CONDITION_DIR, myfile[0])
		else:
			raise ValueError(f'Unrecognised file descriptor: {myfile[5]}')
		
		dataset=Dataset(myfilename)
		
#		if len(myfile[3])==0: 
#			for key in dataset.variables:
#				if len(dataset.variables[key].dimensions) >=2:
#					myvar=key
#					break
#		else:
#			myvar=myfile[3]
#		print('   Interpolation based on variable %s ...' % myvar)		
		
		if len(myfile[3])==0 or len(myfile[4])==0: 
			for key in dataset.variables:
				if re.search('lat',key.lower()):
					mylat=key
				if re.search('lon',key.lower()):
					mylon=key
		else:
			mylon=myfile[3]
			mylat=myfile[4]
		print('   Interpolation based on longitude %s ...' % mylon)		
		print('   Interpolation based on latitude  %s ...' % mylat)		

		f2=open("namelist_new","w+")
		with open(namelist,"r") as f:
			for line in f:
				match = re.search('nemo_file',line)
				match2 = re.search('input_file',line)
				match3 = re.search('input_lon',line)
				match4 = re.search('input_lat',line)
				match5 = re.search('output_file',line)
				match6 = re.search('output_name',line)
				if match != None :
					line='nemo_file=''\''+listcfg[i]+'\''"\n"
				if match2 != None :
					line='input_file=''\''+myfilename+'\''"\n"
				if match3 != None :
					line='input_lon=''\''+mylon+'\''"\n"
				if match4 != None :
					line='input_lat=''\''+mylat+'\''"\n"
				if match5 != None :
					line='output_file=''\''+wfile+'\''"\n"
			#	if match5 != None :
		##			line='output_name=''\''+myvar+'\''"\n"
				f2.write(line)
			f.close()
			f2.close()
		mycheck=subprocess.check_output('./scripgrid.exe namelist_new',shell=True	)
		mycheck=subprocess.check_output('./scrip.exe namelist_new',shell=True	)
		mycheck=subprocess.check_output('./scripshape.exe namelist_new',shell=True	)
		print('   Success ...')
		print('   => weight file is %s' % wfile)

	print()


				

