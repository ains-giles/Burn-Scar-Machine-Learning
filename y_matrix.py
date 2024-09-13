#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 15:12:53 2024

@author: ainsleyg
"""
import h5py
import numpy as np
import os
from pbsm import get_burn_scar_composite 

pbsm_file_path = '/Users/ainsleyg/Documents/CISESS/python scripts/pbsm.py'
composite_file_path = '/Users/ainsleyg/Documents/CISESS/burn_composites/combined_rectangles_08_10_21.h5'

Y = []

with h5py.File(composite_file_path, 'r') as hf_composite: 
    rectangle_keys = list(hf_composite.keys())
    
    for rectangle in rectangle_keys:
        Y.append(hf_composite[rectangle+"/data"][:])
        
for idx, burnscar in enumerate(Y):
    pbsm = get_burn_scar_composite(burnscar[:,:,1], burnscar[:,:,0], burnscar[:,:,2])
    pbsm[~np.isnan(pbsm)] = 1
    pbsm[np.isnan(pbsm)] = 0
    Y[idx] = pbsm
    #print (Y[idx])
    #print (len(Y[idx][Y[idx]==0]) / (len(Y[idx].flat)))

#Y_array= np.array(Y)
#np.save('/Users/ainsleyg/Documents/CISESS/y_park.npy', Y_array)

output_file = '/Users/ainsleyg/Documents/CISESS/burn_composites/boolean_burn_composite_training.h5'
with h5py.File(output_file, 'w') as hdf_out:
    for idx, pbsm in enumerate(Y):
        dataset_name = rectangle_keys[idx]
        hdf_out.create_dataset(f'{dataset_name}/data', data=pbsm)
        #hdf_out.create_dataset(f'{dataset_name}/length', data=pbsm.shape[0])
        #hdf_out.create_dataset(f'{dataset_name}/width', data=pbsm.shape[1])
