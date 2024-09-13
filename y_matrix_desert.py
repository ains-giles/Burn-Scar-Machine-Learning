#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 12:50:28 2024

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
    dataset_name = rectangle_keys[idx]
    if dataset_name.startswith('desert_rectangle_'):
        pbsm = np.zeros_like(burnscar[:, :, 0])
    else:
        pbsm = get_burn_scar_composite(burnscar[:,:,1], burnscar[:,:,0], burnscar[:,:,2])
        pbsm[~np.isnan(pbsm)] = 1
        pbsm[np.isnan(pbsm)] = 0
    
    Y[idx] = pbsm

output_file = '/Users/ainsleyg/Documents/CISESS/burn_composites/boolean_burn_composite_training.h5'
with h5py.File(output_file, 'w') as hdf_out:
    for idx, pbsm in enumerate(Y):
        dataset_name = rectangle_keys[idx]
        hdf_out.create_dataset(f'{dataset_name}/data', data=pbsm)
        
