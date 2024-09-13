#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 12:28:08 2024

@author: ainsleyg
"""

import h5py
import numpy as np
import pandas as pd

def load_hdf5_image(file_path, dataset_name):
    with h5py.File(file_path, 'r') as hdf:
        image_data = hdf[dataset_name][:]
        image_data = np.nan_to_num(image_data)
        #print("Raw image data shape:", image_data.shape)
        return image_data

def append_to_hdf5(output_file, data, length, width, dataset_name):
    with h5py.File(output_file, 'a') as hdf_out:
        hdf_out.create_dataset(f'{dataset_name}/data', data=data)
        hdf_out.create_dataset(f'{dataset_name}/length', data=length)
        hdf_out.create_dataset(f'{dataset_name}/width', data=width)

# Provide your paths
# image_path = '/Users/ainsleyg/Documents/CISESS/HDF5_files/park_fire_07_24_2024_daily_DLCF_RGB_composites_no_ref_QC_no_cldshadow_QC_keep_landwatermask_1_3_4_cldmsk_keep_probclear_clear_fix_bowtie_del_age_quality_flag.h5'
# dataset_name = '07.28.2024'  # input intended date in MM/DD/YYYY format

image_path = '/Users/ainsleyg/Documents/CISESS/HDF5_files/2021_daily_DLCF_RGB_composites_no_ref_QC_no_cldshadow_QC_keep_landwatermask_1_3_4_cldmsk_keep_probclear_clear_fix_bowtie_del_age_quality_flag.h5'
dataset_name = '08.10.2021'
burnscar_semi_labeled_dataset_file = '/Users/ainsleyg/Documents/CISESS/coord_files/coordinates_08_10_21.txt'
desert_training_coordinates_file = '/Users/ainsleyg/Documents/CISESS/coord_files/desert_training_coordinates.txt'


image_data = load_hdf5_image(image_path, dataset_name)

burnscar_semi_labeled_dataset_file = '/Users/ainsleyg/Documents/CISESS/coord_files/subsetted_burn_scar_coordinates.txt'
df_burnscar_semi_labeled = pd.read_csv(burnscar_semi_labeled_dataset_file, header=0, delimiter=', ', engine='python', skiprows=7) #og files
#df_burnscar_semi_labeled = pd.read_csv(burnscar_semi_labeled_dataset_file, header=0, delimiter=',', engine='python') #later fire files
#print(df_burnscar_semi_labeled.columns)
rectangles_burnscar = list(zip(df_burnscar_semi_labeled['col1'], df_burnscar_semi_labeled['row1'], df_burnscar_semi_labeled['col2'], df_burnscar_semi_labeled['row2']))
# col1 = df_burnscar_semi_labeled['col1'].tolist()
# col2 = df_burnscar_semi_labeled['col2'].tolist()
# row1 = df_burnscar_semi_labeled['row1'].tolist()
# row2 = df_burnscar_semi_labeled['row2'].tolist()

df_desert_training = pd.read_csv(desert_training_coordinates_file, header=None, delimiter=',', engine='python', skiprows=7)
rectangles_desert = list(zip(df_desert_training['col1'], df_desert_training['row1'], df_desert_training['col2'], df_desert_training['row2']))

band_indices = [0, 1, 2] #2.25, .86, .67 bands

rectangles = list(zip(col1, row1, col2, row2))
print("Rectangles:")

num_rectangles = len(rectangles)
band_matrix = np.zeros((num_rectangles, len(band_indices)), dtype=np.float32)

X = []
print(rectangles) 
for idx, (x1, y1, x2, y2) in enumerate(rectangles):
    if x1 < 0 or y1 < 0 or x2 > image_data.shape[1] or y2 > image_data.shape[0]:
        print(f"Skipping invalid rectangle {x1, y1, x2, y2}")
        continue
    cropped_burnscar = image_data[y1:y2, x1:x2, :]
    length = y2 - y1
    width = x2 - x1
    #print(f"Cropped size for rectangle {x1, y1, x2, y2}:", cropped_burnscar.shape)
    X.append((cropped_burnscar, length, width))
    
desert_output_file = '/Users/ainsleyg/Documents/CISESS/burn_composites/desert_training_composite.h5'

for idx, (x1, y1, x2, y2) in enumerate(rectangles_desert):
    if x1 < 0 or y1 < 0 or x2 > image_data.shape[1] or y2 > image_data.shape[0]:
        print(f"Skipping invalid desert rectangle {x1, y1, x2, y2}")
        continue
    cropped_desert = image_data[y1:y2, x1:x2, :]
    length = y2 - y1
    width = x2 - x1
    dataset_name = f'rectangle_{idx:03d}'
    append_to_hdf5(desert_output_file, cropped_desert, length, width, dataset_name)
    
output_file = '/Users/ainsleyg/Documents/CISESS/burn_composites/burn_scar_composite_08_10_21.h5'
with h5py.File(output_file, 'w') as hdf_out:
    for idx, (cropped_burnscar, length, width) in enumerate(X):
        dataset_name = 'rectangle_{:03d}'.format(idx)
        hdf_out.create_dataset(f'{dataset_name}/data', data=cropped_burnscar)
        hdf_out.create_dataset(f'{dataset_name}/length', data=length)
        hdf_out.create_dataset(f'{dataset_name}/width', data=width)

