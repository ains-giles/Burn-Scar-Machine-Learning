#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 11:46:47 2024

@author: ainsleyg
"""
import h5py
import numpy as np
import pandas as pd

def load_hdf5_image(file_path, dataset_name):
    with h5py.File(file_path, 'r') as hdf:
        image_data = hdf[dataset_name][:]
        image_data = np.nan_to_num(image_data)
        return image_data

def append_to_hdf5(output_file, data, length, width, dataset_name):
    with h5py.File(output_file, 'a') as hdf_out:
        hdf_out.create_dataset(f'{dataset_name}/data', data=data)
        hdf_out.create_dataset(f'{dataset_name}/length', data=length)
        hdf_out.create_dataset(f'{dataset_name}/width', data=width)

# Paths to your files
image_path = '/Users/ainsleyg/Documents/CISESS/HDF5_files/2021_daily_DLCF_RGB_composites_no_ref_QC_no_cldshadow_QC_keep_landwatermask_1_3_4_cldmsk_keep_probclear_clear_fix_bowtie_del_age_quality_flag.h5'
dataset_name = '08.10.2021'
burnscar_semi_labeled_dataset_file = '/Users/ainsleyg/Documents/CISESS/coord_files/subsetted_burn_scar_coordinates.txt'
desert_training_coordinates_file = '/Users/ainsleyg/Documents/CISESS/coord_files/desert_training_coordinates.txt'

# Load image data
image_data = load_hdf5_image(image_path, dataset_name)

# Read and process burn scar coordinates
df_burnscar_semi_labeled = pd.read_csv(burnscar_semi_labeled_dataset_file, header=None, delimiter=',', skiprows=7, engine='python')

# Print the first few rows and columns of the DataFrame to understand its structure
print(df_burnscar_semi_labeled.head())
print("Columns in burn scar DataFrame:", df_burnscar_semi_labeled.columns)

# Rename columns correctly, including 'description'
df_burnscar_semi_labeled.columns = ['col1', 'row1', 'col2', 'row2', 'description']

# Extract only the first four columns for rectangles
df_burnscar_semi_labeled = df_burnscar_semi_labeled[['col1', 'row1', 'col2', 'row2']]

# Convert columns to integers and handle any missing data
df_burnscar_semi_labeled = df_burnscar_semi_labeled.apply(pd.to_numeric, errors='coerce').dropna().astype(int)

rectangles_burnscar = list(zip(df_burnscar_semi_labeled['col1'], 
                               df_burnscar_semi_labeled['row1'], 
                               df_burnscar_semi_labeled['col2'], 
                               df_burnscar_semi_labeled['row2']))

try:
    df_desert_training = pd.read_csv(desert_training_coordinates_file, header=None, delimiter=',', engine='python', names=['col1', 'row1', 'col2', 'row2', 'description'], skiprows=1)
    df_desert_training = df_desert_training[['col1', 'row1', 'col2', 'row2']]
    df_desert_training = df_desert_training.apply(pd.to_numeric, errors='coerce').dropna()
    df_desert_training = df_desert_training.astype(int)
    rectangles_desert = list(zip(df_desert_training['col1'], df_desert_training['row1'], df_desert_training['col2'], df_desert_training['row2']))
except pd.errors.ParserError as e:
    print(f"Error reading the desert training CSV file: {e}")
    rectangles_desert = []
except ValueError as e:
    print(f"Value error during conversion: {e}")
    rectangles_desert = []

def process_and_save_rectangles(burnscar_rectangles, desert_rectangles, output_file):
    with h5py.File(output_file, 'w') as hdf_out:
        # Process burn scar rectangles
        for idx, (x1, y1, x2, y2) in enumerate(burnscar_rectangles):
            if x1 < 0 or y1 < 0 or x2 > image_data.shape[1] or y2 > image_data.shape[0]:
                print(f"Skipping invalid burn scar rectangle {x1, y1, x2, y2}")
                continue
            cropped = image_data[y1:y2, x1:x2, :]
            length = y2 - y1
            width = x2 - x1
            dataset_name = f'rectangle_{idx:03d}'
            hdf_out.create_dataset(f'{dataset_name}/data', data=cropped)
            hdf_out.create_dataset(f'{dataset_name}/length', data=length)
            hdf_out.create_dataset(f'{dataset_name}/width', data=width)

        # Process desert rectangles
        for idx, (x1, y1, x2, y2) in enumerate(desert_rectangles):
            if x1 < 0 or y1 < 0 or x2 > image_data.shape[1] or y2 > image_data.shape[0]:
                print(f"Skipping invalid desert rectangle {x1, y1, x2, y2}")
                continue
            cropped = image_data[y1:y2, x1:x2, :]
            length = y2 - y1
            width = x2 - x1
            dataset_name = f'desert_rectangle_{idx:03d}'
            hdf_out.create_dataset(f'{dataset_name}/data', data=cropped)
            hdf_out.create_dataset(f'{dataset_name}/length', data=length)
            hdf_out.create_dataset(f'{dataset_name}/width', data=width)

combined_output_file = '/Users/ainsleyg/Documents/CISESS/burn_composites/combined_rectangles_08_10_21.h5'
process_and_save_rectangles(rectangles_burnscar, rectangles_desert, combined_output_file)
