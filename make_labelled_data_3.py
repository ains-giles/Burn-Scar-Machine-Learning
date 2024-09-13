#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 15:54:24 2024

@author: ainsleyg
"""
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import os
import numpy as np
import h5py
import pandas as pd

# Set up backend for plotting
#plt.switch_backend('qtAgg')

# Define paths
home_dir = '/Users/ainsleyg/Documents/python/CISSES/'
database_name = '2020_daily_DLCF_RGB_composites_no_ref_QC_no_cldshadow_QC_keep_landwatermask_1_3_4_cldmsk_keep_probclear_clear_fix_bowtie_del_age_quality_flag.h5'
file_path = os.path.join(home_dir, database_name)

def load_and_display_dataset(file_path, date):
    with h5py.File(file_path, 'r') as file:
        dataset = file[date][:]
        print(f"Dataset shape: {dataset.shape}")
        print(f"Dataset dtype: {dataset.dtype}")
        
        # Display the image
        plt.imshow(dataset)  # Assuming the values are in a range that can be cast to uint8 #dont do this, should cast to 0
        plt.title(f"Image for {date}")
        plt.axis('off')
        plt.show()

# Example date to load
example_date = "08.10.2020"
load_and_display_dataset(file_path, example_date)
# Load HDF5 file
try:
    with h5py.File(file_path, 'r') as hf_database:
        X = hf_database[example_date][:]  # pick any day in the database

    rgb_OG = np.copy(X)
    rgb_OG_copy = np.copy(X)

    # Ancillary composites to view
    Rbrn, Rveg, Rvis = rgb_OG_copy[:, :, 0], rgb_OG_copy[:, :, 1], rgb_OG_copy[:, :, 2]

except OSError as e:
    print(f"Error opening file: {e}")
    raise
except Exception as e:
    print(f"An error occurred: {e}")
    raise
    
burnscar_semi_labeled_dataset_file = 'subsetted_burn_scar_coordinates.txt'
df_burnscar_semi_labeled = pd.read_csv(burnscar_semi_labeled_dataset_file,\
                                         header=0, delimiter=', ', skiprows=7)
# print(df_burnscar_semi_labeled)

#build boxes around burn scars then visualize on RGB
col1 = df_burnscar_semi_labeled['col1'].tolist()
col2 = df_burnscar_semi_labeled['col2'].tolist()
row1 = df_burnscar_semi_labeled['row1'].tolist()
row2 = df_burnscar_semi_labeled['row2'].tolist()

# Example extraction of burn scar data from HDF5 file
# Replace 'burn_scar_data' with the correct dataset name in the HDF5 file
#try:
    #with h5py.File(file_path, 'r') as hf_database:
     #   col1 = hf_database['col1'][:]
     #   col2 = hf_database['col2'][:]
     #   row1 = hf_database['row1'][:]
     #   row2 = hf_database['row2'][:]
#except KeyError as e:
    #print(f"Dataset not found in HDF5 file: {e}")
   # raise
#except Exception as e:
   # print(f"An error occurred: {e}")
   # raise

# Plot RGB image and burn scars
plt.rcParams.update({'font.size': 12})
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(35, 20))

ax.imshow(1.5 * rgb_OG)
area_total = 0

for i in range(len(col1)):
    width, length = col2[i] - col1[i], row2[i] - row1[i]
    area_total += width * length
    rect = patches.Rectangle((col1[i], row1[i]), width, length, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

ax.set_title('Day Land Cloud Fire RGB\n[2.25, 0.86, 0.67]µm\nManual Labels (Red Squares)')
fig.suptitle('NOAA-20 VIIRS Valid 08.10.2020 Composited & Cloud-Cleared Over Previous 8 Days')

ax.set_xticks([])
ax.set_yticks([])

plt.tight_layout()
plt.show()
