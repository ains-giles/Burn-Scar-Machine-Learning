#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 16:42:34 2024

@author: ainsleyg
"""

import numpy as np
from PIL import Image
import os
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

original_image_path = '/Users/ainsleyg/Documents/CISESS/HDF5_files/chunks_08_10_21.h5'
coords_txt_file = '/Users/ainsleyg/Documents/CISESS/coord_files/coordinates_08_10_21.txt'
#rectangle_overlay = '/Users/ainsleyg/Documents/CISESS/coord_files/subsetted_burn_scar_coordinates.txt'
#desert_overlay = '/Users/ainsleyg/Documents/CISESS/coord_files/desert_training_coordinates.txt'
park_chunks_path = '/Users/ainsleyg/Documents/CISESS/HDF5_files/predicted_data_08_10_21.h5'
dataset_name = '08.10.2021' #MM.DD.YYYY
reassembled_image_path = '/Users/ainsleyg/Documents/CISESS/reassembled_image_08_10_21.png'

coords_df = pd.read_csv(coords_txt_file)
#overlay_df = pd.read_csv(rectangle_overlay, header=0, delimiter=', ', engine='python', skiprows=7)
#desert_overlay_df = pd.read_csv(desert_overlay, header=0, delimiter=', ', engine='python', skiprows=7)


max_row = coords_df['row2'].max()
max_col = coords_df['col2'].max()

original_image_array = np.zeros((max_row, max_col, 3))
reassembled_image = np.zeros((max_row, max_col))

with h5py.File(original_image_path, 'r') as h5_original, h5py.File(park_chunks_path, 'r') as h5_park_chunks:
    original_rectangle_keys = list(h5_original.keys())
    predicted_rectangle_keys = list(h5_park_chunks.keys())
    
    for rectangle, (index, row) in zip(original_rectangle_keys, coords_df.iterrows()):
        col1, row1, col2, row2 = int(row['col1']), int(row['row1']), int(row['col2']), int(row['row2'])
        original_image_array[row1:row2, col1:col2] = h5_original[rectangle][:]

    for rectangle,(index,row) in zip(predicted_rectangle_keys, coords_df.iterrows()):
        col1, row1, col2, row2 = int(row['col1']), int(row['row1']), int(row['col2']), int(row['row2'])
        reassembled_image[row1:row2, col1:col2] = h5_park_chunks[rectangle][:].reshape((160,128))


num_cols = 2
fig, axs = plt.subplots(ncols=num_cols, figsize=(15, 10), sharex=True, sharey=True)

axs[0].imshow(original_image_array)
im = axs[1].imshow(reassembled_image, cmap='jet', vmin=0, vmax=0.25)
axs[0].set_title(f'DCLF RGB - {dataset_name}')
axs[1].set_title(f'Model Prediction - {dataset_name}')
for a in axs.flat:
    a.axis('off')
    
# for index, row in overlay_df.iterrows():
#     col1, row1, col2, row2 = int(row['col1']), int(row['row1']), int(row['col2']), int(row['row2'])
#     rect = Rectangle((col1, row1), col2 - col1, row2 - row1, linewidth=1, edgecolor='r', facecolor='none')
#     axs[1].add_patch(rect)
    
cb_ax = fig.add_axes([0.93, 0.1, 0.02, 0.8])
cbar = fig.colorbar(im, cax=cb_ax)
cbar.set_ticks(np.arange(0, 0.3, 0.05))
cbar.ax.set_yticklabels([f'{i:.2f}' for i in np.arange(0, 0.3, 0.05)])
# img = plt.imshow(reassembled_image, cmap='jet')
# cbar = plt.colorbar(img, ticks=np.arange(0, 1.05, 0.05))
# cbar.ax.set_yticklabels([f'{i:.2f}' for i in np.arange(0, 1.05, 0.05)])
plt.show()

reassembled_image_uint8 = (reassembled_image * 255).astype(np.uint8)
image_to_save = Image.fromarray(reassembled_image_uint8)
image_to_save.save(reassembled_image_path)