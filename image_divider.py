#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 15:21:34 2024

@author: ainsleyg
"""

import h5py
import numpy as np
import os
from PIL import Image

def extract_and_save_image_chunks(h5_file, dataset_name, output_h5_file, coords_txt_file, chunk_height=160, chunk_width=128, output_png_dir= 'output/park_pngs'): #defines and saves function to create chunks
    if not os.path.exists(os.path.dirname(output_h5_file)):
        os.makedirs(os.path.dirname(output_h5_file)) #makes output dir
    if not os.path.exists(output_png_dir):
        os.makedirs(output_png_dir)
        
    with h5py.File(h5_file, 'r') as file:
        image = file[dataset_name][:]
        
        img_height, img_width, channels = image.shape
        
        if channels != 3:
            raise ValueError("The image does not have 3 channels.")
        
        chunk_id = 0
        coordinates = [] #creates list to store data
        with h5py.File(output_h5_file, 'w') as out_file: #chunk extraction and append to txt
            for y in range(0, img_height, chunk_height):
                for x in range(0, img_width, chunk_width):
                    chunk = image[y:y+chunk_height, x:x+chunk_width, :]
                    
                    if chunk.shape[0] == chunk_height and chunk.shape[1] == chunk_width:
                        dataset_name = f'chunk_{chunk_id:03d}'
                        out_file.create_dataset(dataset_name, data=chunk)
                        description = f'description_{chunk_id}'
                        coordinates.append(f"{chunk_id},{x},{y},{x+chunk_width},{y+chunk_height}")
                        
                        chunk_uint8 = (chunk * 255).astype(np.uint8)
                        chunk_image = Image.fromarray(chunk_uint8)
                        chunk_image.save(os.path.join(output_png_dir, f'{dataset_name}.png'))
                        chunk_id += 1

        with open(coords_txt_file, 'w') as coord_file:
            coord_file.write("description,col1,row1,col2,row2\n")
            coord_file.write("\n".join(coordinates))

# h5_file = '/Users/ainsleyg/Documents/CISESS/HDF5_files/park_fire_07_24_2024_daily_DLCF_RGB_composites_no_ref_QC_no_cldshadow_QC_keep_landwatermask_1_3_4_cldmsk_keep_probclear_clear_fix_bowtie_del_age_quality_flag.h5'
# dataset_name = '07.28.2024'
# output_h5_file = '/Users/ainsleyg/Documents/CISESS/HDF5_files/park_chunks_07_28.h5'
# coords_txt_file = '/Users/ainsleyg/Documents/CISESS/coord_files/park_coordinates_07_28.txt'

h5_file = '/Users/ainsleyg/Documents/CISESS/HDF5_files/2021_daily_DLCF_RGB_composites_no_ref_QC_no_cldshadow_QC_keep_landwatermask_1_3_4_cldmsk_keep_probclear_clear_fix_bowtie_del_age_quality_flag.h5'
dataset_name = '08.10.2021'
output_h5_file = '/Users/ainsleyg/Documents/CISESS/HDF5_files/chunks_08_10_21.h5'
coords_txt_file = '/Users/ainsleyg/Documents/CISESS/coord_files/coordinates_08_10_21.txt'

extract_and_save_image_chunks(h5_file, dataset_name, output_h5_file, coords_txt_file) #executes function
