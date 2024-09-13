#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 15:28:48 2024

@author: ainsleyg
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt

def reassemble_image_from_chunks(h5_file, coords_txt_file, chunk_height=160, chunk_width=128): #defines function and reassembles using txt file
    with h5py.File(h5_file, 'r') as file:
        with open(coords_txt_file, 'r') as coord_file:
            coordinates = coord_file.readlines()
        
        if not coordinates:
            raise ValueError("No coordinates found in the file.")
        
        coords_list = [line.strip().split(',') for line in coordinates] #coordinates = list of lists
        num_chunks = len(coords_list)
        
        x_coords = [int(coords[1]) for coords in coords_list] 
        y_coords = [int(coords[2]) for coords in coords_list]
        max_x = max(x_coords) + chunk_width
        max_y = max(y_coords) + chunk_height
        print(max_x)
        print(max_y)
        
        complete_image = np.zeros((max_y, max_x, 3), dtype=np.uint8)
        
        for coords in coords_list: #basically puts image back together in the correct position
            chunk_id, x_min, y_min, x_max, y_max = map(int, coords) #takes chunk coordinates
            dataset_name = f'chunk_{chunk_id}'
            chunk = file[dataset_name][:]
            chunk_shape = chunk.shape
            
            # Debugging statements
           # print(f"Processing chunk {chunk_id}: {chunk_shape}, coords: ({x_min},{y_min},{x_max},{y_max})")
            if chunk_shape != (chunk_height, chunk_width, 3):
                raise ValueError(f"Chunk {chunk_id} has unexpected shape {chunk_shape}")
            #print(y_min,y_max,x_min,x_max)
            complete_image[y_min:y_max, x_min:x_max, :] = chunk 
    
    return complete_image

h5_file = 'output/chunks.h5'
coords_txt_file = 'output/coordinates.txt'

reassembled_image = reassemble_image_from_chunks(h5_file, coords_txt_file) #executes function to reassemble image 

plt.imshow(reassembled_image) #plots image
plt.axis('off')
plt.show()
