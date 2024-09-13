#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 10:40:59 2024

@author: ainsleyg
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.models import load_model
import os
import h5py

model_path = '/Users/ainsleyg/Documents/CISESS/unet_model_08_10_21.h5'
model = load_model(model_path)

X_test = np.load('/Users/ainsleyg/Documents/CISESS/X_08_10_21.npy', allow_pickle=True)
y_test = np.load('/Users/ainsleyg/Documents/CISESS/y_08_10_21.npy', allow_pickle=True)
X_test = X_test.astype(np.float32)
y_test = y_test.astype(np.float32)


y_pred = model.predict(X_test)

output_dir = '/Users/ainsleyg/Documents/CISESS/dataset_analysis'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def save_all_images(X, y_pred, output_dir):
    for i in range(len(X)):
        if X[i].shape[-1] == 3:
            input_img = (X[i] * 255).astype(np.uint8)
            input_img_path = os.path.join(output_dir, f'input_{i+1}.png')
            Image.fromarray(input_img).save(input_img_path)
        else:
            print(f"Skipping input image {i+1} due to incorrect shape: {X[i].shape}")

        if y_pred[i].ndim == 2 or (y_pred[i].ndim == 3 and y_pred[i].shape[-1] in [1, 3]):
            pred_img = y_pred[i]
            if y_pred[i].ndim == 3 and y_pred[i].shape[-1] == 1:
                pred_img = y_pred[i][:, :, 0]

            pred_img_colored = plt.cm.jet(pred_img / np.max(pred_img))
            pred_img_colored = (pred_img_colored[:, :, :3] * 255).astype(np.uint8)

            pred_img_path = os.path.join(output_dir, f'predicted_{i+1}.png')
            Image.fromarray(pred_img_colored).save(pred_img_path)
        else:
            print(f"Skipping predicted image {i+1} due to incorrect shape or channels: {y_pred[i].shape}")

save_all_images(X_test, y_pred, output_dir)

print("All images saved.")

def save_predictions_to_h5(y_pred, output_file):
    with h5py.File(output_file, 'w') as f:
        for i, pred in enumerate(y_pred):
            dataset_name = f'prediction_{i+1:03d}'
            f.create_dataset(dataset_name, data=pred)

output_file = '/Users/ainsleyg/Documents/CISESS/HDF5_files/predicted_data_08_10_21.h5'
save_predictions_to_h5(y_pred, output_file)


def plot_comparison(X, y_test, y_pred, n_samples=5):
    fig, axs = plt.subplots(n_samples, 2, figsize=(10, 2 * n_samples), sharex=True, sharey=True)

    for i in range(n_samples):
        axs[i, 0].imshow(X[i])
        axs[i, 0].axis('off')
        axs[i, 0].set_title(f'Input {i+1}')

        pred_img_colored = plt.cm.jet(y_pred[i].squeeze() / np.max(y_pred[i]))
        pred_img_colored = (pred_img_colored[:, :, :3] * 255).astype(np.uint8)
        axs[i, 1].imshow(pred_img_colored)
        axs[i, 1].axis('off')
        axs[i, 1].set_title(f'Predicted {i+1}')
        print(y_pred[i].max())

    plt.tight_layout()
    plt.show()

n_samples = 5

plot_indices = np.random.choice(range(len(X_test)), n_samples, replace=False)
X_plot = X_test[plot_indices]
y_true_plot = y_test[plot_indices]
y_pred_plot = y_pred[plot_indices]

plot_comparison(X_plot, y_true_plot, y_pred_plot, n_samples)
