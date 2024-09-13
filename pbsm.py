#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 14:44:35 2024

@author: ainsleyg
"""
import numpy as np

def get_normalized_differenced_vegetation_index(R_M1, R_M7):
    return (R_M7 - R_M1)/(R_M7 + R_M1)
def get_normalized_burn_ratio(R_M7, R_M11):
    return (R_M11-R_M7)/(R_M11+R_M7)

def get_burn_scar_composite(R_M7, R_M11, R_M1=None, geotiff=False, landwater_mask=None):
    from scipy import ndimage
    if not geotiff:
        # R_M11[R_M7  > 0.12] = np.nan #for clear no smoke only
        # R_M11[R_M7 < 0.0281] = np.nan
        # R_M11[R_M11 < 0.01] = np.nan
        # R_M11[R_M7  > 0.1346] = np.nan #for clear no smoke only
        # R_M11[R_M11 < 0.0281]
        # R_M11 = ndimage.gaussian_filter(R_M11, sigma=1)
        # R_M7 = ndimage.gaussian_filter(R_M7, sigma=1)
        # R_M11 = ndimage.gaussian_filter(R_M11, sigma=2)
        R_M11_forward_VZA  = np.copy(R_M11)
        R_M11_backward_VZA = np.copy(R_M11)
        NDVI = get_normalized_differenced_vegetation_index(R_M1, R_M7)
        # brightness_thresh = 0.3
        # R_M11_forward_VZA[R_M11_forward_VZA < brightness_thresh] = np.nan
        # R_M11_backward_VZA[R_M11_backward_VZA > brightness_thresh] = np.nan
        # # forward
        # R_M11_forward_VZA[R_M7  > 0.2]   = np.nan #for clear no smoke only
        # R_M11_forward_VZA[R_M7  < 0.0281] = np.nan
        # R_M11_forward_VZA[R_M11_forward_VZA < 0.05]   = np.nan
        # # backward
        # brightness_factor = 1.5
        # thresh_1 = brightness_factor * 0.2
        # thresh_2 = brightness_factor * 0.0281
        # thresh_3 = brightness_factor * 0.05
        # R_M11_backward_VZA[R_M7  > thresh_1] = np.nan #for clear no smoke only
        # R_M11_backward_VZA[R_M7  < thresh_2] = np.nan
        # R_M11_backward_VZA[R_M11_backward_VZA < thresh_3] = np.nan
        NBR_thresh  = -0.1
        NBR_forward = get_normalized_burn_ratio(R_M7, R_M11_forward_VZA)
        NBR_forward[NBR_forward < NBR_thresh] = np.nan
        NBR_backward = get_normalized_burn_ratio(R_M7, R_M11_backward_VZA)
        NBR_backward[NBR_backward < NBR_thresh] = np.nan
        burn_scar_mask = NBR_forward
        burn_scar_mask_nan_idx = np.where(np.isnan(burn_scar_mask)==True)
        burn_scar_mask[burn_scar_mask_nan_idx] = NBR_backward[burn_scar_mask_nan_idx]
        # burn_scar_mask[burn_scar_mask<0.07] = np.nan
        if R_M1 is not None:
            # bright_thresh = 0.2 # 9/30/20
            # bright_thresh = 0.15 # 8/31/20
            burn_scar_mask[(R_M1 > 0.2) & (burn_scar_mask<0.1)] = np.nan
        burn_scar_mask[NDVI>0.35] = np.nan
        if landwater_mask is None:
            return burn_scar_mask
        else:#need to add some flags from cloud mask
            burn_scar_mask[landwater_mask==desert] = 0
            return burn_scar_mask
    else:
        R_M11[R_M7  > 55] = 0
        R_M11[R_M11 < 45] = 0
        return ndimage.gaussian_filter(R_M11, sigma=1.1)