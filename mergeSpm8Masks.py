#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 12:20:05 2017

@author: lukas
"""
import nibabel as nib
import numpy as np

path = '/home/lukas/Documents/projects/ATLASdataset/native_part1/c0005/c0005s0044t01/'
name = path[-14:-1] 
suffixes = ['_mask_csf.nii', '_mask_gray.nii', '_mask_white.nii']

for i in range(0,len(suffixes)):
    
    f = nib.load(path+name+suffixes[i])
    data = np.array(f.get_data(), dtype=np.uint8)
    vdata = np.reshape(data, (1,data.shape[0]*data.shape[1]*data.shape[2]))    
    if i == 0:
        mask = vdata   
    else:
        indx = np.argwhere(vdata[0,:] > 0 )
        mask[0,indx] = 1
    f.uncache()
    
mask = np.reshape(mask, data.shape)
mask = nib.Nifti1Image(mask, affine=f.affine, header=f.header)
nib.save(mask, path+name+'_brainMask.nii.gz')

#mask.to_filename(path+name+'_brainMask.nii.gz')
