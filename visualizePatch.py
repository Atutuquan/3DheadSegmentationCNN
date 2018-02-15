#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 14:15:46 2017

@author: lukas
"""
import numpy as np
import scipy.misc as smp
import nibabel as nib
#from helpers import extractPatches

#filename = './Data/trainChannels20_t1c.cfg'
#labelsFileName = './Data/trainGtLabels20.cfg'
#batch, labels = extractPatches(filename,labelsFileName, n_patches = 1, n_subjects = 1, dpatch = 190, output_classes=2)

# Dimensions:  batch (number patches, axial , sagittal, coronal, channels)
image = patches[2550,10,0:50,0:50,1]
img = smp.toimage( image )       # Create a PIL image
img.show()                      # View in default viewer
img


groundTruthChannel = '/home/lukas/Documents/projects/public_SegmentationData/BRATS2015_Training/HGG/brats_2013_pat0001_1/VSD.Brain_3more.XX.O.OT.54517/VSD.Brain_3more.XX.O.OT.54517.nii.gz'
img = nib.load(groundTruthChannel)
data = np.array(img.get_data(),dtype=np.uint8)
img.uncache()    
maxv = np.amax(data)
np.argwhere(data == maxv)
image = data[:,111,:] * (255/maxv)
np.amax(image)
img = smp.toimage( image )       # Create a PIL image
img.show()                      # View in default viewer
img


int8 = np.array(data[0], dtype = np.uint8)
image = int8[:,100,:] 
img = smp.toimage( image )       # Create a PIL image
img.show()   
img

int16 = np.array(data[0], dtype = np.uint16)
