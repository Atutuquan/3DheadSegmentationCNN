#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 13:55:34 2017

@author: lukas
"""
from __future__ import print_function
import psutil
import nibabel as nib
import numpy as np
import sys

fname =  '/home/lukas/Documents/projects/ATLASdataset/nativeDatasetAddress.txt'
channelsList = ['/home/lukas/Documents/projects/headSegmentation/deepMedicKeras/Data/BRATS15_examples/trainFlair_30.cfg','/home/lukas/Documents/projects/headSegmentation/deepMedicKeras/Data/BRATS15_examples/trainT2_30.cfg','/home/lukas/Documents/projects/headSegmentation/deepMedicKeras/Data/BRATS15_examples/trainT1c_30.cfg','/home/lukas/Documents/projects/headSegmentation/deepMedicKeras/Data/BRATS15_examples/trainT1_30.cfg']

fname = '/home/lukas/Documents/projects/headSegmentation/deepMedicKeras/Data/BRATS15_examples/trainFlair_30.cfg'

f = open(fname,"r")

allsubjects = f.readlines()

print(psutil.virtual_memory())

images = []
data = []

for i in xrange(0,len(allsubjects)):
    images.append(nib.load(str(allsubjects[i])[:-1]))    

for i in xrange(0,len(images)):
    data.append(np.array(images[i].get_data(),dtype=np.uint16))
    images[i].uncache()

size = 0
overheads=0
for i in xrange(0,len(data)):
    size = size + data[i].nbytes
    overheads = overheads + (sys.getsizeof(data[i])-data[i].nbytes)
    
sizeGB = size * 1e-9
ohGB = overheads * 1e-9
totalGB = sizeGB + ohGB
print(totalGB)
