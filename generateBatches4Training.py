#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 14:14:50 2018

@author: lukas
"""

import os
os.chdir('/home/lukas/Documents/projects/headSegmentation/deepMedicKeras/')
import time

from helpers import sampleTrainData, getForegroundBackgroundVoxels, file_len, getSubjectChannels, extractImagePatch, extractLabels, to_categorical
from random import shuffle
import numpy as np

###################   parameters // replace with config files ########################

'''Example data BRATS 2015 - 4 input channels - 5 output classes'''
#channelsList = ['/home/lukas/Documents/projects/headSegmentation/deepMedicKeras/Data/BRATS15_examples/trainFlair_30.cfg','/home/lukas/Documents/projects/headSegmentation/deepMedicKeras/Data/BRATS15_examples/trainT2_30.cfg','/home/lukas/Documents/projects/headSegmentation/deepMedicKeras/Data/BRATS15_examples/trainT1c_30.cfg','/home/lukas/Documents/projects/headSegmentation/deepMedicKeras/Data/BRATS15_examples/trainT1_30.cfg']
#groundTruthChannel_list = '/home/lukas/Documents/projects/headSegmentation/deepMedicKeras/Data/BRATS15_examples/trainGT_30.cfg'
#output_classes = 5 # Including background!!

'''Example data ATLAS 2017 - 1 input channel - 2 output classes'''
channelsList = ['/home/lukas/Documents/projects/ATLASdataset/nativeDatasetAddress.txt']
groundTruthChannel_list = '/home/lukas/Documents/projects/ATLASdataset/nativeDatasetSegmentsAddress.txt'
output_classes = 2 # Including background!!

#channelsList = ['/home/lukas/Documents/projects/headSegmentation/deepMedicKeras/Data/ATLAS_dataset_examples/trainChannels20_t1c.cfg']
#groundTruthChannel_list = '/home/lukas/Documents/projects/headSegmentation/deepMedicKeras/Data/ATLAS_dataset_examples/trainGtLabels20.cfg'
# Parameters // this could be assigned in separate configuration files

num_iter=50
n_patches = 1000
n_subjects = 100
num_channels = len(channelsList)
samplingMethod = 1

#model Parameters
dpatch=51

t1 = time.time()

allBatches = []
allLabels = []

allValBatches = []
allValLabels = []

sampling = "random"

if (sampling == "random"):
    for i in range(0, num_iter):
    
        #batch, labels = extractPatches(filename,labelsFileName, n_patches, n_subjects, dpatch, output_classes)
        #valbatch, vallabels = extractPatches(valfilename, vallabelsfilename, n_patches, n_subjects, dpatch, output_classes)
        
        "Dense Training"
        "Output of one 25x25x25 input is 9x9x9, and all of these voxels get classified (they are all central voxels of the 17x17x17 recpetive fields that fit the input segment)." 
        "Need labels of size 9x9x9 and a loss function that works over all classified voxels in the batch"
        print("Extracting image patches for training")
        batch, labels = sampleTrainData(channelsList, groundTruthChannel_list, n_patches, n_subjects, dpatch, output_classes, samplingMethod)
        print("Extracting image patches for Validation")
        valbatch, vallabels = sampleTrainData(channelsList, groundTruthChannel_list, n_patches, n_subjects, dpatch, output_classes, samplingMethod)
        print(str(i) + '/' + str(num_iter))
        allBatches.append(batch)
        allLabels.append(labels)
        allValBatches.append(valbatch)
        allValLabels.append(vallabels)
    
    print('Total data collection took seconds:')
    print(round(time.time()-t1,2))
    print("Collected " + str(len(allBatches)*n_patches) +" image patches")

    import pickle
    with open('/home/lukas/Documents/projects/headSegmentation/deepMedicKeras/Batches.pkl','wb') as fp:
        pickle.dump(allBatches,fp)
    with open('/home/lukas/Documents/projects/headSegmentation/deepMedicKeras/Labels.pkl','wb') as fp:
        pickle.dump(allLabels,fp)
    with open('/home/lukas/Documents/projects/headSegmentation/deepMedicKeras/ValBatches.pkl','wb') as fp:
        pickle.dump(allValBatches,fp)
    with open('/home/lukas/Documents/projects/headSegmentation/deepMedicKeras/ValLabels.pkl','wb') as fp:
        pickle.dump(allValLabels,fp)
    

elif(sampling == "foreground_exhaustive"):
    
    
    channelsList = ['/home/lukas/Documents/projects/ATLASdataset/nativeDatasetAddress.txt']
    groundTruthChannel_list = '/home/lukas/Documents/projects/ATLASdataset/nativeDatasetSegmentsAddress.txt'
    
    labelsFile = open(groundTruthChannel_list,"r")    
    total_subjects = file_len(groundTruthChannel_list)
    labelsFile.close()    
    total_subjects = total_subjects
    subjectIndexes = range(0,total_subjects)
    channels = getSubjectChannels(subjectIndexes, groundTruthChannel_list)
    allForegrounds = []
    n_patches = 0
    
    for index in subjectIndexes:
        
        fg = getForegroundBackgroundVoxels(channels[index], dpatch)
        allForegrounds.append(fg)

    

#%%
import pickle
with open('/home/lukas/Documents/projects/headSegmentation/deepMedicKeras/allBatches.pkl','wb') as fp:
    pickle.dump(allBatches,fp)
with open('/home/lukas/Documents/projects/headSegmentation/deepMedicKeras/allLabels.pkl','wb') as fp:
    pickle.dump(allLabels,fp)
with open('/home/lukas/Documents/projects/headSegmentation/deepMedicKeras/allValBatches.pkl','wb') as fp:
    pickle.dump(allValBatches,fp)
with open('/home/lukas/Documents/projects/headSegmentation/deepMedicKeras/allValLabels.pkl','wb') as fp:
    pickle.dump(allValLabels,fp)

#%%
import pickle
with open('allBatches.pkl','rb') as fp:
        allBatches = pickle.load(fp)
with open('allLabels.pkl','rb') as fp:
        allLabels = pickle.load(fp)
with open('allValBatches.pkl','rb') as fp:
        allValBatches = pickle.load(fp)
with open('allValLabels.pkl','rb') as fp:
        allValLabels = pickle.load(fp)


"Work on: Make exhaustive subject list for iteration"
"Make exhaustive list for foreground voxels."
"--> Generate lists for all foreground voxels for all subjects. So if in total 300 subjects, make 300 lists of all foreground voxels, and make a partition of it. So at the end we have just a random walkthrough through all foreground voxels of all subjects"
"How to deal with background? "

