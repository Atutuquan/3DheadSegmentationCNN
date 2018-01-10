#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 10:34:06 2018

@author: lukas
"""
wd = '/home/lukas/Documents/projects/headSegmentation/deepMedicKeras/'

###################   parameters // replace with config files ########################

dataset = 'BRATS15'

usingLoadedData = False
allBatchesAddress = '/home/lukas/Documents/projects/headSegmentation/deepMedicKeras/allBatches.pkl'
allLabelsAddress =  '/home/lukas/Documents/projects/headSegmentation/deepMedicKeras/allLabels.pkl'
allValBatchesAddress = '/home/lukas/Documents/projects/headSegmentation/deepMedicKeras/allValBatches.pkl'
allValLabelsAddress = '/home/lukas/Documents/projects/headSegmentation/deepMedicKeras/allValLabels.pkl'

# Parameters // this could be assigned in separate configuration files

# MODEL PARAMETERS
dpatch=51
L2 = 0.001
load_model = False
path_to_model = wd + '/Output/models/newModel_' +dataset + '.h5'
num_channels = 4
output_classes = 5
dropout = [0,0]  # dropout for last two fully connected layers
learning_rate = 0.001
optimizer_decay = 0

# TRAIN PARAMETERS
num_iter = 1
epochs = 1
n_patches = 200
n_subjects = 10 # Check that this is not larger than subjects in training file
size_minibatches = 20  # Check that this value is not larger than the ammount of patches per subject per class
n_patches_val = 50
n_subjects_val = 25 # Check that this is not larger than subjects in validation file
size_minibatches_val = 10  # Check that this value is not larger than the ammount of patches per subject per class
samplingMethod = 1


# TEST PARAMETERS
list_subjects_fullSegmentation = [0,1,2]
epochs_for_fullSegmentation = [1,3,6,9]
size_test_minibatches = 200
saveSegmentation = True


