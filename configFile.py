#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 10:34:06 2018

@author: lukas
"""
wd = '/home/lukas/Documents/projects/brainSegmentation/deepMedicKeras/'

###################   parameters // replace with config files ########################

#availabledatasets :'ATLAS17', 'BRATS15', 'Custom' (for explicitly giving channels)
dataset = 'BRATS15'

trainChannels = ['/home/lukas/Documents/projects/deepmedic/examples/configFiles/deepMedicBRATS15/train/splits/trainFlair.cfg',
                '/home/lukas/Documents/projects/deepmedic/examples/configFiles/deepMedicBRATS15/train/splits/trainT1.cfg',
                '/home/lukas/Documents/projects/deepmedic/examples/configFiles/deepMedicBRATS15/train/splits/trainT1c.cfg',
                '/home/lukas/Documents/projects/deepmedic/examples/configFiles/deepMedicBRATS15/train/splits/trainT2.cfg']
trainLabels = '/home/lukas/Documents/projects/deepmedic/examples/configFiles/deepMedicBRATS15/train/splits/trainGT.cfg'

validationChannels = ['/home/lukas/Documents/projects/deepmedic/examples/configFiles/deepMedicBRATS15/train/validation/splits/valFlair.cfg',
                      '/home/lukas/Documents/projects/deepmedic/examples/configFiles/deepMedicBRATS15/train/validation/splits/valT1.cfg',
                      '/home/lukas/Documents/projects/deepmedic/examples/configFiles/deepMedicBRATS15/train/validation/splits/valT1c.cfg',
                      '/home/lukas/Documents/projects/deepmedic/examples/configFiles/deepMedicBRATS15/train/validation/splits/valT2.cfg']
validationLabels = '/home/lukas/Documents/projects/deepmedic/examples/configFiles/deepMedicBRATS15/train/validation/splits/valGT.cfg'

testChannels = ['/home/lukas/Documents/projects/deepmedic/examples/configFiles/deepMedicBRATS15/test/splits/testFlair.cfg',
                '/home/lukas/Documents/projects/deepmedic/examples/configFiles/deepMedicBRATS15/test/splits/testT1.cfg',
                '/home/lukas/Documents/projects/deepmedic/examples/configFiles/deepMedicBRATS15/test/splits/testT1c.cfg',
                '/home/lukas/Documents/projects/deepmedic/examples/configFiles/deepMedicBRATS15/test/splits/testT2.cfg']
testLabels = '/home/lukas/Documents/projects/deepmedic/examples/configFiles/deepMedicBRATS15/test/splits/testGT.cfg'
output_classes = 5 # Including background!!


usingLoadedData = False
allBatchesAddress = '/home/lukas/Documents/projects/brainSegmentation/deepMedicKeras/allBatches.pkl'
allLabelsAddress =  '/home/lukas/Documents/projects/brainSegmentation/deepMedicKeras/allLabels.pkl'
allValBatchesAddress = '/home/lukas/Documents/projects/brainSegmentation/deepMedicKeras/allValBatches.pkl'
allValLabelsAddress = '/home/lukas/Documents/projects/brainSegmentation/deepMedicKeras/allValLabels.pkl'

# Parameters // this could be assigned in separate configuration files

# MODEL PARAMETERS
dpatch=51
L2 = 0.0
load_model = False
path_to_model = wd + '/Output/models/newModel_' +dataset + '.h5'
num_channels = 4
output_classes = 5
dropout = [0,0]  # dropout for last two fully connected layers
learning_rate = 0.001
optimizer_decay = 0

# TRAIN PARAMETERS
num_iter = 30
epochs = 10
n_patches = 1000
n_subjects = 25 # Check that this is not larger than subjects in training file
size_minibatches = 100  # Check that this value is not larger than the ammount of patches per subject per class

n_patches_val = 50
n_subjects_val = 10 # Check that this is not larger than subjects in validation file
size_minibatches_val = 50  # Check that this value is not larger than the ammount of patches per subject per class
samplingMethod = 1


# TEST PARAMETERS
list_subjects_fullSegmentation = [0,5]
epochs_for_fullSegmentation = [5,9]
size_test_minibatches = 200
saveSegmentation = True


