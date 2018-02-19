#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 10:34:06 2018

@author: lukas
"""
wd = '/home/lukas/Documents/projects/brainSegmentation/deepMedicKeras/'

###################   parameters // replace with config files ########################


#availabledatasets :'ATLAS17', 'BRATS15', 'BRATS15_TEST', 'BRATS15_wholeNormalized' ,BRATS15_ENTIRE', 'Custom' (for explicitly giving channels)
dataset = 'BRATS15_ENTIRE'

############################## Load dataset #############################

if dataset == 'BRATS15':
    '''Example data BRATS 2015 (brain tissue normalized) - 4 input channels - 5 output classes'''
    
    trainChannels = ['/home/lukas/Documents/projects/brainSegmentation/deepMedicKeras/Data/BRATS15_splits/trainFlair_2018',
                    '/home/lukas/Documents/projects/brainSegmentation/deepMedicKeras/Data/BRATS15_splits/trainT1_2018',
                    '/home/lukas/Documents/projects/brainSegmentation/deepMedicKeras/Data/BRATS15_splits/trainT1c_2018',
                    '/home/lukas/Documents/projects/brainSegmentation/deepMedicKeras/Data/BRATS15_splits/trainT2_2018']
    trainLabels = '/home/lukas/Documents/projects/brainSegmentation/deepMedicKeras/Data/BRATS15_splits/trainGT_2018'
    '''
    validationChannels = ['/home/lukas/Documents/projects/deepmedic/examples/configFiles/deepMedicBRATS15/train/validation/splits/valFlair.cfg',
                          '/home/lukas/Documents/projects/deepmedic/examples/configFiles/deepMedicBRATS15/train/validation/splits/valT1.cfg',
                          '/home/lukas/Documents/projects/deepmedic/examples/configFiles/deepMedicBRATS15/train/validation/splits/valT1c.cfg',
                          '/home/lukas/Documents/projects/deepmedic/examples/configFiles/deepMedicBRATS15/train/validation/splits/valT2.cfg']
    validationLabels = '/home/lukas/Documents/projects/deepmedic/examples/configFiles/deepMedicBRATS15/train/validation/splits/valGT.cfg'
    '''
    testChannels = ['/home/lukas/Documents/projects/brainSegmentation/deepMedicKeras/Data/BRATS15_splits/testFlair_2018',
                    '/home/lukas/Documents/projects/brainSegmentation/deepMedicKeras/Data/BRATS15_splits/testT1_2018',
                    '/home/lukas/Documents/projects/brainSegmentation/deepMedicKeras/Data/BRATS15_splits/testT1c_2018',
                    '/home/lukas/Documents/projects/brainSegmentation/deepMedicKeras/Data/BRATS15_splits/testT2_2018']
    
    testLabels = '/home/lukas/Documents/projects/brainSegmentation/deepMedicKeras/Data/BRATS15_splits/testGT_2018'
    
    validationChannels = testChannels
    validationLabels = testLabels
    
    output_classes = 5 # Including background!!

elif dataset == 'BRATS15_wholeNormalized':
    '''Example data BRATS 2015 whole image normalization (including background)- 4 input channels - 5 output classes'''
    
    trainChannels = ['/home/lukas/Documents/projects/brainSegmentation/deepMedicKeras/Data/BRATS15_splits/whole_image_normalized/trainFlair_2018',
                    '/home/lukas/Documents/projects/brainSegmentation/deepMedicKeras/Data/BRATS15_splits/whole_image_normalized/trainT1_2018',
                    '/home/lukas/Documents/projects/brainSegmentation/deepMedicKeras/Data/BRATS15_splits/whole_image_normalized/trainT1c_2018',
                    '/home/lukas/Documents/projects/brainSegmentation/deepMedicKeras/Data/BRATS15_splits/whole_image_normalized/trainT2_2018']
    trainLabels = '/home/lukas/Documents/projects/brainSegmentation/deepMedicKeras/Data/BRATS15_splits/trainGT_2018'

    testChannels = ['/home/lukas/Documents/projects/brainSegmentation/deepMedicKeras/Data/BRATS15_splits/whole_image_normalized/testFlair_2018',
                    '/home/lukas/Documents/projects/brainSegmentation/deepMedicKeras/Data/BRATS15_splits/whole_image_normalized/testT1_2018',
                    '/home/lukas/Documents/projects/brainSegmentation/deepMedicKeras/Data/BRATS15_splits/whole_image_normalized/testT1c_2018',
                    '/home/lukas/Documents/projects/brainSegmentation/deepMedicKeras/Data/BRATS15_splits/whole_image_normalized/testT2_2018']
    
    testLabels = '/home/lukas/Documents/projects/brainSegmentation/deepMedicKeras/Data/BRATS15_splits/testGT_2018'
    
    validationChannels = testChannels
    validationLabels = testLabels
    
    output_classes = 5 # Including background!!



elif dataset == 'BRATS15_ENTIRE':
    '''Example data BRATS 2015 - 4 input channels - 5 output classes'''
    '''No test channels. Training on entire set of 274 heads. Whole image normalized'''
    
    trainChannels = ['/home/lukas/Documents/projects/public_SegmentationData/BRATS2015_Training/BRATS_fully_normalized_nifti/flairChannels',
                    '/home/lukas/Documents/projects/public_SegmentationData/BRATS2015_Training/BRATS_fully_normalized_nifti/t1Channels',
                    '/home/lukas/Documents/projects/public_SegmentationData/BRATS2015_Training/BRATS_fully_normalized_nifti/t1cChannels',
                    '/home/lukas/Documents/projects/public_SegmentationData/BRATS2015_Training/BRATS_fully_normalized_nifti/t2Channels']
    trainLabels = '/home/lukas/Documents/projects/public_SegmentationData/BRATS2015_Training/allGTlabels'

    validationChannels = trainChannels
    validationLabels = trainLabels
    
    testChannels = trainChannels
    testLabels = trainLabels
    
    output_classes = 5 # Including background!!

elif dataset == 'ATLAS17':
    '''Example data ATLAS 2017 - 1 input channel - 2 output classes'''
    
    trainChannels = ['/home/lukas/Documents/projects/deepmedic/examples/configFiles/deepMedicATLAS/train/splits1/ChannelsTrain_splits1.cfg']
    trainLabels = '/home/lukas/Documents/projects/deepmedic/examples/configFiles/deepMedicATLAS/train/splits1/SegmentsTrain_splits1.cfg'
    
    validationChannels = ['/home/lukas/Documents/projects/deepmedic/examples/configFiles/deepMedicATLAS/train/validation/splits1/ChannelsVal_splits1.cfg']
    validationLabels = '/home/lukas/Documents/projects/deepmedic/examples/configFiles/deepMedicATLAS/train/validation/splits1/SegmentsVal_splits1.cfg'

    testChannels = ['/home/lukas/Documents/projects/deepmedic/examples/configFiles/deepMedicATLAS/test/splits1/ChannelsTest_splits1.cfg']
    testLabels = '/home/lukas/Documents/projects/deepmedic/examples/configFiles/deepMedicATLAS/test/splits1/SegmentsTest_splits1.cfg'

    output_classes = 2 # Including background!!
    
    
elif dataset =='Custom':
        
    trainChannels = ['/home/lukas/Documents/projects/deepmedic/examples/configFiles/deepMedicBRATS15/train/splits/trainFlair_1subj.cfg',
                    '/home/lukas/Documents/projects/deepmedic/examples/configFiles/deepMedicBRATS15/train/splits/trainT1_1subj.cfg',
                    '/home/lukas/Documents/projects/deepmedic/examples/configFiles/deepMedicBRATS15/train/splits/trainT1c_1subj.cfg',
                    '/home/lukas/Documents/projects/deepmedic/examples/configFiles/deepMedicBRATS15/train/splits/trainT2_1subj.cfg']
    trainLabels = '/home/lukas/Documents/projects/deepmedic/examples/configFiles/deepMedicBRATS15/train/splits/trainGT_1subj.cfg'
    
    '''validationChannels = ['/home/lukas/Documents/projects/deepmedic/examples/configFiles/deepMedicBRATS15/train/validation/splits/valFlair.cfg',
                          '/home/lukas/Documents/projects/deepmedic/examples/configFiles/deepMedicBRATS15/train/validation/splits/valT1.cfg',
                          '/home/lukas/Documents/projects/deepmedic/examples/configFiles/deepMedicBRATS15/train/validation/splits/valT1c.cfg',
                          '/home/lukas/Documents/projects/deepmedic/examples/configFiles/deepMedicBRATS15/train/validation/splits/valT2.cfg']
    validationLabels = '/home/lukas/Documents/projects/deepmedic/examples/configFiles/deepMedicBRATS15/train/validation/splits/valGT.cfg'
    
    
    testChannels = ['/home/lukas/Documents/projects/deepmedic/examples/configFiles/deepMedicBRATS15/test/splits/testFlair.cfg',
                    '/home/lukas/Documents/projects/deepmedic/examples/configFiles/deepMedicBRATS15/test/splits/testT1.cfg',
                    '/home/lukas/Documents/projects/deepmedic/examples/configFiles/deepMedicBRATS15/test/splits/testT1c.cfg',
                    '/home/lukas/Documents/projects/deepmedic/examples/configFiles/deepMedicBRATS15/test/splits/testT2.cfg']
    testLabels = '/home/lukas/Documents/projects/deepmedic/examples/configFiles/deepMedicBRATS15/test/splits/testGT.cfg'
    '''
    
    validationChannels = trainChannels
    validationLabels = trainLabels
    
    testChannels = trainChannels
    testLabels = trainLabels
    output_classes = 5
#-------------------------------------------------------------------------------------------------------------

test_subjects = 54


# Parameters // this could be assigned in separate configuration files

######################################### MODEL PARAMETERS
usingAlternativeModel = False
dpatch=51
L2 = 0.0001
load_model = False
path_to_model = '/home/lukas/Documents/projects/brainSegmentation/deepMedicKeras/Output/models/TrainSessionBRATS15_WHOLE_DeepMedic2018-02-17_1742.h5'
logfile_model = 'TrainSessionBRATS15_WHOLE_DeepMedic2018-02-17_1742'
num_channels = 4
num_channels = len(trainChannels)
dropout = [0,0]  # dropout for last two fully connected layers
learning_rate = 0.0001
optimizer_decay = 0

########################################## TRAIN PARAMETERS
num_iter = 50
epochs = 10
samplingMethod_train = 1

n_patches = 5206
n_subjects = 274 # Check that this is not larger than subjects in training file
size_minibatches = 70 # Check that this value is not larger than the ammount of patches per subject per class

quickmode = True # Train without validation. Full segmentation often but only report dice score (whole)
n_patches_val = 500
n_subjects_val = 50 # Check that this is not larger than subjects in validation file
size_minibatches_val = 500 # Check that this value is not larger than the ammount of patches per subject per class
samplingMethod_val = 0

########################################### TEST PARAMETERS
n_fullSegmentations = 10
#list_subjects_fullSegmentation = [0,1,2,3,4,12,13,14,15]
epochs_for_fullSegmentation = [0,1,2,3,4,5,6,7,8,9]
size_test_minibatches = 200
saveSegmentation = True


