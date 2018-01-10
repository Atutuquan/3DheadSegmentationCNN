#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 13:58:48 2017

@author: lukas
"""
import os
import matplotlib.pyplot as plt
import time
from model import DeepMedic
from helpers import sampleTrainData, fullHeadSegmentation
from random import shuffle


os.chdir('/home/lukas/Documents/projects/headSegmentation/deepMedicKeras/')

###################   parameters // replace with config files ########################


dataset = 'ATLAS17'


if dataset == 'BRATS15':
    '''Example data BRATS 2015 - 4 input channels - 5 output classes'''
    
    channelsList = ['/home/lukas/Documents/projects/deepmedic/examples/configFiles/deepMedicBRATS15/train/splits/trainFlair.cfg',
                    '/home/lukas/Documents/projects/deepmedic/examples/configFiles/deepMedicBRATS15/train/splits/trainT1.cfg',
                    '/home/lukas/Documents/projects/deepmedic/examples/configFiles/deepMedicBRATS15/train/splits/trainT1c.cfg',
                    '/home/lukas/Documents/projects/deepmedic/examples/configFiles/deepMedicBRATS15/train/splits/trainT2.cfg']
    groundTruthChannel_list = '/home/lukas/Documents/projects/deepmedic/examples/configFiles/deepMedicBRATS15/train/splits/trainGT.cfg'

    channelsList_validation = ['/home/lukas/Documents/projects/deepmedic/examples/configFiles/deepMedicBRATS15/train/validation/splits/valFlair.cfg',
                               '/home/lukas/Documents/projects/deepmedic/examples/configFiles/deepMedicBRATS15/train/validation/splits/valT1.cfg',
                               '/home/lukas/Documents/projects/deepmedic/examples/configFiles/deepMedicBRATS15/train/validation/splits/valT1c.cfg',
                               '/home/lukas/Documents/projects/deepmedic/examples/configFiles/deepMedicBRATS15/train/validation/splits/valT2.cfg']
    groundTruthChannel_list_validation = '/home/lukas/Documents/projects/deepmedic/examples/configFiles/deepMedicBRATS15/train/validation/splits/valGT.cfg'
    
    testChannels = ['/home/lukas/Documents/projects/deepmedic/examples/configFiles/deepMedicBRATS15/test/splits/testFlair.cfg',
                    '/home/lukas/Documents/projects/deepmedic/examples/configFiles/deepMedicBRATS15/test/splits/testT1.cfg',
                    '/home/lukas/Documents/projects/deepmedic/examples/configFiles/deepMedicBRATS15/test/splits/testT1c.cfg',
                    '/home/lukas/Documents/projects/deepmedic/examples/configFiles/deepMedicBRATS15/test/splits/testT2.cfg']
    testLabels = '/home/lukas/Documents/projects/deepmedic/examples/configFiles/deepMedicBRATS15/test/splits/testGT.cfg'
    output_classes = 5 # Including background!!

elif dataset == 'ATLAS17':
    '''Example data ATLAS 2017 - 1 input channel - 2 output classes'''
    
    channelsList = ['/home/lukas/Documents/projects/deepmedic/examples/configFiles/deepMedicATLAS/train/splits1/ChannelsTrain_splits1.cfg']
    groundTruthChannel_list = '/home/lukas/Documents/projects/deepmedic/examples/configFiles/deepMedicATLAS/train/splits1/SegmentsTrain_splits1.cfg'
    
    channelsList_validation = ['/home/lukas/Documents/projects/deepmedic/examples/configFiles/deepMedicATLAS/train/validation/splits1/ChannelsVal_splits1.cfg']
    groundTruthChannel_list_validation = '/home/lukas/Documents/projects/deepmedic/examples/configFiles/deepMedicATLAS/train/validation/splits1/SegmentsVal_splits1.cfg'

    testChannels = ['/home/lukas/Documents/projects/deepmedic/examples/configFiles/deepMedicATLAS/test/splits1/ChannelsTest_splits1.cfg']
    testLabels = '/home/lukas/Documents/projects/deepmedic/examples/configFiles/deepMedicATLAS/test/splits1/SegmentsTest_splits1.cfg'

    output_classes = 2 # Including background!!
    
    
usingLoadedData = False
allBatchesAddress = '/home/lukas/Documents/projects/headSegmentation/deepMedicKeras/allBatches.pkl'
allLabelsAddress =  '/home/lukas/Documents/projects/headSegmentation/deepMedicKeras/allLabels.pkl'
allValBatchesAddress = '/home/lukas/Documents/projects/headSegmentation/deepMedicKeras/allValBatches.pkl'
allValLabelsAddress = '/home/lukas/Documents/projects/headSegmentation/deepMedicKeras/allValLabels.pkl'

if(usingLoadedData):
    print("Loading collected data")
    import pickle
    with open(allBatchesAddress,'rb') as fp:
        print(allBatchesAddress)
        allBatches = pickle.load(fp)
    with open(allLabelsAddress,'rb') as fp:
        print(allLabelsAddress)
        allLabels = pickle.load(fp)
    with open(allValBatchesAddress,'rb') as fp:
        print(allValBatchesAddress)
        allValBatches = pickle.load(fp)
    with open(allValLabelsAddress,'rb') as fp:
        print(allValLabelsAddress)
        allValLabels = pickle.load(fp)

# Parameters // this could be assigned in separate configuration files

# MODEL PARAMETERS
num_channels = len(channelsList)
dpatch=51
L2 = 0

# TRAIN PARAMETERS
num_iter = 1
epochs = 10
n_patches = 10
n_patches_val = 10
n_subjects = 2
samplingMethod = 1
size_minibatches = 1


# TEST PARAMETERS
list_subjects_fullSegmentation = [0]
epochs_for_fullSegmentation = [0,3,6,9]
size_test_minibatches = 200

load_model = False

############################## create model ###########################################
if load_model == False:
    dm = DeepMedic(dpatch, output_classes, num_channels, L2)
    model = dm.createModel()
    #plot_model(model, to_file='multichannel.png', show_shapes=True)

############################## train model ###########################################

train_performance = []
val_performance = []
val1_performance = []

#allForegroundVoxels = generateAllForegroundVoxels(groundTruthChannel_list, dpatch)

l = 0
t1 = time.time()

for epoch in xrange(0,epochs):
    print("######################################################")
    print("TRAINING EPOCH " + str(epoch) + "/" + str(epochs))
    print("######################################################")

    
    if(usingLoadedData):
        k = [i for i in range(len(allBatches))]
        shuffle(k)
    
    for i in range(0, num_iter):
    
        #batch, labels = extractPatches(filename,labelsFileName, n_patches, n_subjects, dpatch, output_classes)
        #valbatch, vallabels = extractPatches(valfilename, vallabelsfilename, n_patches, n_subjects, dpatch, output_classes)
    
        if(usingLoadedData == False):
            print("Extracting image patches for training")
            batch, labels = sampleTrainData(channelsList, groundTruthChannel_list, n_patches, n_subjects, dpatch, output_classes, samplingMethod)
            
            #print(classesInSample(labels, output_classes))
            
            print("Extracting image patches for Validation")
            valbatch, vallabels = sampleTrainData(channelsList_validation, groundTruthChannel_list_validation, n_patches_val, n_subjects, dpatch, output_classes, samplingMethod)
            #valbatch1, vallabels1 = sampleTrainData(channelsList_validation, groundTruthChannel_list_validation, n_patches_val1, n_subjects, dpatch, output_classes, samplingMethod=0)
            
        
        # A batch is too big, divide in mini-batches 
            start = 0
            n_minibatches = len(batch)/size_minibatches
            for j in range(0,n_minibatches):
                #print("training on minibatch " +str(j)+ "/" + str(n_minibatches))
                end = start + size_minibatches
                minibatch = batch[start:end,:,:,:,:]    
                minibatch_labels = labels[start:end,:,:,:,:]    
                train_performance.append(model.train_on_batch(minibatch, minibatch_labels))#, class_weight = class_weight))
                minivalbatch = valbatch[start:end,:,:,:,:]    
                minivalbatch_labels = vallabels[start:end,:,:,:,:]    
                val_performance.append(model.evaluate(minivalbatch, minivalbatch_labels))
                #minivalbatch1 = valbatch1[start:end,:,:,:,:]    
                #minivalbatch_labels1 = vallabels1[start:end,:,:,:,:]  
                #val1_performance.append(model.evaluate(minivalbatch1, minivalbatch_labels1))
                start = end
        else:
            train_performance.append(model.train_on_batch(allBatches[k[i]], allLabels[k[i]]))#, class_weight = class_weight1)
            val_performance.append(model.evaluate(allValBatches[k[i]], allValLabels[k[i]]))
            
        print(str(i) + '/' + str(num_iter))
        print(train_performance[-1])
        l = l+1
    print('Total training this epoch took seconds:')
    print(round(time.time()-t1,2))
    if epoch in epochs_for_fullSegmentation:
        test_performance = []
        for subjectIndex in list_subjects_fullSegmentation:
            fullHeadSegmentation(model, testChannels, testLabels, subjectIndex, output_classes, dpatch, size_test_minibatches, saveSegmentation = False)
        
#https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model
#from keras.models import load_model     
#model.save('Brats15_model2.h5')
#model.save_weights('Brats15_model2_weights.h5')
#model = load_model('Brats15_model2.h5')
#json_string = model.to_json()

#model = model_from_json(open(modelFile).read())
#model.load_weights(os.path.join(os.path.dirname(modelFile), 'model_weights.h5'))

#If you only need to save the architecture of a model, and not its weights or its training configuration, you can do:
#json_string = model.to_json()
#%%############################# plot ##################################################

plt.clf()
plt.subplot(311)
ax = plt.gca()
t0 = [row[0] for row in train_performance]
v0 = [row[0] for row in val_performance]
#v00 = [row[0] for row in val1_performance]

#vl0 = [x for x in val_performance if x != []]
#vl01 = [x for x in val1_performance if x != []]


plt.plot(range(0,len(t0)),t0,'-',v0)#,'-',vl01,'-')
#ax.plot( np.concatenate((train_performance[:,[0]], val_performance[:,[0]]),1))
plt.xlabel('epochs')
plt.ylabel('loss')
plt.axis('tight')
plt.legend(('train set', 'validation set','uniform sample validation set'))
plt.subplot(312)
ax = plt.gca()
t1 = [row[1] for row in train_performance]
v1 = [row[1] for row in val_performance]
#v01 = [row[1] for row in val1_performance]
plt.plot(range(0,len(t1)),t1,'-',v1)#,'-',v01,'-')
#ax.plot( np.concatenate((train_performance[:,[1]], val_performance[:,[1]]),1))
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.axis('tight')
plt.legend(('train set', 'validation set'))
plt.subplot(313)
ax = plt.gca()
t2 = [row[2] for row in train_performance]
v2 = [row[2] for row in val_performance]
#v02 = [row[2] for row in val1_performance]
plt.plot(range(0,len(t2)),t2,'-',v2)#,'-',v02,'-')
#ax.plot( np.concatenate((train_performance[:,[1]], val_performance[:,[1]]),1))
plt.xlabel('epochs')
plt.ylabel('F1')
plt.axis('tight')
plt.legend(('train set', 'validation set'))
plt.show()


