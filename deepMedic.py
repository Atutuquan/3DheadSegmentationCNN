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
from helpers import sampleTrainData, fullHeadSegmentation, my_logger, classesInSample
from random import shuffle
import matplotlib
from keras.callbacks import ModelCheckpoint#, EarlyStopping, History

from configFile import *   # Get all session parameters

os.chdir(wd)


logfile = 'Output/logs/TrainSession' + dataset + '_DeepMedic' + time.strftime("%Y-%m-%d_%H%M")

# Load dataset

if dataset == 'BRATS15':
    '''Example data BRATS 2015 - 4 input channels - 5 output classes'''
    
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

elif dataset == 'ATLAS17':
    '''Example data ATLAS 2017 - 1 input channel - 2 output classes'''
    
    trainChannels = ['/home/lukas/Documents/projects/deepmedic/examples/configFiles/deepMedicATLAS/train/splits1/ChannelsTrain_splits1.cfg']
    trainLabels = '/home/lukas/Documents/projects/deepmedic/examples/configFiles/deepMedicATLAS/train/splits1/SegmentsTrain_splits1.cfg'
    
    validationChannels = ['/home/lukas/Documents/projects/deepmedic/examples/configFiles/deepMedicATLAS/train/validation/splits1/ChannelsVal_splits1.cfg']
    validationLabels = '/home/lukas/Documents/projects/deepmedic/examples/configFiles/deepMedicATLAS/train/validation/splits1/SegmentsVal_splits1.cfg'

    testChannels = ['/home/lukas/Documents/projects/deepmedic/examples/configFiles/deepMedicATLAS/test/splits1/ChannelsTest_splits1.cfg']
    testLabels = '/home/lukas/Documents/projects/deepmedic/examples/configFiles/deepMedicATLAS/test/splits1/SegmentsTest_splits1.cfg'

    output_classes = 2 # Including background!!
    
    
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


############################## create model ###########################################
num_channels = len(trainChannels)
        
if load_model == False:
    dm = DeepMedic(dpatch, output_classes, num_channels, L2, dropout, learning_rate, optimizer_decay)
    model = dm.createModel()
    print(model.summary())
    #plot_model(model, to_file='multichannel.png', show_shapes=True)
    model_path = wd+'/Output/models/'+logfile[12:]+'.h5'



elif load_model == True:
    from keras.models import load_model  
    model = load_model(path_to_model)
    
    
############################## train model ###########################################

train_performance = []
val_performance = []
val1_performance = []

#allForegroundVoxels = generateAllForegroundVoxels(trainLabels, dpatch)

l = 0
t1 = time.time()


my_logger('#######################################  NEW TRAINING SESSION  #######################################', logfile)    
my_logger(trainChannels, logfile)
my_logger(trainLabels, logfile)
my_logger(validationChannels, logfile)        
my_logger(validationLabels, logfile)  
my_logger(testChannels, logfile) 
my_logger(testLabels, logfile) 
my_logger('Session parameters: ', logfile)
my_logger('[num_iter, epochs, n_patches, n_patches_val, n_subjects, samplingMethod, size_minibatches, list_subjects_fullSegmentation, epochs_for_fullSegmentation, size_test_minibatches]', logfile)
my_logger([num_iter, epochs, n_patches, n_patches_val, n_subjects, samplingMethod, size_minibatches, list_subjects_fullSegmentation, epochs_for_fullSegmentation, size_test_minibatches], logfile)
my_logger('Dropout for last two fully connected layers: ' + str(dropout), logfile)
my_logger('Save full head segmentation of subjects: ' + str(saveSegmentation), logfile)
if load_model:
    my_logger("USING PREVIOUSLY SAVED MODEL -  Model retrieved from: " + path_to_model, logfile)

for epoch in xrange(0,epochs):
    my_logger("######################################################",logfile)
    my_logger("                   TRAINING EPOCH " + str(epoch) + "/" + str(epochs),logfile)
    my_logger("######################################################",logfile)

    
    if(usingLoadedData):
        k = [i for i in range(len(allBatches))]
        shuffle(k)
    
    for i in range(0, num_iter):
        
        if(usingLoadedData):
            train_performance.append(model.train_on_batch(allBatches[k[i]], allLabels[k[i]]))#, class_weight = class_weight1)
            val_performance.append(model.evaluate(allValBatches[k[i]], allValLabels[k[i]]))
            
        else:
            batch, labels = sampleTrainData(trainChannels,trainLabels, n_patches, n_subjects, dpatch, output_classes, samplingMethod, logfile)
            
            my_logger("Sampled following number of classes in training batch: " + str(classesInSample(labels, output_classes)), logfile)
            
            valbatch, vallabels = sampleTrainData(validationChannels, validationLabels, n_patches_val, n_subjects_val, dpatch, output_classes, samplingMethod, logfile)
            #valbatch1, vallabels1 = sampleTrainData(validationChannels, validationLabels, n_patches_val1, n_subjects, dpatch, output_classes, samplingMethod=0)
            
        
        # A batch is too big, divide in mini-batches 
        #TRAINING ON BATCHES
            start = 0
            n_minibatches = len(batch)/size_minibatches
            for j in range(0,n_minibatches):
                print("training on minibatch " +str(j)+ "/" + str(n_minibatches))
                end = start + size_minibatches
                minibatch = batch[start:end,:,:,:,:]    
                minibatch_labels = labels[start:end,:,:,:,:]    
                train_performance.append(model.train_on_batch(minibatch, minibatch_labels))#, class_weight = class_weight))
                #minivalbatch = valbatch[start:end,:,:,:,:]    
                #minivalbatch_labels = vallabels[start:end,:,:,:,:]    
                #val_performance.append(model.evaluate(minivalbatch, minivalbatch_labels))
                #minivalbatch1 = valbatch1[start:end,:,:,:,:]    
                #minivalbatch_labels1 = vallabels1[start:end,:,:,:,:]  
                #val1_performance.append(model.evaluate(minivalbatch1, minivalbatch_labels1))
                start = end
            my_logger(str(i) + '/' + str(num_iter),logfile)
            my_logger('Train cost and accuracy      ' + str(train_performance[-1]),logfile)
                
            # VALIDATION ON BATCHES
            start = 0
            n_minibatches = len(valbatch)/size_minibatches
            for j in range(0,n_minibatches):
                print("validation on minibatch " +str(j)+ "/" + str(n_minibatches))
                end = start + size_minibatches
                #minibatch = batch[start:end,:,:,:,:]    
                #minibatch_labels = labels[start:end,:,:,:,:]    
                #train_performance.append(model.train_on_batch(minibatch, minibatch_labels))#, class_weight = class_weight))
                minivalbatch = valbatch[start:end,:,:,:,:]    
                minivalbatch_labels = vallabels[start:end,:,:,:,:]    
                val_performance.append(model.evaluate(minivalbatch, minivalbatch_labels))
                #minivalbatch1 = valbatch1[start:end,:,:,:,:]    
                #minivalbatch_labels1 = vallabels1[start:end,:,:,:,:]  
                #val1_performance.append(model.evaluate(minivalbatch1, minivalbatch_labels1))
                start = end
            my_logger('Validation cost and accuracy ' + str(val_performance[-1]),logfile)
        l = l+1
    my_logger('Total training this epoch took ' + str(round(time.time()-t1,2)) + ' seconds',logfile)
    if epoch in epochs_for_fullSegmentation:
        my_logger("------------------------------------------------------", logfile)
        my_logger("                 FULL HEAD SEGMENTATION", logfile)
        my_logger("------------------------------------------------------", logfile)
        test_performance = []
        for subjectIndex in list_subjects_fullSegmentation:
            fullHeadSegmentation(model, testChannels, testLabels, subjectIndex, output_classes, dpatch, size_test_minibatches, logfile,epoch, saveSegmentation)
        #my_logger('--------------- TEST EVALUATION ---------------', logfile)
        #my_logger(np.average(test_performance,axis=0),logfile)
#https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model


# SAVING AND LOADING A MODEL. Works fine when not using non-standard keras library metrics (e.g. dice)
my_logger('###### SAVING TRAINED MODEL AT : ' + wd+'/Output/models/'+logfile[12:]+'.h5', logfile)
model.save(wd+'/Output/models/'+logfile[12:]+'.h5')
#model.save_weights(wd+'/Output/models/'+logfile[12:]+'_weights_.h5')



#%%############################# plot ##################################################

plt.clf()
plt.subplot(211)
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
plt.subplot(212)
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

matplotlib.pyplot.savefig(logfile + '_Training.png')
