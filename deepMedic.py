#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 13:58:48 2017

@author: lukas
"""
import os
import time
from model import DeepMedic
from tiny_DeepMedic import tiny_DeepMedic
from helpers import sampleTrainData, fullHeadSegmentation, my_logger, classesInSample, getClassProportions, evaluation_metrics
from random import shuffle
from random import sample
from keras.callbacks import ModelCheckpoint#, EarlyStopping, History
import numpy as np
from configFile import *   # Get all session parameters

os.chdir(wd)
logfile = 'Output/logs/TrainSession' + dataset + '_DeepMedic' + time.strftime("%Y-%m-%d_%H%M")

############################## create model ###########################################
        
#np.random.seed(1337) # for reproducibility
        
if load_model == False:
    if(usingAlternativeModel):
        dm = tiny_DeepMedic(dpatch, output_classes, num_channels, L2, dropout, learning_rate, optimizer_decay)
        model = dm.createModel()
    else:
        dm = DeepMedic(dpatch, output_classes, num_channels, L2, dropout, learning_rate, optimizer_decay)
        model = dm.createModel()
    print(model.summary())
    train_performance = []
    val_performance = []
    #plot_model(model, to_file='multichannel.png', show_shapes=True)
    model_path = wd+'/Output/models/'+logfile[12:]+'.h5'
elif load_model == True:
    from keras.models import load_model  
    model = load_model(path_to_model)
    logfile = 'Output/logs/'  + logfile_model
    
    
############################## train model ###########################################

# OUTCOMMENTED SO I CAN KEEP USING SAME TRAINING DATA FOR SAME MODEL.
val_performance = []
np.set_printoptions(precision=3)

#allForegroundVoxels = generateAllForegroundVoxels(trainLabels, dpatch)
l = 0

my_logger('#######################################  NEW TRAINING SESSION  #######################################', logfile)    
my_logger(trainChannels, logfile)
my_logger(trainLabels, logfile)
my_logger(validationChannels, logfile)        
my_logger(validationLabels, logfile)  
my_logger(testChannels, logfile) 
my_logger(testLabels, logfile) 
my_logger('Session parameters: ', logfile)
my_logger('[num_iter, epochs, n_patches, n_patches_val, n_subjects, samplingMethod_train, size_minibatches, n_fullSegmentations, epochs_for_fullSegmentation, size_test_minibatches]', logfile)
my_logger([num_iter, epochs, n_patches, n_patches_val, n_subjects, samplingMethod_train, size_minibatches, n_fullSegmentations, epochs_for_fullSegmentation, size_test_minibatches], logfile)
my_logger('Dropout for last two fully connected layers: ' + str(dropout), logfile)
my_logger('Model loss function: ' + str(model.loss), logfile)
my_logger('Model number of parameters: ' + str(model.count_params()), logfile)
my_logger('Optimizer used: ' +  str(model.optimizer.from_config), logfile)
my_logger('Optimizer parameters: ' + str(model.optimizer.get_config()), logfile)
my_logger('Save full head segmentation of subjects: ' + str(saveSegmentation), logfile)
if load_model:
    my_logger("USING PREVIOUSLY SAVED MODEL -  Model retrieved from: " + path_to_model, logfile)

for epoch in xrange(0,epochs):
    t1 = time.time()
    my_logger("######################################################",logfile)
    my_logger("                   TRAINING EPOCH " + str(epoch+1) + "/" + str(epochs),logfile)
    my_logger("######################################################",logfile)
    
    for i in range(0, num_iter):
        my_logger("                   Batch " + str(i+1) + "/" + str(num_iter) ,logfile)
        my_logger("###################################################### ",logfile)
        valbatch, vallabels = sampleTrainData(validationChannels, validationLabels, n_patches_val, n_subjects_val, dpatch, output_classes, samplingMethod_val, logfile)
        #valbatch1, vallabels1 = sampleTrainData(validationChannels, validationLabels, n_patches_val1, n_subjects, dpatch, output_classes, samplingMethod=0)
        positives = []
        negatives = []
        truePositives = []
        trueNegatives = []
        falsePositives = []
        falseNegatives = []
        sens = []
        spec = []
        Dice = []
        accuracy = []  # per class
        total_accuracy = []  # as a whole
        auc_roc = []
        
        ############################# VALIDATION ON BATCHES ############################
        
        start = 0
        n_minibatches = len(valbatch)/size_minibatches_val
        for j in range(0,n_minibatches):
            print("validation on minibatch " +str(j+1)+ "/" + str(n_minibatches))
            
            end = start + size_minibatches_val
            minivalbatch = valbatch[start:end,:,:,:,:]    
            minivalbatch_labels = vallabels[start:end,:,:,:,:]    
            val_performance.append(model.evaluate(minivalbatch, minivalbatch_labels))

            #my evaluation
            #freq = classesInSample(minivalbatch_labels, output_classes)
            #my_logger("Sampled following number of classes in VALIDATION : " + str(freq), logfile)
            #my_logger("proportions: " + str(getClassProportions(freq)), logfile)
            
            prediction = model.predict(minivalbatch, verbose=0)
            class_pred = np.argmax(prediction, axis=4)  
            P,N,TP,TN,FP,FN,ACC,acc,roc =  evaluation_metrics(class_pred, prediction, output_classes, minivalbatch_labels )
            positives.append(P)
            negatives.append(N)
            truePositives.append(TP)
            trueNegatives.append(TN)
            falsePositives.append(FP)
            falseNegatives.append(FN)
            accuracy.append(ACC)  # per class
            total_accuracy.append(acc)
            auc_roc.append(roc)
            start = end
            my_logger('Validation cost and accuracy ' + str(val_performance[-1]),logfile) 
             
        del valbatch
        del vallabels
            
        sumTP = np.sum(truePositives,0)
        sumTN = np.sum(trueNegatives,0)
        sumP = np.sum(positives,0)
        sumN = np.sum(negatives,0)
        sumFP = np.sum(falsePositives,0)
        sumFN = np.sum(falseNegatives,0)    
        
        total_sens = np.divide(np.array(sumTP,dtype='float32'),np.array(sumP,dtype='float32'))
        total_spec = np.divide(np.array(sumTN,dtype='float32'),np.array(sumN,dtype='float32'))
        total_precision = np.divide(np.array(sumTP,dtype='float32'),(np.array(sumTP,dtype='float32') + np.array(sumFP,dtype='float32')))
        NPV = np.divide(np.array(sumTN,dtype='float32'),(np.array(sumTN,dtype='float32') + np.array(sumFN,dtype='float32')))
       
        total_DSC = np.divide(2*np.array(sumTP,dtype='float64'),(2 * np.array(sumTP,dtype='float64') + np.array(sumFP,dtype='float64') + np.array(sumFN,dtype='float64')))
         
        mean_acc = np.average(accuracy, axis=0)
        mean_total_accuracy = np.average(total_accuracy, axis=0)
        mean_AUC_ROC = np.average(auc_roc, axis=0)
        
        my_logger('--------------- VALIDATION EVALUATION ---------------', logfile)
        my_logger('Mean Accuracy :' + str(np.round(mean_total_accuracy,4)) + ' => Correctly-Classified-Voxels/All-Predicted-Voxels = ' + str(np.sum([x[:-1] for x in truePositives]) + np.sum([x[:-1] for x in trueNegatives])) + '/' + str(np.sum([x[:-1] for x in positives]) + np.sum([x[:-1] for x in negatives])) , logfile)
        my_logger('Per class Accuracy :            ' + str(np.round(mean_acc,4)), logfile)
        my_logger('Per class Sensitivity :         ' + str(np.round(total_sens,4)), logfile)
        my_logger('Per class Specificity :         ' + str(np.round(total_spec,4)), logfile)
        my_logger('Per class Precision :           ' + str(np.round(total_precision,4)), logfile)
        my_logger('Negative predictive value :     ' + str(np.round(NPV,4)), logfile)
        my_logger('Per class DCS :                 ' + str(np.round(total_DSC,4)), logfile)
        my_logger('Per class  AUC-ROC :            ' + str(np.round(mean_AUC_ROC, 4)), logfile)
        
        
        ####################### TRAINING ON BATCHES ##############################
        
        batch, labels = sampleTrainData(trainChannels,trainLabels, n_patches, n_subjects, dpatch, output_classes, samplingMethod_train, logfile)
        
        shuffleOrder = np.arange(batch.shape[0])
        np.random.shuffle(shuffleOrder)
        batch = batch[shuffleOrder]
        labels = labels[shuffleOrder]
        #freq = classesInSample(labels, output_classes)
        #my_logger("Sampled following number of classes in training batch: " + str(freq), logfile)
        #print(getClassProportions(freq))
        start = 0
        n_minibatches = len(batch)/size_minibatches
        for j in range(0,n_minibatches):
            print("training on minibatch " +str(j+1)+ "/" + str(n_minibatches))
            end = start + size_minibatches
            minibatch = batch[start:end,:,:,:,:]    
            minibatch_labels = labels[start:end,:,:,:,:]   
            #freq = classesInSample(minibatch_labels, output_classes)
            #my_logger("Sampled following number of classes in training MINIBATCH: " + str(freq), logfile)
            #print(getClassProportions(freq))
            train_performance = model.train_on_batch(minibatch, minibatch_labels)#, class_weight = class_weight))
            start = end
            my_logger('Train cost and accuracy      ' + str(train_performance),logfile)
            
        del batch
        del labels
        l = l+1
    my_logger('Total training this epoch took ' + str(round(time.time()-t1,2)) + ' seconds',logfile)
    
    
    ####################### FULL HEAD SEGMENTATION ##############################
    
    if epoch in epochs_for_fullSegmentation:
        my_logger("------------------------------------------------------", logfile)
        my_logger("                 FULL HEAD SEGMENTATION", logfile)
        my_logger("------------------------------------------------------", logfile)
        test_performance = []
        list_subjects_fullSegmentation = sample(range(test_subjects),n_fullSegmentations)
        statistics = []
        statistics_mean = []
        for subjectIndex in list_subjects_fullSegmentation: 
            mean_sens, mean_spec, mean_DICE, mean_acc, mean_total_accuracy, mean_AUC_ROC, total_precision = fullHeadSegmentation(model, testChannels, testLabels, subjectIndex, output_classes, dpatch, size_test_minibatches, logfile,epoch, saveSegmentation)
            my_logger('--------------- TEST EVALUATION ---------------', logfile)
            my_logger('          Full segmentation evaluation of subject' + str(subjectIndex), logfile)
            my_logger('Mean Accuracy :' + str(np.round(mean_total_accuracy,4)) + ' => Correctly-Classified-Voxels/All-Predicted-Voxels = ' + str(np.sum([x[:-1] for x in truePositives]) + np.sum([x[:-1] for x in trueNegatives])) + '/' + str(np.sum([x[:-1] for x in positives]) + np.sum([x[:-1] for x in negatives])) , logfile)
            my_logger('Per class Accuracy :            ' + str(np.round(mean_acc,4)), logfile)
            my_logger('Per class Specificity :         ' + str(np.round(mean_spec,4)), logfile)
            my_logger('Per class Sensitivity :         ' + str(np.round(mean_sens,4)), logfile)
            my_logger('Per class Precision :           ' + str(np.round(total_precision,4)), logfile)
            my_logger('Per class DCS :                 ' + str(np.round(mean_DICE,4)), logfile)
            my_logger('Per class  AUC-ROC :            ' + str(np.round(mean_AUC_ROC, 4)), logfile)
            statistics.append([mean_sens, mean_spec, mean_DICE, mean_acc, mean_total_accuracy, mean_AUC_ROC, total_precision])
         
        for i in range(len(statistics[0])):
            s = [item[i] for item in statistics]
            m = np.nanmean(s,0)
            statistics_mean.append(m)

        my_logger('         FULL SEGMENTATION SUMMARY STATISTICS ', logfile)
        my_logger('Mean Accuracy:                         ' + str(statistics_mean[4]), logfile)
        my_logger('Overall Accuracy:                      ' + str(statistics_mean[3]), logfile)
        my_logger('Overall Specificity:                   ' + str(statistics_mean[1]), logfile)
        my_logger('Overall Sensitivity:                   ' + str(statistics_mean[0]), logfile)
        my_logger('Overall Precision:                     ' + str(statistics_mean[6]), logfile)
        my_logger('Overall DCS:                           ' + str(statistics_mean[2]), logfile)
        my_logger('Overall AUC-ROC:                       ' + str(statistics_mean[5]), logfile)

    my_logger('###### SAVING TRAINED MODEL AT : ' + wd +'/Output/models/'+logfile[12:]+'.h5', logfile)
    model.save(wd+'/Output/models/'+logfile[12:]+'.h5')
    #model.save_weights(wd+'/Output/models/'+logfile[12:]+'_weights_.h5')


