#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 11:35:14 2017

@author: lukas
"""

import nibabel as nib
import numpy as np
import time
from keras.utils import to_categorical
import random
import os
from sklearn import metrics
import matplotlib.pyplot as plt

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def generateRandomIndexesSubjects(n_subjects, total_subjects):
    indexSubjects = random.sample(xrange(total_subjects), n_subjects)
    return indexSubjects

def getSubjectChannels(subjectIndexes, channel):
    "With the channels (any modality) and the indexes of the selected subjects, return the addresses of the subjects channels"
    fp = open(channel)
    # read file, per subject index extract patches given the indexesPatch
    lines = fp.readlines()
    selectedSubjects = [lines[i][:-1] for i in subjectIndexes]
    fp.close()
    return selectedSubjects

def getSubjectShapes(subjectIndexes, n_patches, channelList):
    # Need to open every nifty file and get the shapes
    fp = open(channelList)
    # read file, per subject index extract patches given the indexesPatch
    lines = fp.readlines()
    selectedSubjects = [lines[i] for i in subjectIndexes]
    fp.close()
    shapes = []
    # Get shapes of all subjects to sample from. Can be a separate function (cause apparently I am needing this everywhere)
    for subjectChannel in selectedSubjects:
        subjectChannel = str(subjectChannel)[:-1]
        proxy_img = nib.load(subjectChannel)
        shapes.append(proxy_img.shape)
    return shapes      


def generateVoxelIndexes(subjectIndexes, shapes, patches_per_subject, dpatch, n_patches, groundTruthChannel_list, samplingMethod, output_classes, allForegroundVoxels = ""):
    "Alternative improved function of the same named one above."
    "Here extract the channels from the subject indexes, and loop over them. Then in second loop extract as many needed voxel coordinates per subject."
    methods =["Random sampling","Equal sampling background/foreground","Equal sampling all classes center voxel","Equal sampling background/foreground with exhaustive foreground samples"]
    print("Generating voxel indexes with method: " + methods[samplingMethod])
    channels = getSubjectChannels(subjectIndexes, groundTruthChannel_list)
    allVoxelIndexes = []
    
    if samplingMethod == 0:
        for i in xrange(0, len(shapes)):
            voxelIndexesSubj = []
            #loop over voxels per subject
            for j in range(0,patches_per_subject[i]):
                # unform sampling
                voxelIndexesSubj.append((np.random.randint(0+dpatch/2, shapes[i][0]-(dpatch/2)-1),np.random.randint(0+dpatch/2, shapes[i][1]-(dpatch/2)-1),np.random.randint(0+dpatch/2, shapes[i][2]-(dpatch/2)-1)))

            allVoxelIndexes.append(voxelIndexesSubj)
            
        random.shuffle(allVoxelIndexes[i])
        return allVoxelIndexes
    
    elif samplingMethod == 1:
        "This samples equally background/foreground. Assumption that foreground is very seldom: Only foreground voxels are sampled, and background voxels are just random samples which are then proofed against foreground ones"
        "Still need to proof for repetition. Although unlikely and uncommon"
        for i in range(0,len(channels)): 
            voxelIndexesSubj = []
            backgroundVoxels = []
            fg = getForegroundBackgroundVoxels(channels[i], dpatch) # This function returns only foreground voxels
            foregroundVoxels = fg[random.sample(xrange(0,len(fg)), patches_per_subject[i]/2 + patches_per_subject[i]%2)].tolist()
            #print("Foreground voxel extracted " + str(foregroundVoxels) + " from channel " + str(channels[i]) + " with index " + str(i))
            # get random voxel coordinates
            for j in range(0,patches_per_subject[i]/2):
                backgroundVoxels.append((np.random.randint(0+dpatch/2, shapes[i][0]-(dpatch/2)-1),np.random.randint(0+dpatch/2, shapes[i][1]-(dpatch/2)-1),np.random.randint(0+dpatch/2, shapes[i][2]-(dpatch/2)-1)))
                
            #Replace the ones that by chance are foreground voxels (not so many in tumor data)
            while any([e for e in foregroundVoxels if e in backgroundVoxels]):
                ix = [e for e in foregroundVoxels if e in backgroundVoxels]
                for index in ix:
                    newVoxel = [np.random.randint(dpatch/2, shapes[i][0]-(dpatch/2)-1),np.random.randint(dpatch/2, shapes[i][1]-(dpatch/2)-1),np.random.randint(dpatch/2, shapes[i][2]-(dpatch/2)-1)]
                    backgroundVoxels[backgroundVoxels.index(index)] = newVoxel


            #backgroundVoxels = bg[random.sample(xrange(0,len(bg)), patches_per_subject[i]/2)].tolist()
            allVoxelIndexes.append(foregroundVoxels + backgroundVoxels)
            random.shuffle(allVoxelIndexes[i])
        return allVoxelIndexes
    
    elif samplingMethod == 2:
        "sample from each class equally"
        
        "use function getAllForegroundClassesVoxels to get coordinates from all classes (not including background)"
        
        for i in range(0,len(channels)):  # iteration over subjects
            voxelIndexesSubj = []
            backgroundVoxels = []
            fg = getAllForegroundClassesVoxels(channels[i], dpatch, output_classes) # This function returns only foreground voxels
            
            # WATCH OUT, FG IS A LIST OF LISTS. fIRST DIMENSION IS THE CLASS, SECOND IS THE LIST OF VOXELS OF THAT CLASS
            foregroundVoxels = []
            patches_to_sample = [patches_per_subject[i]/output_classes] * output_classes #
            extra = random.sample(range(output_classes),1)
            patches_to_sample[extra[0]] = patches_to_sample[extra[0]] + patches_per_subject[i]%output_classes
            
            for c in range(0, output_classes-1):
                foregroundVoxels.extend(fg[c][random.sample(xrange(0,len(fg[c])), min(patches_to_sample[c],len(fg[c])))].tolist())
                #foregroundVoxels.extend([1] * min(patches_to_sample[c],len(fg[c])))
                #print("Foreground voxel extracted " + str(foregroundVoxels) + " from channel " + str(channels[i]) + " with index " + str(i))
            # get random voxel coordinates
            for j in range(0,patches_per_subject[i]/output_classes):
                backgroundVoxels.append([np.random.randint(0+dpatch/2, shapes[i][0]-(dpatch/2)-1),np.random.randint(0+dpatch/2, shapes[i][1]-(dpatch/2)-1),np.random.randint(0+dpatch/2, shapes[i][2]-(dpatch/2)-1)])
                #backgroundVoxels.extend([0])
            #Replace the ones that by chance are foreground voxels (not so many in tumor data)
            while any([e for e in foregroundVoxels if e in backgroundVoxels]):
                ix = [e for e in foregroundVoxels if e in backgroundVoxels]
                for index in ix:
                    newVoxel = [np.random.randint(dpatch/2, shapes[i][0]-(dpatch/2)-1),np.random.randint(dpatch/2, shapes[i][1]-(dpatch/2)-1),np.random.randint(dpatch/2, shapes[i][2]-(dpatch/2)-1)]
                    backgroundVoxels[backgroundVoxels.index(index)] = newVoxel

            #backgroundVoxels = bg[random.sample(xrange(0,len(bg)), patches_per_subject[i]/2)].tolist()
            allVoxelIndexes.append(foregroundVoxels + backgroundVoxels)
            random.shuffle(allVoxelIndexes[i])
        return allVoxelIndexes
    '''
    #samplingMethod 2 : include foreground Voxels in arguments for this option. These were poped from the total list. Then generate background voxels just like for method 1
    elif samplingMethod == 3:

        for i in range(0,len(channels)): 
            voxelIndexesSubj = []
            backgroundVoxels = []
             # This function returns only foreground voxels
            fg_voxels = random.sample(xrange(0,len(allForegroundVoxels[i])), patches_per_subject[i]/2 + patches_per_subject[i]%2)
    
            for voxelIndex in fg_voxels:
                foregroundVoxels.append(allForegroundVoxels[subjectIndexes[i]].pop(voxelIndex)) # The pop function is what keeps shortening the list. "PROBLEM LATER WHEN LIST IS ALMOST EMPTY AND HAS LESS ELEMENTS THAN PATCHES PER SUBJECT"
    

            #print("Foreground voxel extracted " + str(foregroundVoxels) + " from channel " + str(channels[i]) + " with index " + str(i))
            # get random voxel coordinates
            for j in range(0,patches_per_subject[i]/2):
                backgroundVoxels.append((np.random.randint(0+dpatch/2, shapes[i][0]-(dpatch/2)-1),np.random.randint(0+dpatch/2, shapes[i][1]-(dpatch/2)-1),np.random.randint(0+dpatch/2, shapes[i][2]-(dpatch/2)-1)))
                
            #Replace the ones that by chance are foreground voxels (not so many in tumor data)
            while any([e for e in foregroundVoxels if e in backgroundVoxels]):
                ix = [e for e in foregroundVoxels if e in backgroundVoxels]
                for index in ix:
                    newVoxel = [np.random.randint(dpatch/2, shapes[i][0]-(dpatch/2)-1),np.random.randint(dpatch/2, shapes[i][1]-(dpatch/2)-1),np.random.randint(dpatch/2, shapes[i][2]-(dpatch/2)-1)]
                    backgroundVoxels[backgroundVoxels.index(index)] = newVoxel

            #backgroundVoxels = bg[random.sample(xrange(0,len(bg)), patches_per_subject[i]/2)].tolist()
            allVoxelIndexes.append(foregroundVoxels + backgroundVoxels)
        return allVoxelIndexes'''
    
def getAllForegroundClassesVoxels(groundTruthChannel, dpatch, output_classes):
    '''Get vector of voxel coordinates for all voxel values for all freground classes'''
    "e.g. groundTruthChannel = '/home/lukas/Documents/projects/ATLASdataset/native_part2/c0011/c0011s0006t01/c0011s0006t01_LesionSmooth_Binary.nii.gz'"
    "NOTE: img in MRICRON starts at (1,1,1) and this function starts at (0,0,0), so points do not match when comparing in MRICRON. Add 1 to all dimensions to match in mricron. Function works properly though"
    img = nib.load(groundTruthChannel)
    data = np.array(img.dataobj[dpatch/2:img.shape[0]-(dpatch/2)-1, dpatch/2:img.shape[1]-(dpatch/2)-1, dpatch/2:img.shape[2]-(dpatch/2)-1],dtype='int16') # Get a cropped image, to avoid CENTRAL foreground voxels that are too near to the border. These will still be included, but not as central voxels. As long as they are in the 9x9x9 volume (-+ 4 voxels from the central, on a segment size of 25x25x25) they will still be included in the training.
    img.uncache()    
    voxels = []
    for c in range(1,output_classes):
        coords = np.argwhere(data==c)
        coords = coords + dpatch/2
        voxels.append(coords)
    #voxels = voxels + dpatch/2 # need to add this, as the cropped image starts again at (0,0,0)
    #backgroundVoxels = np.argwhere(data==0)
    return voxels#, backgroundVoxels  # This is a List! Use totuple() to convert if this makes any trouble



            
def getSubjectsToSample(channelList, subjectIndexes):
    "Actually returns channel of the subjects to sample"
    fp = open(channelList)
    lines = fp.readlines()
    subjects = [lines[i] for i in subjectIndexes]
    fp.close()
    return subjects

def extractLabels(groundTruthChannel_list, subjectIndexes, voxelCoordinates, dpatch):
    print('extracting labels from ' + str(len(subjectIndexes))+ ' subjects.')
    subjects = getSubjectsToSample(groundTruthChannel_list,subjectIndexes)
    labels = []
    if (len(subjectIndexes) > 1):
        for i in xrange(0,len(voxelCoordinates)):
            subject = str(subjects[i])[:-1]
            proxy_label = nib.load(subject)
            label_data = np.array(proxy_label.get_data(),dtype='int16')
            for j in xrange(0,len(voxelCoordinates[i])):     
                D1,D2,D3 = voxelCoordinates[i][j]
                labels.append(label_data[D1-4:D1+5,D2-4:D2+5,D3-4:D3+5])
            proxy_label.uncache()
            del label_data
        return labels
    elif(len(subjectIndexes) == 1):
        subject = str(subjects[0])[:-1]
        proxy_label = nib.load(subject)
        label_data = np.array(proxy_label.get_data(),dtype='int16')
        for i in xrange(0,len(voxelCoordinates[0])):
            D1,D2,D3 = voxelCoordinates[0][i]
            labels.append(label_data[D1-4:D1+5,D2-4:D2+5,D3-4:D3+5])
            #print("Extracted labels " + str(i))
        proxy_label.uncache()
        del label_data
        return labels
    
def get_patches_per_subject( n_patches, n_subjects):
    patches_per_subject = [n_patches/n_subjects]*n_subjects
    randomAdd = random.sample(range(0,len(patches_per_subject)),k=n_patches%n_subjects)
    randomAdd.sort()
    for index in randomAdd:
        patches_per_subject[index] = patches_per_subject[index] + 1
    return patches_per_subject

def extractImagePatch(channel, subjectIndexes, patches, voxelCoordinates, n_patches, dpatch, debug=False):
    subjects = getSubjectsToSample(channel, subjectIndexes)
    vol = np.zeros((n_patches,dpatch,dpatch,dpatch),dtype='int16')
    k = 0
    if (len(subjectIndexes) > 1):
        for i in xrange(0,len(voxelCoordinates)):
            subject = str(subjects[i])[:-1]
            proxy_img = nib.load(subject)
            
            img_data = np.array(proxy_img.get_data(),dtype='int16')

            
            #img_data = normalizeMRI(img_data)
            
            for j in xrange(0,len(voxelCoordinates[i])):     
                D1,D2,D3 = voxelCoordinates[i][j]    
                
                
                vol[k,:,:,:] = img_data[D1-(dpatch/2):D1+(dpatch/2)+1,D2-(dpatch/2):D2+(dpatch/2)+1,D3-(dpatch/2):D3+(dpatch/2)+1]
                #print(i,j)
                #print(D1,D2,D3)
                
                k=k+1
            proxy_img.uncache()
            del img_data
            if debug: print('extracted [' + str(len(voxelCoordinates[i])) + '] patches from subject ' + str(i) +'/'+ str(len(subjectIndexes)) +  ' with index [' + str(subjectIndexes[i]) + ']')
        return vol
    elif(len(subjectIndexes) == 1):
        #print("only subject " + str(subjects))
        subject = str(subjects[0])[:-1]
        proxy_img = nib.load(subject)
        img_data = np.array(proxy_img.get_data(),dtype='int16')
        
        #img_data = normalizeMRI(img_data)
        
        for i in xrange(0,len(voxelCoordinates[0])):
            
            
            D1,D2,D3 = voxelCoordinates[0][i]        
            #print(i,j)
            #print(D1,D2,D3)
            vol[k,:,:,:] = img_data[D1-(dpatch/2):D1+(dpatch/2)+1,D2-(dpatch/2):D2+(dpatch/2)+1,D3-(dpatch/2):D3+(dpatch/2)+1]  # at some point change this to be the central voxel. And change how the voxels are sampled (does not need to subtract the dpatch size)
            k=k+1
            
            if debug: print('extracted [' + str(i) + '] patches from subject ')
        proxy_img.uncache()
        del img_data
        #if debug: print('extracted [' + str(len(voxelCoordinates[i])) + '] patches from subject ' + str(i) +'/'+ str(len(subjectIndexes)) +  ' with index [' + str(subjectIndexes[i]) + ']')
    return vol

def sampleTrainData(channelsList, groundTruthChannel_list, n_patches, n_subjects, dpatch, output_classes, samplingMethod, logfile):
    '''output is a batch containing n-patches and their labels'''
    '''main function, called in the training process'''  
    num_channels = len(channelsList)
    start = time.time()
    patches_per_subject = get_patches_per_subject( n_patches, n_subjects)
    patches = np.zeros((n_patches,dpatch,dpatch,dpatch,num_channels),dtype='int8')
    labelsFile = open(groundTruthChannel_list,"r")    
    total_subjects = file_len(groundTruthChannel_list)
    labelsFile.close()    
    subjectIndexes = generateRandomIndexesSubjects(n_subjects, total_subjects) 
    shapes = getSubjectShapes(subjectIndexes, n_patches, groundTruthChannel_list)
    voxelCoordinates = generateVoxelIndexes(subjectIndexes, shapes, patches_per_subject, dpatch, n_patches, groundTruthChannel_list, samplingMethod, output_classes)

    for i in xrange(0,len(channelsList)):
        patches[:,:,:,:,i] = extractImagePatch(channelsList[i], subjectIndexes, patches, voxelCoordinates, n_patches, dpatch, debug=False)
             
    labels = np.array(extractLabels(groundTruthChannel_list, subjectIndexes, voxelCoordinates, dpatch))
    labels = to_categorical(labels.astype(int),output_classes)
    
    if(samplingMethod == 2):
        patches = patches[0:len(labels)]  # when using equal sampling (samplingMethod 2), because some classes have very few voxels in a head, there are fewer patches as intended. Patches is initialized as the maximamum value, so needs to get cut to match labels.
    end = time.time()
    my_logger("Finished extracting " + str(n_patches) + " patches, from "  + str(n_subjects) + " subjects and " + str(num_channels) + " channels. Timing: " + str(round(end-start,2)) + "s", logfile)
    return patches, labels

def generateAllForegroundVoxels(groundTruthChannel_list, dpatch):
    "Gets called once, outside whole training iterations"
    
    #Need to input subject indexes to see which channels in list to use.
    # Or, like deepMedic, create a predefined list of channels with the subjects that go into training.
    
    labelsFile = open(groundTruthChannel_list,"r")    
    total_subjects = file_len(groundTruthChannel_list)
    labelsFile.close()    
    total_subjects = total_subjects
    subjectIndexes = range(0,total_subjects)
    channels = getSubjectChannels(subjectIndexes, groundTruthChannel_list)
    allForegroundVoxels = []    
    for index in subjectIndexes:
        fg = list(getForegroundBackgroundVoxels(channels[index], dpatch))
        allForegroundVoxels.append(fg)
    return allForegroundVoxels
    
'''
def popForegroundVoxels(allForegroundVoxels, patches_per_subject):
    "To be included inside generateVoxelIndexes in samplingMethod 2. Needs input of allForegroundVoxels that has to be passed through all function wrappers..."
    # syntax: allForegroundVoxels[subject].pop(voxelIndex)
    #generate pseudorandom numbers to pop from list
    # numbers are random, but the sample size is defined. N patches from M subjects
    
    random.sample(range(0,10),2)
    
    i = subject
    
    fg_voxels = random.sample(xrange(0,len(allForegroundVoxels[i])), patches_per_subject[i]/2 + patches_per_subject[i]%2)
    
    for voxelIndex in fg_voxels:
        foregroundVoxels.append(allForegroundVoxels[i].pop(voxelIndex))
    
'''
    


def getForegroundBackgroundVoxels(groundTruthChannel, dpatch):
    '''Get vector of voxel coordinates for all voxel values > 0'''
    "e.g. groundTruthChannel = '/home/lukas/Documents/projects/ATLASdataset/native_part2/c0011/c0011s0006t01/c0011s0006t01_LesionSmooth_Binary.nii.gz'"
    "NOTE: img in MRICRON starts at (1,1,1) and this function starts at (0,0,0), so points do not match when comparing in MRICRON. Add 1 to all dimensions to match in mricron. Function works properly though"
    img = nib.load(groundTruthChannel)
    data = np.array(img.dataobj[dpatch/2:img.shape[0]-(dpatch/2)-1, dpatch/2:img.shape[1]-(dpatch/2)-1, dpatch/2:img.shape[2]-(dpatch/2)-1],dtype='int16') # Get a cropped image, to avoid CENTRAL foreground voxels that are too near to the border. These will still be included, but not as central voxels. As long as they are in the 9x9x9 volume (-+ 4 voxels from the central, on a segment size of 25x25x25) they will still be included in the training.
    img.uncache()    
    foregroundVoxels = np.argwhere(data>0)
    foregroundVoxels = foregroundVoxels + dpatch/2 # need to add this, as the cropped image starts again at (0,0,0)
    #backgroundVoxels = np.argwhere(data==0)
    return foregroundVoxels#, backgroundVoxels  # This is a List! Use totuple() to convert if this makes any trouble
    
    '''Actually even nicer: vectorize/flatten image, get start-end values of only-background / only-foreground indexes. MUCH less values to store. Then: 
    dataVector = np.reshape(data, (1, data.shape[0]*data.shape[1]*data.shape[2]))    
    indx = np.argwhere(dataVector[0,:]>0)
    from itertools import groupby
    from operator import itemgetter
    for k,g in groupby(enumerate(indx), lambda (i,x):i-x):
        print(map(itemgetter(1),g))'''
        


def totuple(a):
    "Returns tuple with tuples"
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a

def totuple2(a):
    "Returns list with tuples"
    try:
        return list(totuple(i) for i in a)
    except TypeError:
        return a


from keras import backend as K

def f1(y_true, y_pred):
    "Only makes sense in multiclass problems somehow. When binary classes, have to change average = 'binary' somewhere somehow "
    "see https://stackoverflow.com/questions/43001014/precision-recall-fscore-support-returns-same-values-for-accuracy-precision-and"
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall))

def dice(y_true, y_pred):
    # Symbolically compute the intersection
    y_int = y_true*y_pred
    # Technically this is the negative of the Sorensen-Dice index. This is done for
    # minimization purposes
    return (2*K.sum(y_int) / (K.sum(y_true) + K.sum(y_pred)))

def ytrue(y_true, y_pred):
    return(y_true)

def ypred(y_true, y_pred):
    return(y_pred)
    

def fbeta(y_true, y_pred, threshold_shift=0):
    beta = 2

    # just in case of hipster activation at the final layer
    y_pred = K.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred + threshold_shift)

    tp = K.sum(K.round(y_true * y_pred_bin), axis=1) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)), axis=1)
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)), axis=1)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    beta_squared = beta ** 2
    return K.mean((beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + K.epsilon()))

'''
def TP(y_true, y_pred, threshold_shift=0):
    beta = 2

    # just in case of hipster activation at the final layer
    y_pred = K.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred + threshold_shift)

    tp = K.sum(K.round(y_true * y_pred_bin), axis=1) + K.epsilon()
    return(tp)

def FP(y_true, y_pred, threshold_shift=0):
    beta = 2

    # just in case of hipster activation at the final layer
    y_pred = K.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred + threshold_shift)

    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)), axis=1)
    return(fp)
'''    
def sampleTestData(testChannels, testLabels, subjectIndex, output_classes, dpatch,logfile):
    "output should be a batch containing all (non-overlapping) image patches of the whole head, and the labels"
    "Actually something like sampleTraindata, thereby inputting extractImagePatch with all voxels of a subject"
    "Voxel coordinates start at index [26] and then increase by 17 in all dimensions."    
    num_channels = len(testChannels)
    labelsFile = open(testLabels,"r")   
    ch = labelsFile.readlines()
    subjectGTchannel = ch[subjectIndex[0]][:-1]
    my_logger('Segmenting subject with GT channel: ' + str(subjectGTchannel), logfile)
    labelsFile.close()      
    proxy_img = nib.load(subjectGTchannel)
    shape = proxy_img.shape
    affine = proxy_img.affine
        
    xend = shape[0]-26
    yend = shape[1]-26
    zend = shape[2]-26

    voxelCoordinates = []
    for x in range(26,xend,9):
        for y in range(26,yend,9):
            for z in range(26,zend,9):
                voxelCoordinates.append([x,y,z])
    
    n_patches = len(voxelCoordinates)
    patches = np.zeros((n_patches,dpatch,dpatch,dpatch,num_channels),dtype='int8')
    
    
    for i in xrange(0,len(testChannels)):
        patches[:,:,:,:,i] = extractImagePatch(testChannels[i], subjectIndex, patches, [voxelCoordinates], n_patches, dpatch, debug=False)
                     
    labels = np.array(extractLabels(testLabels, subjectIndex, [voxelCoordinates], dpatch))
    labels = to_categorical(labels.astype(int),output_classes)
    #print("Finished extracting " + str(n_patches) + " patches, from "  + str(n_subjects) + " subjects and " + str(num_channels) + " channels. Timing: " + str(round(end-start,2)) + "s")
    return patches, labels, voxelCoordinates, shape, affine


def classesInSample(labels, output_classes, axis=4):
    classes = np.argmax(labels, axis=axis)
    classes = classes.flatten()
    occurences = []
    for c in range(0,output_classes):
        occurences.append(list(classes).count(c))
    return occurences

def getClassProportions(classFrequencies):
    t = float(sum(classFrequencies))
    p = []
    for i in classFrequencies:
        p.append(round(i*100/t,1))
    return p

def fullHeadSegmentation(model, testChannels, testLabels, subjectIndex, output_classes, dpatch, size_minibatches,logfile, epoch, saveSegmentation = False):    
    subjectIndex = [subjectIndex]
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
    
    flairCh = getSubjectsToSample(testChannels[0], subjectIndex)
    subID = flairCh[0][-13:-8]
    "EXTRACT SUBJECT ID FROM FLAIR CHANNEL -LAST 5 DIGITS BEFORE .NII.GZ-. STORE AND USE FOR CORRECT NAMING OF SEGMENTED OUTPUT"
    
    '''here are 3 requirements for the successfull upload and validation of your segmentation:

    Use the MHA filetype to store your segmentations (not mhd) [use short or ushort if you experience any upload problems]
    Keep the same labels as the provided truth.mha (see above)
    Name your segmentations according to this template: VSD.your_description.###.mha

    replace the ### with the ID of the corresponding Flair MR images. This allows the system to relate your segmentation to the correct training truth. Download an example list for the training data and testing data.
    '''

    "STORE IMAGE AS MHA. CONVERT EXTERNALLY IF NECESSARY, using c3d from terminal, just like I did to convert to nii"
    "CAN USE SIMPLEITK TO CONVERT DATA IN PYTHON"
    "https://stackoverflow.com/questions/29738822/how-to-convert-mha-file-to-nii-file-in-python-without-using-medpy-or-c"
    
    batch, labels, voxelCoordinates, shape, affine = sampleTestData(testChannels, testLabels, subjectIndex, output_classes, dpatch,logfile)
    print("Extracted image patches for full head segmentation")
    
    start = 0
    n_minibatches = len(batch)/size_minibatches
    indexes = []
    for j in range(0,n_minibatches):
        print("training on minibatch " +str(j)+ "/" + str(n_minibatches))
        end = start + size_minibatches
        miniTestbatch = batch[start:end,:,:,:,:]    
        miniTestbatch_labels = labels[start:end,:,:,:,:]    
        
        prediction = model.predict(miniTestbatch, verbose=0)
        class_pred = np.argmax(prediction, axis=4)
        indexes.extend(class_pred)        
        #test_performance.append(model.evaluate(miniTestbatch, miniTestbatch_labels, verbose=0))
        P,N,TP,TN,FP,FN,ACC,acc,roc =  evaluation_metrics(class_pred, prediction, output_classes, miniTestbatch_labels )
        
        positives.append(P)
        negatives.append(N)
        truePositives.append(TP)
        trueNegatives.append(TN)
        falsePositives.append(FP)
        falseNegatives.append(FN)
        #sens.append(TPR)
        #spec.append(SPC)
        #Dice.append(DSC)
        accuracy.append(ACC)  # per class
        total_accuracy.append(acc)
        auc_roc.append(roc)
        start = end
        
    
    #last one
    end = start + (len(voxelCoordinates)-n_minibatches*size_minibatches)
    miniTestbatch = batch[start:end,:,:,:,:]    
    miniTestbatch_labels = labels[start:end,:,:,:,:]    
    prediction = model.predict(miniTestbatch, verbose=0)
    class_pred = np.argmax(prediction, axis=4)
    indexes.extend(class_pred)            
    #test_performance.append(model.evaluate(miniTestbatch, miniTestbatch_labels, verbose=0))
    P,N,TP,TN,FP,FN,ACC,acc,roc =  evaluation_metrics(class_pred, prediction, output_classes, miniTestbatch_labels )
    positives.append(P)
    negatives.append(N)
    truePositives.append(TP)
    trueNegatives.append(TN)
    falsePositives.append(FP)
    falseNegatives.append(FN)
    #sens.append(TPR)
    #spec.append(SPC)
    #Dice.append(DSC)
    accuracy.append(ACC)  # per class
    total_accuracy.append(acc)
    auc_roc.append(roc)

    mean_acc = np.average(accuracy, axis=0)
    mean_total_accuracy = np.average(total_accuracy, axis=0)
    mean_AUC_ROC = np.nanmean(auc_roc, axis=0)
    
    
    sumTP = np.sum(truePositives,0)
    sumTN = np.sum(trueNegatives,0)
    sumP = np.sum(positives,0)
    sumN = np.sum(negatives,0)
    sumFP = np.sum(falsePositives,0)
    sumFN = np.sum(falseNegatives,0)    

    total_sens = np.divide(np.array(sumTP,dtype='float32'),np.array(sumP,dtype='float32'))
    total_spec = np.divide(np.array(sumTN,dtype='float32'),np.array(sumN,dtype='float32'))
    total_precision = np.divide(np.array(sumTP,dtype='float32'),(np.array(sumTP,dtype='float32') + np.array(sumFP,dtype='float32')))
    total_DSC = np.divide(2*np.array(sumTP,dtype='float64'),(2 * np.array(sumTP,dtype='float64') + np.array(sumFP,dtype='float64') + np.array(sumFN,dtype='float64')))
    
    if(saveSegmentation):
    
        head = np.zeros(shape, dtype=np.int16)  # same size as input head, start index for segmentation start at 26,26,26, rest filled with zeros....
        i = 0
        for x,y,z in voxelCoordinates:
            
            head[x-4:x+5,y-4:y+5,z-4:z+5] = indexes[i]
            i = i+1

        img = nib.Nifti1Image(head, affine)
        
        segmentationName = 'VSD.' + logfile[12:] + '.' + subID
        
        nib.save(img, os.path.join('/home/lukas/Documents/projects/brainSegmentation/deepMedicKeras/Output/Predictions/' + segmentationName + '.nii.gz'))
        my_logger('Saved segmentation of subject at: ' + '/home/lukas/Documents/projects/brainSegmentation/deepMedicKeras/Output/Predictions/' + segmentationName + '.nii.gz', logfile)
    #p = p+1
    #print(subjectIndex)
    # print(test_performance[-1])
    # print("Mean test performance: " + str(np.mean(test_performance,axis=0)))
    #print('Total egmentation on subject took seconds:')
    #print(round(time.time()-t1,2))
    
    return total_sens, total_spec, total_DSC, mean_acc, mean_total_accuracy, mean_AUC_ROC, total_precision

# calculate AUC as well as another metric

'sklearn.metrics.f1_score(y_true, y_pred, labels=None, pos_label=1, average=’binary’, sample_weight=None)'
"Inpputs should be 1d arrays (just flatten them). use argument 'labels' to include all classes, in case in a batch a class is absent. Argument 'average' set to None returns scores for each class"
"Maybe use 'average = samples' too"

def evaluation_metrics(class_pred, prediction, output_classes, miniTestbatch_labels ):

    # Add classes to fullfill requisites for F1 
    #tmp = list(class_pred[-1][-1][-1][0:output_classes]) # store original values
    #class_pred[-1][-1][-1][0:output_classes] = [u for u in range(output_classes)]
    newlist = [u for u in class_pred for u in u]
    newlist = [u for u in newlist for u in u]
    newlist = [u for u in newlist for u in u]
    newlist = np.array(newlist)
    y_pred = newlist
    
    ytrueALL = (np.argmax(miniTestbatch_labels, axis=4))
    newlist = [u for u in ytrueALL for u in u]
    newlist = [u for u in newlist for u in u]
    newlist = [u for u in newlist for u in u]
    newlist = np.array(newlist)
    y_true = newlist
    
    P,N,TP,TN,FP,FN,ACC = stats(y_pred, y_true, output_classes)
    # if normalize = True, same as hamming_score. Just a summarized accuracy for all labels.
    acc = metrics.accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)
    
    # add one class to be able to compute AUC ROC
    #y_true[-output_classes:len(y_true)] = [u for u in range(output_classes)] # replace with classes.
    newlist = [u for u in prediction for u in u]
    newlist = [u for u in newlist for u in u]
    newlist = [u for u in newlist for u in u]
    newlist = np.array(newlist)
    y_score = newlist
    y_true = to_categorical(y_true.astype(int),output_classes)
    "roc_auc_score needs inputs as arrays of shape (n_samples , n_classes), y_score is the output probs of the softmax layer, and both y_true and y_score are one-hot-encoded"
    "There need to be at least one present class in the sample. Not defined if a class is never present in the sample"
    try:
        roc = metrics.roc_auc_score(y_true, y_score, average=None, sample_weight=None)
    except ValueError:
        roc = np.array([np.nan]*(output_classes))
    #coverage = metrics.coverage_error(y_true, y_score, sample_weight=None)
    #label_ranking_loss = metrics.label_ranking_loss(y_true, y_score, sample_weight=None)
    
    # restore to original values, to avoid weird grid patterns in segmentation output. As this alterns globally the outout.
    #class_pred[-1][-1][-1][0:output_classes] = tmp
    
    return P,N,TP,TN,FP,FN,ACC,acc,roc#, coverage, label_ranking_loss



def stats(y_pred, y_true, output_classes):
    P = []
    N = []
    TP = []
    TN = []
    FP = []
    FN = []
    TPR = []
    SPC = []
    DSC = []
    ACC = []

    for c in range(output_classes):
        tmp_pred = y_pred == c
        tmp_true = y_true == c
        p = sum(tmp_true)
        n = sum(np.invert(tmp_true))
        tp = sum(tmp_pred * tmp_true)
        tn = sum(np.invert(tmp_pred)*np.invert(tmp_true))
        fp = sum(tmp_pred * np.invert(tmp_true))
        fn = sum(np.invert(tmp_pred) * tmp_true)
        a = sum(tmp_true == tmp_pred)/float(len(tmp_pred))
        '''
        if p:
            tpr = tp/float(p)
        else:
            tpr = np.nan

        if n:
            spc = tn/float(n)
        else:
            spc = np.nan
        try:
            dsc = 2*tp/float(2*tp+fp+fn)
        except ZeroDivisionError:
            dsc = np.nan'''
        P.append(p)
        N.append(n)
        TP.append(tp)
        TN.append(tn)
        FP.append(fp)
        FN.append(fn)
        #TPR.append(tpr)
        #SPC.append(spc)
        #DSC.append(dsc)
        ACC.append(a)
        
        # Foreground
    tmp_pred = y_pred > 0
    tmp_true = y_true > 0
    p = sum(tmp_true)
    n = sum(np.invert(tmp_true))
    tp = sum(tmp_pred * tmp_true)
    tn = sum(np.invert(tmp_pred)*np.invert(tmp_true))
    fp = sum(tmp_pred * np.invert(tmp_true))
    fn = sum(np.invert(tmp_pred) * tmp_true)
    a = sum(tmp_true == tmp_pred)/float(len(tmp_pred))
    '''
    if p:
        tpr = tp/float(p)
    else:
        tpr = np.nan
    if n:
        spc = tn/float(n)
    else:
        spc = np.nan
    try:
        dsc = 2*tp/float(2*tp+fp+fn)
    except ZeroDivisionError:
        dsc = np.nan
    '''
    P.append(p)
    N.append(n)
    TP.append(tp)
    TN.append(tn)
    FP.append(fp)
    FN.append(fn)
    #TPR.append(tpr)
    #SPC.append(spc)
    #DSC.append(dsc)
    ACC.append(a)
    
    return P,N,TP,TN,FP,FN,ACC#TPR,SPC,DSC,ACC
    

def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    '''
    Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
    https://stackoverflow.com/q/32239577/395857
    '''
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        #print('\nset_true: {0}'.format(set_true))
        #print('set_pred: {0}'.format(set_pred))
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float( len(set_true.union(set_pred)) )
        #print('tmp_a: {0}'.format(tmp_a))
        acc_list.append(tmp_a)
    return np.mean(acc_list)


def my_logger(string, logfile):
    f = open(logfile,'a')
    f.write('\n' + str(string))
    f.close()
    print(string)
    
def movingAverageConv(a, window_size=1) :
    if not a : return a
    window = np.ones(int(window_size))
    result = np.convolve(a, window, 'full')[ : len(a)] # Convolve full returns array of shape ( M + N - 1 ).
    slotsWithIncompleteConvolution = min(len(a), window_size-1)
    result[slotsWithIncompleteConvolution:] = result[slotsWithIncompleteConvolution:]/float(window_size)
    if slotsWithIncompleteConvolution > 1 :
        divisorArr = np.asarray(range(1, slotsWithIncompleteConvolution+1, 1), dtype=float)
        result[ : slotsWithIncompleteConvolution] = result[ : slotsWithIncompleteConvolution] / divisorArr
    return result
        
        
def dice_completeImages(img1,img2):
    return(2*np.sum(np.multiply(img1>0,img2>0))/float(np.sum(img1>0)+np.sum(img2>0)))

def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


def normalizeMRI(data):
    #img = nib.load('/home/lukas/Documents/projects/public_SegmentationData/BRATS2015_Training/HGG/brats_2013_pat0001_1/VSD.Brain.XX.O.MR_Flair.54512/VSD.Brain.XX.O.MR_Flair.54512.nii')
    #data = img.get_data()
    data1 = np.ma.masked_array(data, data==0)
    m = data1.mean()
    s = data1.std()
    data1 = (data1 - m)/s
    data1 = np.ma.getdata(data1)
    return(data1)
    
    
    
def sampleTestData2(testChannels, testLabels, subjectIndex, output_classes, dpatch,logfile):
    "output should be a batch containing all (non-overlapping) image patches of the whole head, and the labels"
    "Actually something like sampleTraindata, thereby inputting extractImagePatch with all voxels of a subject"
    "Voxel coordinates start at index [26] and then increase by 17 in all dimensions."    
    
    if len(testLabels) == 0:
        
        num_channels = len(testChannels)
        labelsFile = open(testChannels[0],"r")   
        ch = labelsFile.readlines()
        subjectGTchannel = ch[subjectIndex[0]][:-1]
        my_logger('Segmenting subject with FLAIR channel: ' + str(subjectGTchannel), logfile)
        labelsFile.close()      
        proxy_img = nib.load(subjectGTchannel)
        shape = proxy_img.shape
        affine = proxy_img.affine
            
        xend = shape[0]-26
        yend = shape[1]-26
        zend = shape[2]-26
    
        voxelCoordinates = []
        for x in range(26,xend,9):
            for y in range(26,yend,9):
                for z in range(26,zend,9):
                    voxelCoordinates.append([x,y,z])
        
        n_patches = len(voxelCoordinates)
        patches = np.zeros((n_patches,dpatch,dpatch,dpatch,num_channels),dtype='int8')
        
        
        for i in xrange(0,len(testChannels)):
            patches[:,:,:,:,i] = extractImagePatch(testChannels[i], subjectIndex, patches, [voxelCoordinates], n_patches, dpatch, debug=False)
                         
        labels = []
        #print("Finished extracting " + str(n_patches) + " patches, from "  + str(n_subjects) + " subjects and " + str(num_channels) + " channels. Timing: " + str(round(end-start,2)) + "s")
        return patches, labels, voxelCoordinates, shape, affine
        
    
    
    else:
        num_channels = len(testChannels)
        labelsFile = open(testChannels[0],"r")   
        ch = labelsFile.readlines()
        subjectGTchannel = ch[subjectIndex[0]][:-1]
        my_logger('Segmenting subject with FLAIR channel: ' + str(subjectGTchannel), logfile)
        labelsFile.close()      
        proxy_img = nib.load(subjectGTchannel)
        shape = proxy_img.shape
        affine = proxy_img.affine
            
        xend = shape[0]-26
        yend = shape[1]-26
        zend = shape[2]-26
    
        voxelCoordinates = []
        for x in range(26,xend,9):
            for y in range(26,yend,9):
                for z in range(26,zend,9):
                    voxelCoordinates.append([x,y,z])
        
        n_patches = len(voxelCoordinates)
        patches = np.zeros((n_patches,dpatch,dpatch,dpatch,num_channels),dtype='int8')
        
        
        for i in xrange(0,len(testChannels)):
            patches[:,:,:,:,i] = extractImagePatch(testChannels[i], subjectIndex, patches, [voxelCoordinates], n_patches, dpatch, debug=False)
                         
        
        labels = []
        #print("Finished extracting " + str(n_patches) + " patches, from "  + str(n_subjects) + " subjects and " + str(num_channels) + " channels. Timing: " + str(round(end-start,2)) + "s")
        return patches, labels, voxelCoordinates, shape, affine


def fullHeadSegmentation2(dice_compare, dsc, model, testChannels, testLabels, subjectIndex, output_classes, dpatch, size_minibatches,logfile, epoch, saveSegmentation = False):    
    subjectIndex = [subjectIndex]
    
    
    flairCh = getSubjectsToSample(testChannels[0], subjectIndex)
    subID = flairCh[0][-13:-8]
    "EXTRACT SUBJECT ID FROM FLAIR CHANNEL -LAST 5 DIGITS BEFORE .NII.GZ-. STORE AND USE FOR CORRECT NAMING OF SEGMENTED OUTPUT"
    
    '''here are 3 requirements for the successfull upload and validation of your segmentation:

    Use the MHA filetype to store your segmentations (not mhd) [use short or ushort if you experience any upload problems]
    Keep the same labels as the provided truth.mha (see above)
    Name your segmentations according to this template: VSD.your_description.###.mha

    replace the ### with the ID of the corresponding Flair MR images. This allows the system to relate your segmentation to the correct training truth. Download an example list for the training data and testing data.
    '''

    "STORE IMAGE AS MHA. CONVERT EXTERNALLY IF NECESSARY, using c3d from terminal, just like I did to convert to nii"
    "CAN USE SIMPLEITK TO CONVERT DATA IN PYTHON"
    "https://stackoverflow.com/questions/29738822/how-to-convert-mha-file-to-nii-file-in-python-without-using-medpy-or-c"
    
    batch, labels, voxelCoordinates, shape, affine = sampleTestData2(testChannels, testLabels, subjectIndex, output_classes, dpatch,logfile)
    print("Extracted image patches for full head segmentation")
    
    start = 0
    n_minibatches = len(batch)/size_minibatches
    indexes = []
    for j in range(0,n_minibatches):
        print("training on minibatch " +str(j)+ "/" + str(n_minibatches))
        end = start + size_minibatches
        miniTestbatch = batch[start:end,:,:,:,:]    
                
        prediction = model.predict(miniTestbatch, verbose=0)
        class_pred = np.argmax(prediction, axis=4)
        indexes.extend(class_pred)        

        start = end
        
    
    #last one
    end = start + (len(voxelCoordinates)-n_minibatches*size_minibatches)
    miniTestbatch = batch[start:end,:,:,:,:]    
    
    prediction = model.predict(miniTestbatch, verbose=0)
    class_pred = np.argmax(prediction, axis=4)
    indexes.extend(class_pred)            
    #test_performance.append(model.evaluate(miniTestbatch, miniTestbatch_labels, verbose=0))
    
    
    if(saveSegmentation):
    
        head = np.zeros(shape, dtype=np.int16)  # same size as input head, start index for segmentation start at 26,26,26, rest filled with zeros....
        i = 0
        for x,y,z in voxelCoordinates:
            
            head[x-4:x+5,y-4:y+5,z-4:z+5] = indexes[i]
            i = i+1

        img = nib.Nifti1Image(head, affine)
        
        if dice_compare:
            
            labelsFile = open(testLabels,"r")   
            ch = labelsFile.readlines()
            subjectGTchannel = ch[subjectIndex[0]][:-1]
            GT = nib.load(subjectGTchannel)  
            dsc.append(dice_completeImages(img.get_data(), GT.get_data()))
            print(dsc[-1])
            print('mean DCS so far:' + str(np.mean(np.round(dsc,2))))
        segmentationName = 'VSD.' + logfile[12:] + '.' + subID
        
        nib.save(img, '/home/lukas/Documents/projects/brainSegmentation/deepMedicKeras/Output/Predictions/' + segmentationName + '.nii.gz')
        
        
        #out = '/home/lukas/Documents/projects/brainSegmentation/deepMedicKeras/Output/Predictions/' + segmentationName + '.mha'
        #hdr = img.header
        #data = img.get_data()
        #medpy.io.save(data, out, hdr )
        
        
        my_logger('Saved segmentation of subject at: ' + '/home/lukas/Documents/projects/brainSegmentation/deepMedicKeras/Output/Predictions/' + segmentationName + '.nii.gz', logfile)
    #p = p+1
    #print(subjectIndex)
    # print(test_performance[-1])
    # print("Mean test performance: " + str(np.mean(test_performance,axis=0)))
    #print('Total egmentation on subject took seconds:')
    #print(round(time.time()-t1,2))
        
