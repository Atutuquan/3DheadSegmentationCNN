#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 10:26:01 2017

@author: lukas
"""
from keras.models import Model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.convolutional import Conv3D
from keras.initializers import he_normal
from keras.initializers import Orthogonal
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1_l2
from keras.layers import Input, Flatten, Reshape, Permute
from keras.layers.merge import Concatenate
from keras.layers import MaxPooling3D
from keras.layers.convolutional import Cropping3D
from keras.layers import UpSampling3D
from keras.layers import concatenate
from keras.layers.advanced_activations import PReLU
from keras.utils import print_summary
from keras import regularizers
from keras.optimizers import RMSprop



class DeepMedic():
    
    def __init__(self, dpatch, output_classes, num_channels, L2, dropout, learning_rate, optimizer_decay):
        
        self.dpatch = dpatch
        self.output_classes = output_classes
        self.conv_features = [30, 30, 40, 40, 40, 40, 50, 50]
        self.fc_features = [150,150, output_classes]
        self.d_factor = 3  # downsampling factor = stride in downsampling pathway
        self.num_channels = num_channels
        self.L2 = L2
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.optimizer_decay = optimizer_decay
        #self.w_initializer=w_initializer, # initialization of layer parameters? Needed here?
        #self.w_regularizer=w_regularizer,
        #self.b_initializer=b_initializer, # initialization of bias parameters? Needed here?
        #self.b_regularizer=b_regularizer,
        #self.acti_func=acti_func
    
    def createModel(self):
        '''Creates model architecture
        Input: Data input dimensions, eventually architecture specifications parsed from a config file? (activations, costFunction, hyperparameters (nr layers), dropout....)
        Output: Keras Model'''
    
        #seed = 1337
    
        mod1      = Input((self.dpatch,self.dpatch,self.dpatch, self.num_channels))
        
        #############   Normal pathway   ##################  
        
        x1        = Cropping3D(cropping = ((13,13),(13,13),(13,13)), input_shape=(self.dpatch,self.dpatch,self.dpatch, self.num_channels))(mod1)
        x1        = Conv3D(filters = self.conv_features[0], 
                           kernel_size = (3,3,3), 
                           #kernel_initializer=he_normal(seed=seed),
                           kernel_initializer=Orthogonal(),
                           kernel_regularizer=regularizers.l2(self.L2))(x1)
        x1        = BatchNormalization()(x1)
        #x1        = Activation('relu')(x1)
        x1        = PReLU()(x1)
        #x1        = BatchNormalization()(x1)
        
        for feature in self.conv_features[1:]:  
            x1        = Conv3D(filters = feature, 
                               kernel_size = (3,3,3), 
                               #kernel_initializer=he_normal(seed=seed),
                               kernel_initializer=Orthogonal(),
                               kernel_regularizer=regularizers.l2(self.L2))(x1)
            x1        = BatchNormalization()(x1)
            #x1        = Activation('relu')(x1)
            x1        = PReLU()(x1)
            #x1        = BatchNormalization()(x1)
            
        #############   Downsampled pathway   ##################   
        
        x2        = MaxPooling3D(pool_size=(self.d_factor,self.d_factor,self.d_factor), padding="same")(mod1)
        x2        = Conv3D(filters = self.conv_features[0], 
                           kernel_size = (3,3,3), 
                           #kernel_initializer=he_normal(seed=seed),
                           kernel_initializer=Orthogonal(),
                           kernel_regularizer=regularizers.l2(self.L2))(x2)
        x2        = BatchNormalization()(x2)
        #x2        = Activation('relu')(x2)
        x2        = PReLU()(x2)
        #x2        = BatchNormalization()(x2)
        
        for feature in self.conv_features[1:]:    
            x2        = Conv3D(filters = feature, 
                               kernel_size = (3,3,3), 
                               #kernel_initializer=he_normal(seed=seed),
                               kernel_initializer=Orthogonal(),
                               kernel_regularizer=regularizers.l2(self.L2))(x2)
            x2        = BatchNormalization()(x2)
            #x2        = Activation('relu')(x2)
            x2        = PReLU()(x2)
            #x2        = BatchNormalization()(x2)
        
        x2        = UpSampling3D(size=(9,9,9))(x2)
        
        #############   Fully connected layers   ################## 
        
        x        = concatenate([x1,x2])
        #x        = Reshape(target_shape = (1, 60))(x)   # do I need this?
        '''x        = Flatten()(x)
        x        = Dense(units = self.fc_features[0], activation = 'elu')(x)
        x        = Dropout(rate = 0.5)(x)
        x        = Dense(units = self.fc_features[1], activation = 'elu')(x)
        x        = Dropout(rate = 0.5)(x)    
        x        = Dense(units = self.fc_features[2], activation = 'softmax', name = 'softmax')(x)'''
        
        #   Fully convolutional variant
        
        #x        = Dropout(rate = self.dropout[0])(x)
        x        = Conv3D(filters = self.fc_features[0], 
                           kernel_size = (1,1,1), 
                           #kernel_initializer=he_normal(seed=seed),
                           kernel_initializer=Orthogonal(),
                           kernel_regularizer=regularizers.l2(self.L2))(x)
        x        = BatchNormalization()(x)
        #x        = Activation('relu')(x)
        x        = PReLU()(x)
        #x        = BatchNormalization()(x)
        #x        = Dropout(rate = self.dropout[0])(x)
        x        = Conv3D(filters = self.fc_features[1], 
                           kernel_size = (1,1,1), 
                           #kernel_initializer=he_normal(seed=seed),
                           kernel_initializer=Orthogonal(),
                           kernel_regularizer=regularizers.l2(self.L2))(x)
        x        = BatchNormalization()(x)
        #x        = Activation('relu')(x)
        x        = PReLU()(x)
        #x        = BatchNormalization()(x)
        #x        = Dropout(rate = self.dropout[1])(x)
        #x        = Flatten()(x)
        x        = Dense(units = self.fc_features[2], activation = 'softmax', name = 'softmax')(x)
        
        model     = Model(inputs = mod1, outputs = x)
        #print_summary(model, positions=[.33, .6, .67,1])
                  
        rmsprop = RMSprop(lr=self.learning_rate, rho=0.9, epsilon=1e-8, decay=self.optimizer_decay)
        
        model.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=['accuracy'])
        
        return model
