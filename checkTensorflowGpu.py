#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 11:22:55 2017

@author: lukas
"""

'''Check if Keras is using GPU version of tensorflow.
    Output should list GPU in devices. Unless one specifies cpu usage with tf.device("..") tensorflow automatically selects the gpu'''

from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
