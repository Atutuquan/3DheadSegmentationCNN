#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 11:00:55 2018

@author: lukas
"""
from model import DeepMedic

from configFile import wd, dpatch, L2, path_to_model, num_channels, output_classes, dropout, learning_rate, optimizer_decay

dm = DeepMedic(dpatch, output_classes, num_channels, L2, dropout, learning_rate, optimizer_decay)
model = dm.createModel()
#plot_model(model, to_file='multichannel.png', show_shapes=True)
model.save(path_to_model)