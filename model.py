#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 16:01:09 2020

We are using a convoluted model (CNN) basically following this paper:
    
https://www.sciencedirect.com/science/article/pii/S0003267019307342?via%3Dihub

@author: tn438
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPool2D, BatchNormalization, Dropout




class EVCNN(Model):
    
    def __init__(self):
        super(EVCNN, self).__init__()
    
        self.conv = Conv1D()
    
    pass

