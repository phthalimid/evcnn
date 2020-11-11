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
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPool1D, BatchNormalization, Dropout


FILTERS = 32
KERNEL_SIZE = 3
STRIDE = 3
POOL_SIZE = 2
DENSE_SIZE = 128

class EVCNN(Model):
    
    def __init__(self):
        super(EVCNN, self).__init__()
    
        self.conv1 = Conv1D(FILTERS, KERNEL_SIZE, strides=STRIDE, activation='relu')
        self.conv2 = Conv1D(FILTERS, KERNEL_SIZE, strides=STRIDE, activation='relu')
        self.conv3 = Conv1D(FILTERS, KERNEL_SIZE, strides=STRIDE, activation='relu')
        self.pool = MaxPool1D(POOL_SIZE)
        self.batchnorm = BatchNormalization()
        self.dropout = Dropout(rate=0.4)      
        self.flatten = Flatten()
        self.dense = Dense(DENSE_SIZE, activation='relu')
        self.softmax = Dense(10, activation='softmax')

        
    def call(self, x, train=False):

        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.batchnorm(x)
        
        if train:
            x = self.dropout(x, training=train)

        x = self.flatten(x)
        x = self.dense(x)
        x = self.softmax(x)

        return x        
    
