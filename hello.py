# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 03:31:06 2018

@author: zfang
"""

import numpy as np
import os.path

num_init = 50
DH = 5

Yin = np.load(r'C:\Users\zfang\Downloads\data\imtrain_noisy_[1, 7].npy')[:100]
Yout = np.load(r'C:\Users\zfang\Downloads\data\labtrain_noisy_[1, 7].npy')[:100]

Xh = np.zeros((Yin.shape[0], DH), dtype = np.float64)
Xout = np.zeros((Yout.shape[0], 2), dtype = np.float64)

def sigmoid(x, W, b):
    linpart = np.dot(W, x) + b
    return 1.0 / (1.0 + np.exp(-linpart))

a = np.array([[1,2],[3,4]])
b = np.array([1,1])
#print(1.0 / (1.0 + np.exp(a)))

c= np.matmul(a,b) + 1