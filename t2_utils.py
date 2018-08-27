#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 09:50:20 2018

@author: Teddy
"""

import numpy as np
import nibabel as nib

def load_data(filename):
    # Files must be bet extracted already and placed in folder /data/sample_filename/
    t2img = nib.load('./data/'+filename+'/GRASE_bet.nii.gz')
    atlasimg = nib.load('./data/'+filename+'/T2_bet_regseg.nii.gz')
    t2bet = t2img.get_data()
    atlasdata = atlasimg.get_data()

    # Divide by max of each channel for every slice
    MAX = np.amax(t2bet,axis=(0,1))
    MAX[MAX==0] = 1
    t2bet = t2bet / MAX[np.newaxis,np.newaxis]
    
    return t2bet.swapaxes(0,-2), np.expand_dims(atlasdata.swapaxes(0,-1), axis=-1)

def get_dataset(filenames):
    # Load more than one sample. Makes it easier
    trainingX, trainingY = load_data(filenames[0])
    filenames.pop(0)
    for filename in filenames:
        t1data, atlasdata = load_data(filename)
        trainingX = np.append(trainingX, t1data, axis=0)
        trainingY = np.append(trainingY, atlasdata, axis=0)
        
    return trainingX, trainingY