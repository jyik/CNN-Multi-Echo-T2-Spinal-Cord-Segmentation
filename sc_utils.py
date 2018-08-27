#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 09:06:06 2018

@author: Teddy
"""

import os
import numpy as np
import nibabel as nib
import pickle
from keras.models import Model
from keras_linknet import linknet
from skimage.transform import rescale
from linknet_model import linknet

def load_data(filename):
    # FFE and centerline file must be in folder data/sample_filename/
    ffeimg = nib.load('./data/'+filename+'/FFE.nii.gz')
    centerline = nib.load('./data/'+filename+'/FFE_centerline_optic.nii.gz')
    ffedata = ffeimg.get_data()
    centerdata = centerline.get_data()
    centerdata = np.sort(np.nonzero(centerdata), axis=0)
    
    # Divide by max
    MAX = np.amax(ffedata)
    ffedata = ffedata / MAX
    crops = [((192-(256-centerdata[1,i])),(192-(256-centerdata[2,i]))) for i in range(16)]
    crops.reverse()
    
    return np.expand_dims(ffedata.swapaxes(0,-1),axis=-1), np.array(crops)

def get_dataset(filenames):
    # Combines each sample into a single data numpy array
    imgs, crops = load_data(filenames[0])
    filenames.pop(0)
    for filename in filenames:
        img1, crop1 = load_data(filename)
        imgs = np.append(imgs, img1, axis=0)
        crops = np.append(crops, crop1, axis=0)
    
    return imgs, crops

def cropping(imgs,crops):
    # Crops using indexing
    for i in range(len(crops)):
        imgs[i,:,:,0] = rescale(imgs[i,crops[i,0]:crops[i,0]+128,crops[i,1]:crops[i,1]+128,0],4.)
    return imgs

def import_crop(filenames):
    # Imports data and crops all-in-one
    if len(filenames) == 1:
        imgs, crops = load_data(filenames[0])
    elif len(filenames) > 1:
        imgs, crops = get_dataset(filenames)
        
    cropped_imgs = cropping(imgs, crops)
    # Save temperary file with cropping indexes
    with open('temp_crop.pkl', 'wb') as f:
        pickle.dump(crops,f)
    
    return cropped_imgs

def nn_seg(cropped_imgs, style='prob', weights='midtrain_sc_crop_weights.h5'):
    # Get prediction by creating model from function 'linknet'
    initial, segmentation = linknet(cropped_imgs)
    sc_model = Model(initial,segmentation)
    sc_model.load_weights(weights)
    pred = sc_model.predict(cropped_imgs)
    pred_bin = np.around(pred)
    if style.lower() == 'prob':
        output = pred
    elif style.lower() == 'bin':
        output = pred_bin
        
    return output

def downsampling(nn_outputs):
    # Resize prediction back into original size
    resize_outputs = np.zeros(nn_outputs.shape)
    with open('temp_crop.pkl', 'rb') as f:
        crops = pickle.load(f)
    if os.path.isfile('temp_crop.pkl'):
        os.remove('temp_crop.pkl')
    else:
        print("Error: %s file not found" % 'temp_crop.pkl')
    for i in range(len(nn_outputs)):
        resize_outputs[i,crops[i,0]:crops[i,0]+128,crops[i,1]:crops[i,1]+128,0] = rescale(nn_outputs[i,:,:,0], 1./4.)
    
    return resize_outputs

def run_total(filename,style='prob'):
    # Do the entire process with one function
    ffeimg = nib.load('./data/'+filename+'/FFE.nii.gz')
    cropped_imgs = import_crop([filename])
    nn_output = nn_seg(cropped_imgs, style)
    final_out = downsampling(nn_output)
    if style.lower() == 'prob':
        final_out[final_out < 0.01] = 0
    pred = nib.Nifti1Image(np.squeeze(final_out.swapaxes(0,-2)),ffeimg.affine)
    pred.to_filename('nn_pred2.nii.gz')
    
    return final_out
