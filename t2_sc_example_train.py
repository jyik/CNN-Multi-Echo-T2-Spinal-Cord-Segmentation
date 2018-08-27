#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 09:43:18 2018

@author: Teddy
"""

"""
# Define augmentation parameters
data_gen_args = dict(rotation_range=180.,
                     width_shift_range=0.2,
                     height_shift_range=0.2,
                     zoom_range=0.2,
                     horizontal_flip=True,
                     vertical_flip=True)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

# Provide the same seed and keyword arguments
seed = 1
image_datagen.fit(trainingx, augment=True, seed=seed)
mask_datagen.fit(trainingy, augment=True, seed=seed)

# Make training data augmentation generator
image_generator = image_datagen.flow(trainingx, batch_size = 16, seed=seed)
mask_generator = mask_datagen.flow(trainingy, batch_size = 16, seed=seed)
train_generator = zip(image_generator, mask_generator)

# Compile model and define training parameters
sc_model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
earlystop = EarlyStopping(monitor='acc', min_delta=0.001, patience=3, mode='auto')
checkpoint = ModelCheckpoint('T2_linknet_loss{loss:.3f}_acc{acc:.4f}.hdf5', monitor='acc', save_best_only=True, period=5)
callbacks_list = [earlystop, checkpoint]

# Train model
model_history = t2model.fit_generator(train_generator, epochs=10, steps_per_epoch=100, callbacks=callbacks_list)
"""