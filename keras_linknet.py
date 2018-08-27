
# coding: utf-8

# # Training Using Tensorflow: LinkNet-Like CNN (Keras)
# # Model Helper Functions

from keras.layers import Add, Conv2D, MaxPooling2D, Input, Cropping2D, ZeroPadding2D, BatchNormalization, Activation, Conv2DTranspose

# ### Initial Block

def initial_block(input_image):
    x = Conv2D(64, kernel_size=(7,7), strides=(2,2), activation=None, padding='same')(input_image)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2,2), strides=(2,2))(x)
    
    return x


# ### Encoding Block (second half)

def identity_block(X_in):
    m = X_in.get_shape().as_list()[-1]
    
    X_out = BatchNormalization()(X_in)
    X_out = Activation('relu')(X_out)
    X_out = Conv2D(m, kernel_size=(3,3), strides=(1,1), activation=None, padding='same')(X_out)
    
    X_out = BatchNormalization()(X_out)
    X_out = Activation('relu')(X_out)
    X_out = Conv2D(m, kernel_size=(3,3), strides=(1,1), activation=None, padding='same')(X_out)
    
    X_out = Add()([X_out,X_in])
    
    return X_out


# ### Encoding Block (first half)

def convolution_block(X_in, s):
    m = X_in.get_shape().as_list()[-1]
    
    X_out = BatchNormalization()(X_in)
    X_out = Activation('relu')(X_out)
    X_out = Conv2D(2*m, kernel_size=(3,3), strides=(s,s), activation=None, padding='same')(X_out)
    
    X_out = BatchNormalization()(X_out)
    X_out = Activation('relu')(X_out)
    X_out = Conv2D(2*m, kernel_size=(3,3), strides=(1,1), activation=None, padding='same')(X_out)
    
    X_shortcut = BatchNormalization()(X_in)
    X_shortcut = Activation('relu')(X_shortcut)
    X_shortcut = Conv2D(2*m, kernel_size=(3,3), strides=(s,s), activation=None, padding='same')(X_shortcut)
    
    X_out = Add()([X_out,X_shortcut])
    
    return X_out


# ### Encoding Block (altogether)

def encoder_block(X_in, s):
    conv = convolution_block(X_in, s)
    ident = identity_block(conv)
    
    return ident

# ### Decoding Block

def decoder_block(X_in,s=2):
    m = X_in.get_shape().as_list()[-1]
    
    transition = BatchNormalization()(X_in)
    transition = Activation('relu')(transition)
    
    conv = Conv2D(m//4, kernel_size=(1,1), strides=(1,1), activation=None, padding='same')(transition)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    
    full_conv = Conv2DTranspose(m//4, kernel_size=(3,3), strides=(s,s), activation=None, padding='same')(conv)
    full_conv = BatchNormalization()(full_conv)
    full_conv = Activation('relu')(full_conv)
    
    conv = Conv2D(m//2, kernel_size=(1,1), strides=(1,1), activation=None, padding='same')(full_conv)
    
    return conv

# ### Overall Network
    
def linknet(numpy_inputs,classifiers=1):
    initial = Input(shape=(numpy_inputs.shape[1],numpy_inputs.shape[2],numpy_inputs.shape[3]))
    # If image resolution is small, zero pad; if image resolution is big, crop to 256x256
    if numpy_inputs.shape[1] < 256:
        diff = (256-numpy_inputs.shape[1])//2
        inputs = ZeroPadding2D(padding=(diff,diff))(initial)
    elif numpy_inputs.shape[1] > 256:
        diff = (numpy_inputs.shape[1]-256)//2
        inputs = Cropping2D(cropping=(diff,diff))(initial)
    # Initial Block
    out = initial_block(inputs)
    # Encoding Sequence
    encode1 = encoder_block(out,s=1)
    encode2 = encoder_block(encode1,s=2)
    encode3 = encoder_block(encode2,s=2)
    encode4 = encoder_block(encode3,s=2)
    # Decoding Sequence
    decode4a = decoder_block(encode4,s=2)
    decode4b = Add()([decode4a,encode3])
    decode3a = decoder_block(decode4b,s=2)
    decode3b = Add()([decode3a,encode2])
    decode2a = decoder_block(decode3b,s=2)
    decode2b = Add()([decode2a,encode1])
    decode1 = decoder_block(decode2b,s=1)
    # Batch Normalization and Activation of final decoding block
    transition = BatchNormalization()(decode1)
    transition = Activation('relu')(transition)
    # Final Block
    full_conv = Conv2DTranspose(32, kernel_size=(3,3), strides=(2,2), activation=None, padding='same')(decode1)
    full_conv = BatchNormalization()(full_conv)
    full_conv = Activation('relu')(full_conv)
    
    conv = Conv2D(32, kernel_size=(3,3), strides=(1,1), activation=None, padding='same')(full_conv)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    
    segmentation = Conv2DTranspose(classifiers, kernel_size=(2,2), strides=(2,2), activation='sigmoid', padding='same')(conv)
    # Crop or Expand back to original size
    if numpy_inputs.shape[1] < 256:
        segmentation = Cropping2D(cropping=(diff,diff))(segmentation)
    elif numpy_inputs.shape[1] > 256:
        segmentation = ZeroPadding2D(padding=(diff,diff))(segmentation)
    
    return initial, segmentation