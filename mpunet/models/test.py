import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow import nn
from tensorflow.keras import regularizers
from mpunet.models.layers import unetConv2
from tensorflow.keras.layers import Input, BatchNormalization, Cropping2D, \
                                    Concatenate, Conv2D, MaxPooling2D, \
                                    UpSampling2D, Reshape
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

inputs=mpimg.imread('image.jpeg')
inputs=np.expand_dims(inputs, axis=3)
kernel_size=3
activation='relu'
padding='same'
kernel_reg=None
def unetConv2(inputs, filters):
        if True:
                conv = Conv2D(filters, kernel_size, activation=activation,
                          padding=padding, kernel_regularizer=kernel_reg)(inputs)
                #conv = BatchNormalization()(conv)
                conv = Conv2D(filters, kernel_size, activation=activation,
                          padding=padding, kernel_regularizer=kernel_reg)(conv)
                conv = BatchNormalization()(conv)
        return conv
filters = [64, 128, 256, 512, 1024]
h1 = unetConv2(inputs, filters[0])  # h1->320*320*64
        
h2 = MaxPooling2D(pool_size=(2, 2))(h1)
h2 = unetConv2(h2, filters[1])  # h2->160*160*128

h3 = MaxPooling2D(pool_size=(2, 2))(h2)
h3 = unetConv2(h3, filters[2])   # h3->80*80*256

h4 = MaxPooling2D(pool_size=(2, 2))(h3)
h4 = unetConv2(h4, filters[3])  # h4->40*40*512

h5 = MaxPooling2D(pool_size=(2, 2))(h4)
hd5 = unetConv2(h5, filters[4])  # h5->20*20*1024
print('hd5 calculated')
#print('encoders value are=', h1, h2, h3, h4, hd5)
          
    	
