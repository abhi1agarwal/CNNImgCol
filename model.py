import keras
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing import image
from keras.engine import Layer
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.layers import Conv2D, UpSampling2D, InputLayer, Conv2DTranspose, Input, Reshape, merge, concatenate, Activation, Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard 
from keras.models import Sequential, Model
from keras.layers.core import RepeatVector, Permute
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from keras.callbacks import ModelCheckpoint
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
import os
import random
import tensorflow as tf
import constants

def conv_stack(data, filters, s):
        output = Conv2D(filters, (3, 3), strides=s, activation='relu', padding='same')(data)
        #output = BatchNormalization()(output)
        return output



def getmodel():
	# 	
	embed_input = Input(shape=(1000,))


	#Encoder
	encoder_input = Input(shape=(256, 256, 1,))
	encoder_output = conv_stack(encoder_input, 64, 2)
	encoder_output = conv_stack(encoder_output, 128, 1)
	encoder_output = conv_stack(encoder_output, 128, 2)
	encoder_output = conv_stack(encoder_output, 256, 1)
	encoder_output = conv_stack(encoder_output, 256, 2)
	encoder_output = conv_stack(encoder_output, 512, 1)
	encoder_output = conv_stack(encoder_output, 512, 1)
	encoder_output = conv_stack(encoder_output, 256, 1)

	#Fusion
	fusion_output = RepeatVector(32 * 32)(embed_input) 
	fusion_output = Reshape(([32, 32, 1000]))(fusion_output)
	fusion_output = concatenate([fusion_output, encoder_output], axis=3) 
	fusion_output = Conv2D(256, (1, 1), activation='relu')(fusion_output) 

	#Decoder
	decoder_output = conv_stack(fusion_output, 128, 1)
	decoder_output = UpSampling2D((2, 2))(decoder_output)
	decoder_output = conv_stack(decoder_output, 64, 1)
	decoder_output = UpSampling2D((2, 2))(decoder_output)
	decoder_output = conv_stack(decoder_output, 32, 1)
	decoder_output = conv_stack(decoder_output, 16, 1)
	decoder_output = Conv2D(2, (2, 2), activation='tanh', padding='same')(decoder_output)
	decoder_output = UpSampling2D((2, 2))(decoder_output)

	model = Model(inputs=[encoder_input, embed_input], outputs=decoder_output)
	return model

