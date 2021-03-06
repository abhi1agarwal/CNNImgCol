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
from pre import create_inception_embedding
from model import getmodel
from loadmodel import getmodelback
import constants

def evaluate(model):
	#Make predictions on validation images
	color_me = []
	for filename in os.listdir(constants.TEST_DIR):
	    color_me.append(img_to_array(load_img(os.path.join(constants.TEST_DIR,filename))))
	color_me = np.array(color_me, dtype=float)
	color_me_embed = create_inception_embedding(color_me)
	color_me = rgb2lab(1.0/255*color_me)[:,:,:,0]
	color_me = color_me.reshape(color_me.shape+(1,))


	# Test model
	output = model.predict([color_me, color_me_embed])
	output = output * 128

	# Output colorizations
	for i in range(len(output)):
	    cur = np.zeros((256, 256, 3))
	    cur[:,:,0] = color_me[i][:,:,0]
	    cur[:,:,1:] = output[i]
	    imsave(os.path.join(constants.OUTPUT_DIR,str(i)+".jpg"), lab2rgb(cur))


model = getmodelback(constants.JSON_FILE,constants.H5_FILE)
evaluate(model)
