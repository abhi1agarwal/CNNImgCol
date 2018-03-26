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
from resizeimage import resizeimage
from PIL import Image

def resize_them(path,outpath,size):
	if not os.path.isdir(path):
		raise Exception('Path doesnt exist')
	if not os.path.isdir(outpath):
		raise Exception('outpath doesnt exist')
			
	for file in os.listdir(path):
		orig_path = os.path.join(path,file)
		img = Image.open(orig_path)
		hratio = size[0]/img.size[0]
		wratio = size[1]/img.size[1]
		xshape = min(hratio,wratio)
		if xshape > 1:
			enlarged_size = (int(img.size[0]*xshape),int(img.size[1]*xshape))
			img = img.resize(enlarged_size)
		img = resizeimage.resize_contain(img,size)
		img.save(os.path.join(outpath,file),img.format)


#Create embedding
def create_inception_embedding(grayscaled_rgb):
	inception = InceptionResNetV2(weights='imagenet', include_top=True)
	inception.graph = tf.get_default_graph()
	grayscaled_rgb_resized = []
	for i in grayscaled_rgb:
		i = resize(i, (299, 299, 3), mode='constant')
		grayscaled_rgb_resized.append(i)
	grayscaled_rgb_resized = np.array(grayscaled_rgb_resized)
	grayscaled_rgb_resized = preprocess_input(grayscaled_rgb_resized)
	print("proceeding to create inception embeding...")
	with inception.graph.as_default():
		embed = inception.predict(grayscaled_rgb_resized)
	print("Successfully created inception embedding...")
	return embed
