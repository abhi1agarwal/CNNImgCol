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
import constants




model = getmodel()




# Image transformer
datagen = ImageDataGenerator(
        shear_range=0.4,
        zoom_range=0.4,
        rotation_range=40,
        horizontal_flip=True)

#Generate training data
batch_size = constants.BATCH_SIZE


# Get images
X = []
for filename in os.listdir('/data/images/Train/'):
    X.append(img_to_array(load_img('/data/images/Train/'+filename)))
X = np.array(X, dtype=float)
Xtrain = 1.0/255*X

def image_a_b_gen(batch_size):
    for batch in datagen.flow(Xtrain, batch_size=batch_size):
        grayscaled_rgb = gray2rgb(rgb2gray(batch))
        embed = create_inception_embedding(grayscaled_rgb)
        lab_batch = rgb2lab(batch)
        X_batch = lab_batch[:,:,:,0]
        X_batch = X_batch.reshape(X_batch.shape+(1,))
        Y_batch = lab_batch[:,:,:,1:] / 128
        yield ([X_batch, create_inception_embedding(grayscaled_rgb)], Y_batch)

print("Training the model now...")

#Train model
filepath="weights-improvement-{epoch:02d}-{loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_weights_only=True, period=20)
callbacks_list = [checkpoint]
model.compile(optimizer='adam', loss='mse')
print("model fit generator called ...")

cc=0
for x in image_a_b_gen(batch_size):
    cc=cc+1
    print("cc has changed to ::",cc)
    print("\n\n")

print("cc final is ::",cc)

print("\n\nNow running model generator ...\n\n\n")
model.fit_generator(image_a_b_gen(batch_size), epochs=constants.TRAIN_EPOCHS, steps_per_epoch=1, callbacks=callbacks_list, verbose=1)


print("Saving the model now...")
# Save model
model_json = model.to_json()
with open(constants.JSON_FILE, "w") as json_file:
    json_file.write(model_json)
model.save_weights(constants.H5_FILE)

