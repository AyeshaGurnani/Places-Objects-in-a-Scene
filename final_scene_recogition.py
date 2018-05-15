import tensorflow as tf

import keras
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2, numpy as np
import pickle
import sys

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
keras.backend.set_session(sess)
	

model = None
labels = None

#Convolutional Neural Network Model definition
def VGG_16(weights_path=None):
	model = Sequential()
	model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(Flatten())
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(365, activation='softmax'))

	model.load_weights(weights_path)
	sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(optimizer=sgd, loss='categorical_crossentropy')
	return model

#Function to predict the scene in the image.
def predict(img_path, ret_dict):
	global model
	global labels
	im = cv2.resize(cv2.imread(img_path), (224, 224)).astype(np.float32)
	im[:,:,0] -= 103.939
	im[:,:,1] -= 116.779
	im[:,:,2] -= 123.68
	im = im.transpose((2,0,1))
	im = np.expand_dims(im, axis=0)
	# Test pretrained model
	out = model.predict(im)
	out  = reversed(np.argsort(out)[0])
	results = []
	for x in out:
		results.append(labels[x])
	#ret_dict['place'] = results[:5]
	# ret_dict = results[:6]
	return results[:6]

#Loading the model and label file.
def init():
	global model
	global labels
	keras.backend.set_image_dim_ordering('th')
	model = VGG_16('models/places/places_vgg_keras.h5')
	labels = pickle.load(open('models/places/labels.pkl','rb'))

#Function to return the directory that have the labels stored for top predictions.
init()
print "Loaded place module"
def run(im_name):


	ret_dict = {}
	z = predict(im_name, ret_dict)
	# print z[:6]
	return z[:6]