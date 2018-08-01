import numpy as np 
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
import os

class autoencoderClass:
	@staticmethod
	def build():
		input_img=Input(shape=(2516/2,3272/2,1))
		x=Conv2D(32,(3,3),activation='relu',padding='same')(input_img)
		x=UpSampling2D((2,2))(x)
		x=Conv2D(32,(3,3),activation='relu',padding='same')(x)
		x=UpSampling2D((2,2))(x)
		x=Conv2D(16, (3, 3), activation='relu',padding='same')(x)
		x=UpSampling2D((2,2))(x)
		x=Conv2D(16, (3, 3), activation='relu',padding='same')(x)
		x= MaxPooling2D((2,2),padding='same')(x)
		x= Conv2D(16, (3, 3), activation='relu',padding='same')(x)
		x= MaxPooling2D((2,2),padding='same')(x)
		x= Conv2D(16, (3, 3), activation='relu',padding='same')(x)
		x= MaxPooling2D((2,2),padding='same')(x)
		x= Conv2D(16, (3, 3), activation='relu',padding='same')(x)  
		decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

		autoencoder=Model(input_img,decoded)
		return autoencoder