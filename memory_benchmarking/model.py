#!/usr/bin/python

# ----------------------------------------------------------------------------
# Copyright 2018 Intel
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

import os.path
import numpy as np
import tensorflow as tf
import keras as K

def dice_coef(y_true, y_pred, axis=(1,2,3), smooth=1.0):
   intersection = tf.reduce_sum(y_true * y_pred, axis=axis)
   union = tf.reduce_sum(y_true + y_pred, axis=axis)
   numerator = tf.constant(2.) * intersection + smooth
   denominator = union + smooth
   coef = numerator / denominator
   return tf.reduce_mean(coef)

def dice_coef_loss(target, prediction, axis=(1,2,3), smooth=1.):
	"""
	Sorenson Dice loss
	Using -log(Dice) as the loss since it is better behaved.
	Also, the log allows avoidance of the division which
	can help prevent underflow when the numbers are very small.
	"""
	intersection = tf.reduce_sum(prediction * target, axis=axis)
	p = tf.reduce_sum(prediction, axis=axis)
	t = tf.reduce_sum(target, axis=axis)
	numerator = tf.reduce_mean(2. * intersection + smooth)
	denominator = tf.reduce_mean(t + p + smooth)
	dice_loss = -tf.log(numerator) + tf.log(denominator)

	return dice_loss

CHANNEL_LAST = True
if CHANNEL_LAST:
	concat_axis = -1
	data_format = "channels_last"

else:
	concat_axis = 1
	data_format = "channels_first"

def unet3D(input_img, use_upsampling=False, n_out=1, dropout=0.2,
			print_summary = False, return_model=False):
	"""
	3D U-Net model
	"""
	print("3D U-Net Segmentation")

	inputs = K.layers.Input(tensor=input_img, name="Input_Image")

	params = dict(kernel_size=(3, 3, 3), activation=None,
				  padding="same", data_format=data_format,
				  kernel_initializer="he_uniform")

	conv1 = K.layers.Conv3D(name="conv1a", filters=32, **params)(inputs)
	conv1 = K.layers.BatchNormalization()(conv1)
	conv1 = K.layers.Activation("relu")(conv1)
	conv1 = K.layers.Conv3D(name="conv1b", filters=64, **params)(conv1)
	conv1 = K.layers.BatchNormalization()(conv1)
	conv1 = K.layers.Activation("relu")(conv1)
	pool1 = K.layers.MaxPooling3D(name="pool1", pool_size=(2, 2, 2))(conv1)

	conv2 = K.layers.Conv3D(name="conv2a", filters=64, **params)(pool1)
	conv2 = K.layers.BatchNormalization()(conv2)
	conv2 = K.layers.Activation("relu")(conv2)
	conv2 = K.layers.Conv3D(name="conv2b", filters=128, **params)(conv2)
	conv2 = K.layers.BatchNormalization()(conv2)
	conv2 = K.layers.Activation("relu")(conv2)
	pool2 = K.layers.MaxPooling3D(name="pool2", pool_size=(2, 2, 2))(conv2)

	conv3 = K.layers.Conv3D(name="conv3a", filters=128, **params)(pool2)
	conv3 = K.layers.BatchNormalization()(conv3)
	conv3 = K.layers.Activation("relu")(conv3)
	conv3 = K.layers.Dropout(dropout)(conv3) ### Trying dropout layers earlier on, as indicated in the paper
	conv3 = K.layers.Conv3D(name="conv3b", filters=256, **params)(conv3)
	conv3 = K.layers.BatchNormalization()(conv3)
	conv3 = K.layers.Activation("relu")(conv3)
	pool3 = K.layers.MaxPooling3D(name="pool3", pool_size=(2, 2, 2))(conv3)

	conv4 = K.layers.Conv3D(name="conv4a", filters=256, **params)(pool3)
	conv4 = K.layers.BatchNormalization()(conv4)
	conv4 = K.layers.Activation("relu")(conv4)
	conv4 = K.layers.Dropout(dropout)(conv4) ### Trying dropout layers earlier on, as indicated in the paper
	conv4 = K.layers.Conv3D(name="conv4b", filters=512, **params)(conv4)
	conv4 = K.layers.BatchNormalization()(conv4)
	conv4 = K.layers.Activation("relu")(conv4)

	if use_upsampling:
		up = K.layers.UpSampling3D(name="up4", size=(2, 2, 2))(conv4)
	else:
		up = K.layers.Conv3DTranspose(name="transConv4", filters=512, data_format=data_format,
						   kernel_size=(2, 2, 2), strides=(2, 2, 2), padding="same")(conv4)

	up4 = K.layers.concatenate([up, conv3], axis=concat_axis)

	conv5 = K.layers.Conv3D(name="conv5a", filters=256, **params)(up4)
	conv5 = K.layers.BatchNormalization()(conv5)
	conv5 = K.layers.Activation("relu")(conv5)
	conv5 = K.layers.Conv3D(name="conv5b", filters=256, **params)(conv5)
	conv5 = K.layers.BatchNormalization()(conv5)
	conv5 = K.layers.Activation("relu")(conv5)

	if use_upsampling:
		up = K.layers.UpSampling3D(name="up5", size=(2, 2, 2))(conv5)
	else:
		up = K.layers.Conv3DTranspose(name="transConv5", filters=256, data_format=data_format,
						   kernel_size=(2, 2, 2), strides=(2, 2, 2), padding="same")(conv5)

	up5 = K.layers.concatenate([up, conv2], axis=concat_axis)

	conv6 = K.layers.Conv3D(name="conv6a", filters=128, **params)(up5)
	conv6 = K.layers.BatchNormalization()(conv6)
	conv6 = K.layers.Activation("relu")(conv6)
	conv6 = K.layers.Conv3D(name="conv6b", filters=128, **params)(conv6)
	conv6 = K.layers.BatchNormalization()(conv6)
	conv6 = K.layers.Activation("relu")(conv6)

	if use_upsampling:
		up = K.layers.UpSampling3D(name="up6", size=(2, 2, 2))(conv6)
	else:
		up = K.layers.Conv3DTranspose(name="transConv6", filters=128, data_format=data_format,
						   kernel_size=(2, 2, 2), strides=(2, 2, 2), padding="same")(conv6)

	up6 = K.layers.concatenate([up, conv1], axis=concat_axis)

	conv7 = K.layers.Conv3D(name="conv7a", filters=64, **params)(up6)
	conv7 = K.layers.BatchNormalization()(conv7)
	conv7 = K.layers.Activation("relu")(conv7)
	conv7 = K.layers.Conv3D(name="conv7b", filters=64, **params)(conv7)
	conv7 = K.layers.BatchNormalization()(conv7)
	conv7 = K.layers.Activation("relu")(conv7)
	pred = K.layers.Conv3D(name="Prediction", filters=n_out, kernel_size=(1, 1, 1),
					data_format=data_format, activation="sigmoid")(conv7)

	if return_model:
		model = K.models.Model(inputs=[inputs], outputs=[pred])

		if print_summary:
			print(model.summary())

		return pred, model
	else:
		return pred

def unet2D(input_tensor, use_upsampling=False,
			n_out=1, dropout=0.2, print_summary = False, return_model=False):
	"""
	2D U-Net
	"""
	print("2D U-Net Segmentation")

	inputs = K.layers.Input(tensor=input_tensor, name="Images")

	params = dict(kernel_size=(3, 3), activation="relu",
				  padding="same", data_format=data_format,
				  kernel_initializer="he_uniform") #RandomUniform(minval=-0.01, maxval=0.01, seed=816))

	conv1 = K.layers.Conv2D(name="conv1a", filters=32, **params)(inputs)
	conv1 = K.layers.Conv2D(name="conv1b", filters=32, **params)(conv1)
	pool1 = K.layers.MaxPooling2D(name="pool1", pool_size=(2, 2))(conv1)

	conv2 = K.layers.Conv2D(name="conv2a", filters=64, **params)(pool1)
	conv2 = K.layers.Conv2D(name="conv2b", filters=64, **params)(conv2)
	pool2 = K.layers.MaxPooling2D(name="pool2", pool_size=(2, 2))(conv2)

	conv3 = K.layers.Conv2D(name="conv3a", filters=128, **params)(pool2)
	conv3 = K.layers.Dropout(dropout)(conv3) ### Trying dropout layers earlier on, as indicated in the paper
	conv3 = K.layers.Conv2D(name="conv3b", filters=128, **params)(conv3)

	pool3 = K.layers.MaxPooling2D(name="pool3", pool_size=(2, 2))(conv3)

	conv4 = K.layers.Conv2D(name="conv4a", filters=256, **params)(pool3)
	conv4 = K.layers.Dropout(dropout)(conv4) ### Trying dropout layers earlier on, as indicated in the paper
	conv4 = K.layers.Conv2D(name="conv4b", filters=256, **params)(conv4)

	pool4 = K.layers.MaxPooling2D(name="pool4", pool_size=(2, 2))(conv4)

	conv5 = K.layers.Conv2D(name="conv5a", filters=512, **params)(pool4)


	if use_upsampling:
		conv5 = K.layers.Conv2D(name="conv5b", filters=256, **params)(conv5)
		up6 = K.layers.concatenate([K.layers.UpSampling2D(name="up6", size=(2, 2))(conv5), conv4], axis=concat_axis)
	else:
		conv5 = K.layers.Conv2D(name="conv5b", filters=512, **params)(conv5)
		up6 = K.layers.concatenate([K.layers.Conv2DTranspose(name="transConv6", filters=256, data_format=data_format,
						   kernel_size=(2, 2), strides=(2, 2), padding="same")(conv5), conv4], axis=concat_axis)

	conv6 = K.layers.Conv2D(name="conv6a", filters=256, **params)(up6)


	if use_upsampling:
		conv6 = K.layers.Conv2D(name="conv6b", filters=128, **params)(conv6)
		up7 = K.layers.concatenate([K.layers.UpSampling2D(name="up7", size=(2, 2))(conv6), conv3], axis=concat_axis)
	else:
		conv6 = K.layers.Conv2D(name="conv6b", filters=256, **params)(conv6)
		up7 = K.layers.concatenate([K.layers.Conv2DTranspose(name="transConv7", filters=128, data_format=data_format,
						   kernel_size=(2, 2), strides=(2, 2), padding="same")(conv6), conv3], axis=concat_axis)

	conv7 = K.layers.Conv2D(name="conv7a", filters=128, **params)(up7)


	if use_upsampling:
		conv7 = K.layers.Conv2D(name="conv7b", filters=64, **params)(conv7)
		up8 = K.layers.concatenate([K.layers.UpSampling2D(name="up8", size=(2, 2))(conv7), conv2], axis=concat_axis)
	else:
		conv7 = K.layers.Conv2D(name="conv7b", filters=128, **params)(conv7)
		up8 = K.layers.concatenate([K.layers.Conv2DTranspose(name="transConv8", filters=64, data_format=data_format,
						   kernel_size=(2, 2), strides=(2, 2), padding="same")(conv7), conv2], axis=concat_axis)


	conv8 = K.layers.Conv2D(name="conv8a", filters=64, **params)(up8)

	if use_upsampling:
		conv8 = K.layers.Conv2D(name="conv8b", filters=32, **params)(conv8)
		up9 = K.layers.concatenate([K.layers.UpSampling2D(name="up9", size=(2, 2))(conv8), conv1], axis=concat_axis)
	else:
		conv8 = K.layers.Conv2D(name="conv8b", filters=64, **params)(conv8)
		up9 = K.layers.concatenate([K.layers.Conv2DTranspose(name="transConv9", filters=32, data_format=data_format,
						   kernel_size=(2, 2), strides=(2, 2), padding="same")(conv8), conv1], axis=concat_axis)


	conv9 = K.layers.Conv2D(name="conv9a", filters=32, **params)(up9)
	conv9 = K.layers.Conv2D(name="conv9b", filters=32, **params)(conv9)

	pred = K.layers.Conv2D(name="Prediction", filters=n_out, kernel_size=(1, 1),
					data_format=data_format, activation="sigmoid")(conv9)

	if return_model:
		model = K.models.Model(inputs=[inputs], outputs=[pred])

		if print_summary:
			print(model.summary())

		return pred, model
	else:
		return pred

def conv3D(input_img, print_summary = False, dropout=0.2, n_out=1,
			return_model=False):
	"""
	Simple 3D convolution model based on VGG-16
	"""
	print("3D Convolutional Binary Classifier based on VGG-16")

	inputs = K.layers.Input(tensor=input_img, name="Images")

	params = dict(kernel_size=(3, 3, 3), activation="relu",
				  padding="same", data_format=data_format,
				  kernel_initializer="he_uniform") #RandomUniform(minval=-0.01, maxval=0.01, seed=816))

	conv1 = K.layers.Conv3D(name="conv1", filters=64, **params)(inputs)
	conv2 = K.layers.Conv3D(name="conv2", filters=64, **params)(conv1)
	pool1 = K.layers.MaxPooling3D(name="pool1", pool_size=(2, 2, 2))(conv2)

	conv3 = K.layers.Conv3D(name="conv3", filters=128, **params)(pool1)
	conv4 = K.layers.Conv3D(name="conv4", filters=128, **params)(conv3)
	pool2 = K.layers.MaxPooling3D(name="pool2", pool_size=(2, 2, 2))(conv4)

	conv5 = K.layers.Conv3D(name="conv5", filters=256, **params)(pool2)
	conv6 = K.layers.Conv3D(name="conv6", filters=256, **params)(conv5)
	conv7 = K.layers.Conv3D(name="conv7", filters=256, **params)(conv6)
	pool3 = K.layers.MaxPooling3D(name="pool3", pool_size=(2, 2, 2))(conv7)

	conv8 = K.layers.Conv3D(name="conv8", filters=512, **params)(pool3)
	conv9 = K.layers.Conv3D(name="conv9", filters=512, **params)(conv8)
	conv10 = K.layers.Conv3D(name="conv10", filters=512, **params)(conv9)
	pool4 = K.layers.MaxPooling3D(name="pool4", pool_size=(2, 2, 2))(conv10)

	conv11 = K.layers.Conv3D(name="conv11", filters=512, **params)(pool4)
	conv12 = K.layers.Conv3D(name="conv12", filters=512, **params)(conv11)
	conv13 = K.layers.Conv3D(name="conv13", filters=512, **params)(conv12)
	pool5 = K.layers.MaxPooling3D(name="pool5", pool_size=(2, 2, 2))(conv13)

	flat = K.layers.Flatten()(pool5)
	dense1 = K.layers.Dense(4096, activation="relu")(flat)
	drop1 = K.layers.Dropout(dropout)(dense1)
	dense2 = K.layers.Dense(4096, activation="relu")(drop1)
	pred = K.layers.Dense(n_out, name="Prediction", activation="sigmoid")(dense2)

	if return_model:
		model = K.models.Model(inputs=[inputs], outputs=[pred])

		if print_summary:
			print(model.summary())

		return pred, model
	else:
		return pred


def conv2D(input_tensor, print_summary = False, dropout=0.2, n_out=1, return_model=False):

	"""
	Simple 2D convolution model based on VGG-16
	"""
	print("2D Convolutional Binary Classifier based on VGG-16")

	inputs = K.layers.Input(tensor=input_tensor, name="Images")

	params = dict(kernel_size=(3, 3), activation="relu",
				  padding="same", data_format=data_format,
				  kernel_initializer="he_uniform") #RandomUniform(minval=-0.01, maxval=0.01, seed=816))

	conv1 = K.layers.Conv2D(name="conv1", filters=64, **params)(inputs)
	conv2 = K.layers.Conv2D(name="conv2", filters=64, **params)(conv1)
	pool1 = K.layers.MaxPooling2D(name="pool1", pool_size=(2, 2))(conv2)

	conv3 = K.layers.Conv2D(name="conv3", filters=128, **params)(pool1)
	conv4 = K.layers.Conv2D(name="conv4", filters=128, **params)(conv3)
	pool2 = K.layers.MaxPooling2D(name="pool2", pool_size=(2, 2))(conv4)

	conv5 = K.layers.Conv2D(name="conv5", filters=256, **params)(pool2)
	conv6 = K.layers.Conv2D(name="conv6", filters=256, **params)(conv5)
	conv7 = K.layers.Conv2D(name="conv7", filters=256, **params)(conv6)
	pool3 = K.layers.MaxPooling2D(name="pool3", pool_size=(2, 2))(conv7)

	conv8 = K.layers.Conv2D(name="conv8", filters=512, **params)(pool3)
	conv9 = K.layers.Conv2D(name="conv9", filters=512, **params)(conv8)
	conv10 = K.layers.Conv2D(name="conv10", filters=512, **params)(conv9)
	pool4 = K.layers.MaxPooling2D(name="pool4", pool_size=(2, 2))(conv10)

	conv11 = K.layers.Conv2D(name="conv11", filters=512, **params)(pool4)
	conv12 = K.layers.Conv2D(name="conv12", filters=512, **params)(conv11)
	conv13 = K.layers.Conv2D(name="conv13", filters=512, **params)(conv12)
	pool5 = K.layers.MaxPooling2D(name="pool5", pool_size=(2, 2))(conv13)

	flat = K.layers.Flatten()(pool5)
	dense1 = K.layers.Dense(4096, activation="relu")(flat)
	drop1 = K.layers.Dropout(dropout)(dense1)
	dense2 = K.layers.Dense(4096, activation="relu")(drop1)
	pred = K.layers.Dense(n_out, name="Prediction", activation="sigmoid")(dense2)

	if return_model:
		model = K.models.Model(inputs=[inputs], outputs=[pred])

		if print_summary:
			print(model.summary())

		return pred, model
	else:
		return pred
