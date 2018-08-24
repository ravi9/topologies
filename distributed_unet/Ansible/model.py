#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: EPL-2.0
#


import os.path
import tensorflow as tf
from tensorflow import keras as K

import settings
from tqdm import tqdm

def dice_coef(y_true, y_pred, smooth=1.0):
   intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
   union = tf.reduce_sum(y_true + y_pred, axis=(1, 2, 3))
   numerator = tf.constant(2.) * intersection + smooth
   denominator = union + smooth
   coef = numerator / denominator
   return tf.reduce_mean(coef)


def dice_coef_loss(y_true, y_pred, smooth = 1.):

	intersection = tf.reduce_sum(y_true * y_pred)
	union_set = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
	loss = -tf.log(tf.constant(2.) * intersection + smooth) + \
		tf.log(union_set + smooth)
	return loss

def sensitivity(y_true, y_pred, smooth = 1.):

	intersection = tf.reduce_sum(y_true * y_pred)
	coef = (intersection + smooth) / (tf.reduce_sum(y_true) + smooth)
	return coef

def specificity(y_true, y_pred, smooth = 1.):

	intersection = tf.reduce_sum(y_true * y_pred)
	coef = (intersection + smooth) / (tf.reduce_sum(y_pred) + smooth)
	return coef


if settings.CHANNELS_LAST:
	concat_axis = -1
	data_format = "channels_last"

else:
	concat_axis = 1
	data_format = "channels_first"

K.backend.set_image_data_format(data_format)

def define_model(FLAGS, input_shape, output_shape, num_replicas):
	"""
	Define the Keras model here along with the TensorBoard summaries
	"""

	n_cl_out = 1 # Number of output classes
	dropout = 0.2   # Percentage of dropout for network layers

	num_datapoints = input_shape[0]

	imgs = tf.placeholder(tf.float32, shape=([None] + list(input_shape[1:])))
	msks = tf.placeholder(tf.float32, shape=([None] + list(output_shape[1:])))

	inputs = K.layers.Input(tensor=imgs, name="Images")

	params = dict(kernel_size=(3, 3), activation="relu",
				  padding="same", data_format=data_format,
				  kernel_initializer="he_uniform")

	trans_params = dict(kernel_size=(2, 2), strides=(2, 2),
						data_format=data_format,
						kernel_initializer="he_uniform",
						padding="same")

	conv1 = K.layers.Conv2D(name="conv1a", filters=32, **params)(inputs)
	conv1 = K.layers.Conv2D(name="conv1b", filters=32, **params)(conv1)
	pool1 = K.layers.MaxPooling2D(name="pool1", pool_size=(2, 2))(conv1)

	conv2 = K.layers.Conv2D(name="conv2a", filters=64, **params)(pool1)
	conv2 = K.layers.Conv2D(name="conv2b", filters=64, **params)(conv2)
	pool2 = K.layers.MaxPooling2D(name="pool2", pool_size=(2, 2))(conv2)

	conv3 = K.layers.Conv2D(name="conv3a", filters=128, **params)(pool2)
	### Trying dropout layers earlier on, as indicated in the paper
	conv3 = K.layers.Dropout(dropout)(conv3)
	conv3 = K.layers.Conv2D(name="conv3b", filters=128, **params)(conv3)

	pool3 = K.layers.MaxPooling2D(name="pool3", pool_size=(2, 2))(conv3)

	conv4 = K.layers.Conv2D(name="conv4a", filters=256, **params)(pool3)
	### Trying dropout layers earlier on, as indicated in the paper
	conv4 = K.layers.Dropout(dropout)(conv4)
	conv4 = K.layers.Conv2D(name="conv4b", filters=256, **params)(conv4)

	pool4 = K.layers.MaxPooling2D(name="pool4", pool_size=(2, 2))(conv4)

	conv5 = K.layers.Conv2D(name="conv5a", filters=512, **params)(pool4)
	conv5 = K.layers.Conv2D(name="conv5b", filters=512, **params)(conv5)

	if FLAGS.use_upsampling:
		up = K.layers.UpSampling2D(name="up6", size=(2, 2))(conv5)
	else:
		up = K.layers.Conv2DTranspose(name="transConv6", filters=256,
							**trans_params)(conv5)

	up6 = K.layers.concatenate([up, conv4], axis=concat_axis)

	conv6 = K.layers.Conv2D(name="conv6a", filters=256, **params)(up6)
	conv6 = K.layers.Conv2D(name="conv6b", filters=256, **params)(conv6)

	if FLAGS.use_upsampling:
		up = K.layers.UpSampling2D(name="up7", size=(2, 2))(conv6)
	else:
		up = K.layers.Conv2DTranspose(name="transConv7", filters=128,
							**trans_params)(conv6)

	up7 = K.layers.concatenate([up, conv3], axis=concat_axis)

	conv7 = K.layers.Conv2D(name="conv7a", filters=128, **params)(up7)
	conv7 = K.layers.Conv2D(name="conv7b", filters=128, **params)(conv7)

	if FLAGS.use_upsampling:
		up = K.layers.UpSampling2D(name="up8", size=(2, 2))(conv7)
	else:
		up = K.layers.Conv2DTranspose(name="transConv8", filters=64,
							**trans_params)(conv7)

	up8 = K.layers.concatenate([up, conv2], axis=concat_axis)

	conv8 = K.layers.Conv2D(name="conv8a", filters=64, **params)(up8)
	conv8 = K.layers.Conv2D(name="conv8b", filters=64, **params)(conv8)

	if FLAGS.use_upsampling:
		up = K.layers.UpSampling2D(name="up9", size=(2, 2))(conv8)
	else:
		up = K.layers.Conv2DTranspose(name="transConv9", filters=32,
							**trans_params)(conv8)

	up9 = K.layers.concatenate([up, conv1], axis=concat_axis)

	conv9 = K.layers.Conv2D(name="conv9a", filters=32, **params)(up9)
	conv9 = K.layers.Conv2D(name="conv9b", filters=32, **params)(conv9)

	predictionMask = K.layers.Conv2D(name="Mask", filters=n_cl_out,
							kernel_size=(1, 1),
							data_format=data_format,
							activation="sigmoid")(conv9)

	"""
	Define the variables, losses, and metrics
	We"ll return these as a dictionary called "model"
	"""
	model = {}
	model["input"] = imgs
	model["label"] = msks
	model["output"] = predictionMask
	model["loss"] = dice_coef_loss(msks, predictionMask)
	model["metric_dice"] = dice_coef(msks, predictionMask)

	model["metric_sensitivity"] = sensitivity(msks, predictionMask)
	model["metric_specificity"] = specificity(msks, predictionMask)

	model["global_step"] = tf.Variable(0,dtype=tf.int32,trainable=False,name="global_step")
	# model["global_step"] = tf.train.get_or_create_global_step()
	#
	model["metric_dice_test"] = dice_coef(msks, predictionMask)
	model["loss_test"] = dice_coef_loss(msks, predictionMask)

	# Print the percent steps complete to TensorBoard
	#   so that we know how much of the training remains.
	num_steps_tf = tf.constant(num_datapoints / FLAGS.batch_size * FLAGS.epochs,
				   tf.float32)
	model["percent_complete"] = tf.constant(100.0) * \
			tf.to_float(model["global_step"]) / num_steps_tf

	learning_rate = tf.convert_to_tensor(FLAGS.learning_rate, dtype=tf.float32)
	optimizer = tf.train.AdamOptimizer(learning_rate)

	# Wrap the optimizer in SyncReplicasOptimizer in order
	# to have synchronous SGD. Otherwise, it will be asynchronous
	if FLAGS.is_sync:
		optimizer = tf.train.SyncReplicasOptimizer(optimizer,
							replicas_to_aggregate=num_replicas,
							total_num_replicas=num_replicas)


	model["optimizer"] = optimizer

	model["train_op"] = optimizer.minimize(model["loss"], model["global_step"])

	"""
	Summaries for TensorBoard
	"""
	tf.summary.scalar("loss", model["loss"])
	tf.summary.histogram("loss", model["loss"])
	tf.summary.scalar("dice", model["metric_dice"])
	tf.summary.histogram("dice", model["metric_dice"])

	tf.summary.scalar("sensitivity", model["metric_sensitivity"])
	tf.summary.histogram("sensitivity", model["metric_sensitivity"])
	tf.summary.scalar("specificity", model["metric_specificity"])
	tf.summary.histogram("specificity", model["metric_specificity"])

	tf.summary.image("predictions", predictionMask, max_outputs=settings.TENSORBOARD_IMAGES)
	tf.summary.image("ground_truth", msks, max_outputs=settings.TENSORBOARD_IMAGES)
	tf.summary.image("images", imgs, max_outputs=settings.TENSORBOARD_IMAGES)

	tf.summary.scalar("percent_complete", model["percent_complete"])

	summary_op = tf.summary.merge_all()
	# tf.summary.scalar("dice_test", model["metric_dice_test"])
	# tf.summary.scalar("loss_test", model["loss_test"])

	return model


def validate_model(FLAGS, sess, model, validation_data, epoch):

	dice_v_test = 0.0
	loss_v_test = 0.0
	sens_v_test = 0.0
	spec_v_test = 0.0

	for idx in tqdm(range(0, validation_data["length"] - FLAGS.batch_size, FLAGS.batch_size),
		desc="Calculating metrics on test dataset", leave=False):

		if ((idx+FLAGS.batch_size) >= validation_data["length"]):
			x_test = validation_data["input"][idx:]
			y_test = validation_data["label"][idx:]
		else:
			x_test = validation_data["input"][idx:(idx+FLAGS.batch_size)]
			y_test = validation_data["label"][idx:(idx+FLAGS.batch_size)]

		feed_dict = {model["input"]: x_test, model["label"]: y_test}

		l_v, d_v, sens_v, spec_v = sess.run([model["loss_test"],
			model["metric_dice_test"], model["metric_sensitivity"],
			model["metric_specificity"]], feed_dict=feed_dict)

		dice_v_test += d_v / (validation_data["length"] // FLAGS.batch_size)
		loss_v_test += l_v / (validation_data["length"] // FLAGS.batch_size)
		sens_v_test += sens_v / (validation_data["length"] // FLAGS.batch_size)
		spec_v_test += spec_v / (validation_data["length"] // FLAGS.batch_size)

	print("\nTEST DATASET (Epoch {} of {})\n" \
		  "Loss on test dataset = {:.4f}\n" \
		  "Dice on test dataset = {:.4f}\n" \
		  "Sensitivity on test dataset = {:.4f}\n" \
		  "Specificity on test dataset = {:.4f}\n" \
		  .format(epoch, FLAGS.epochs,
		  loss_v_test, dice_v_test,
		  sens_v_test, spec_v_test))
