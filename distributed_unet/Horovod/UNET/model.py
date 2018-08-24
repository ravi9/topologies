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

import tensorflow as tf
import tensorflow.keras as K


def dice_coef(y_true, y_pred, smooth=1.0):
   intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
   union = tf.reduce_sum(y_true + y_pred, axis=(1, 2, 3))
   numerator = tf.constant(2.) * intersection + smooth
   denominator = union + smooth
   coef = numerator / denominator
   return tf.reduce_mean(coef)


def dice_coef_loss(y_true, y_pred, smooth=1.0):

	intersection = tf.reduce_sum(y_true * y_pred)
	union_set = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
	loss = -tf.log(tf.constant(2.) * intersection + smooth) + \
		tf.log(union_set + smooth)
	return loss


def sensitivity(y_true, y_pred, smooth=1.):

	intersection = tf.reduce_sum(y_true * y_pred)
	coef = (intersection + smooth) / (tf.reduce_sum(y_true) + smooth)
	return coef


def specificity(y_true, y_pred, smooth=1.):

	intersection = tf.reduce_sum(y_true * y_pred)
	coef = (intersection + smooth) / (tf.reduce_sum(y_pred) + smooth)
	return coef

def define_model(input_shape, output_shape, FLAGS):
    """
    Define the model along with the TensorBoard summaries
    """

    data_format = "channels_last"
    concat_axis = -1
    n_cl_out = 1  # Number of output classes
    dropout = 0.2   # Percentage of dropout for network layers

    num_datapoints = input_shape[0]

    imgs = tf.placeholder(tf.float32,
                          shape=([None] + list(input_shape[1:])))
    msks = tf.placeholder(tf.float32,
                          shape=([None] + list(output_shape[1:])))

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
    # Trying dropout layers earlier on, as indicated in the paper
    conv3 = K.layers.Dropout(dropout)(conv3)
    conv3 = K.layers.Conv2D(name="conv3b", filters=128, **params)(conv3)

    pool3 = K.layers.MaxPooling2D(name="pool3", pool_size=(2, 2))(conv3)

    conv4 = K.layers.Conv2D(name="conv4a", filters=256, **params)(pool3)
    # Trying dropout layers earlier on, as indicated in the paper
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

    model["metric_dice_test"] = dice_coef(msks, predictionMask)
    model["loss_test"] = dice_coef_loss(msks, predictionMask)

    model["metric_sensitivity_test"] = sensitivity(msks, predictionMask)
    model["metric_specificity_test"] = specificity(msks, predictionMask)

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

    tf.summary.image("predictions", predictionMask, max_outputs=3)
    tf.summary.image("ground_truth", msks, max_outputs=3)
    tf.summary.image("images", imgs, max_outputs=3)

    summary_op = tf.summary.merge_all()

    return model


def validate_model(mon_sess, model, imgs_test, msks_test):
    """
    Code for model validation on test set
    """
    # test_dice, test_sens, test_spec = mon_sess.run(
    #              [model["metric_dice_test"],
    #              model["metric_sensitivity_test"],
    #              model["metric_specificity_test"]],
    #              feed_dict={model["input"]: imgs_test,
    #              model["label"]: msks_test})
    #
    # tf.logging.info("VALIDATION METRICS: Test Dice = {:.4f}, "
    #                 "Test Sensitivity = {:.4f}, "
    #                 "Test Specificity = {:.4f}".format(test_dice,
    #                 test_sens, test_spec))
    pass
