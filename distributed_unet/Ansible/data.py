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

from preprocess import load_data, update_channels
import settings
import numpy as np

def load_all_data():

	# Load train data
	print("-"*42)
	print("Loading and preprocessing training data...")
	print("-"*42)
	imgs_train, msks_train = load_data(settings.OUT_PATH,"_train")
	imgs_train, msks_train = update_channels(imgs_train, msks_train,
											 settings.IN_CHANNEL_NO,
											 settings.OUT_CHANNEL_NO,
											 settings.MODE)

	# Load test data
	print("-"*38)
	print("Loading and preprocessing test data...")
	print("-"*38)
	imgs_test, msks_test = load_data(settings.OUT_PATH,"_test")
	imgs_test, msks_test = update_channels(imgs_test, msks_test,
										   settings.IN_CHANNEL_NO,
										   settings.OUT_CHANNEL_NO,
										   settings.MODE)

	print("Training images shape: {}".format(imgs_train.shape))
	print("Training masks shape:  {}".format(msks_train.shape))
	print("Testing images shape:  {}".format(imgs_test.shape))
	print("Testing masks shape:   {}".format(msks_test.shape))

	return imgs_train, msks_train, imgs_test, msks_test


def get_epochs(batch_size,imgs_train,msks_train):
	"""
	Get an epoch of batches in random order
	"""

	# Assuming imgs_train and msks_train are the same size
	train_size = imgs_train.shape[0]
	image_width = imgs_train.shape[1]
	image_height = imgs_train.shape[2]
	image_channels = imgs_train.shape[3]

	epoch_length = train_size - train_size%batch_size
	batch_count = epoch_length//batch_size

	# Shuffle and truncate arrays to equal 1 epoch
	zipped = zip(imgs_train,msks_train)
	np.random.shuffle(zipped)
	data,labels = zip(*zipped)
	data = np.asarray(data)[:epoch_length]
	labels = np.asarray(labels)[:epoch_length]

	# Reshape arrays into batch_count batches of length batch_size
	data = data.reshape((batch_count,batch_size,image_width,image_height,image_channels))
	labels = labels.reshape((batch_count,batch_size,image_width,image_height,image_channels))

	# Join batches of training examples with batches of labels
	epoch_of_batches = zip(data,labels)

	return np.array(epoch_of_batches)

def load_datasets():

	validation_data = {}
	training_data = {}
	# Load the data
	training_data["input"], training_data["label"], \
			validation_data["input"], \
			validation_data["label"] = load_all_data()
	training_data["length"] = training_data["input"].shape[0]  # Number of train datasets

	validation_data["length"] = validation_data["input"].shape[0]   # Number of test datasets

	return training_data, validation_data
