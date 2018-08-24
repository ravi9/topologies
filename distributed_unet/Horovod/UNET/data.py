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
import tensorflow as tf

def load_datasets(FLAGS):

    # Load train data
    print("-"*42)
    print("Loading and preprocessing training data...")
    print("-"*42)
    imgs_train, msks_train = load_data(FLAGS.data_path,"_train")
    imgs_train, msks_train = update_channels(imgs_train, msks_train,
                                             settings.NUM_IN_CHANNELS,
                                             settings.NUM_OUT_CHANNELS,
                                             settings.MODE)

    # Load test data
    print("-"*38)
    print("Loading and preprocessing test data...")
    print("-"*38)
    imgs_test, msks_test = load_data(FLAGS.data_path,"_test")
    imgs_test, msks_test = update_channels(imgs_test, msks_test,
                                           settings.NUM_IN_CHANNELS,
                                           settings.NUM_OUT_CHANNELS,
                                           settings.MODE)

    print("Training images shape: {}".format(imgs_train.shape))
    print("Training masks shape:  {}".format(msks_train.shape))
    print("Testing images shape:  {}".format(imgs_test.shape))
    print("Testing masks shape:   {}".format(msks_test.shape))

    """
    Iterator for Dataset
    """
    # train = (imgs_train, msks_train)
    # train_dataset = tf.data.Dataset.from_tensor_slices(train_data).\
    #             shuffle(32000, reshuffle_each_iteration=True).repeat().batch(FLAGS.batch_size)
    # test = (imgs_test, msks_test)
    # test_dataset = tf.data.Dataset.from_tensor_slices(test_data).\
    #             batch(FLAGS.batch_size).repeat()
    #
    # train_iterator = train_dataset.make_initializable_iterator()
    # test_iterator = test_dataset.make_initializable_iterator()

    # return train_iterator, test_iterator

    return imgs_train, msks_train, imgs_test, msks_test

def get_batch(imgs, msks, batch_size):

    idx = np.random.choice(len(imgs), batch_size)
    return imgs[idx,:,:,:], msks[idx,:,:,:]
