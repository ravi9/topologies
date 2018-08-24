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

'''
This loads the trained model and runs it on the test set.
Should provide a sanity check on the TensorFlow model.
'''

import tensorflow as tf
from preprocess import load_data, update_channels
from tqdm import tqdm
import numpy as np
import settings_dist
import os
from tempfile import TemporaryFile

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Get rid of the AVX, SSE warnings

batch_size = 128
export_dir=settings_dist.CHECKPOINT_DIRECTORY + "saved_model/"
print("Loading trained TensorFlow model from directory {}".format(export_dir))

def load_test_data():

    # Load test data
    print('-'*38)
    print('Loading and preprocessing test data...')
    print('-'*38)
    imgs_test, msks_test = load_data(settings_dist.OUT_PATH,"_test")
    imgs_test, msks_test = update_channels(imgs_test, msks_test,
                            settings_dist.IN_CHANNEL_NO,
                            settings_dist.OUT_CHANNEL_NO,
                            settings_dist.MODE)

    return imgs_test, msks_test

def calc_dice(a,b):

    a1 = np.ndarray.flatten(a)
    b1 = np.ndarray.flatten(b)

    return 2.0*(np.sum(a1*b1)+1.0)/(np.sum(a1+b1)+1.0)

with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, ["serve"], export_dir)
    graph = tf.get_default_graph()
    imgs = graph.get_tensor_by_name("Placeholder:0")
    preds = graph.get_tensor_by_name("Mask/Sigmoid:0")

    imgs_test, msks_test = load_test_data()

    dice = 0.0
    i = 0

    msks_test_predictions = []

    for idx in tqdm(range(0, imgs_test.shape[0] - batch_size, batch_size),
                    desc="Calculating metrics on test dataset", leave=False):
        x_test = imgs_test[idx:(idx+batch_size)]
        y_test = msks_test[idx:(idx+batch_size)]

        feed_dict = {imgs: x_test}

        p = np.array(sess.run([preds], feed_dict=feed_dict))
        dice += calc_dice(y_test, p)

        # Add prediction to numpy array
        msks_test_predictions.append(p)

        i += 1

# Save predictions
save_dir = settings_dist.OUT_PATH
np.save("{0}msks_test_predictions.npy".format(save_dir), msks_test_predictions)

print("Average Dice for Test Set = {0}".format(dice/i))
