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
from tensorflow import layers
from tensorflow.examples.tutorials.mnist import input_data
import horovod.tensorflow as hvd

import os
from datetime import datetime


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("inter_op_threads", 2,
							"# inter op threads")
tf.app.flags.DEFINE_integer("intra_op_threads", os.cpu_count(),
							"# intra op threads")

tf.app.flags.DEFINE_integer("total_steps", 2000,
							"Number of training steps")

tf.app.flags.DEFINE_integer("log_steps", 20,
							"Number of steps between logs")
tf.app.flags.DEFINE_integer("batch_size", 128,
							"Batch Size for Training")

config = tf.ConfigProto(intra_op_parallelism_threads=FLAGS.intra_op_threads,
                        inter_op_parallelism_threads=FLAGS.inter_op_threads)

tf.logging.set_verbosity(tf.logging.INFO)


def get_model(feature, label):

    # Reshape the input vector into a 28x28 image
    input_layer = tf.reshape(feature, [-1, 28, 28, 1])

    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        name="conv1")

    pool1 = tf.layers.max_pooling2d(
            inputs=conv1,
            pool_size=[2, 2],
            strides=2,
            name="pool1")

    conv2 = tf.layers.conv2d(
         inputs=pool1,
         filters=64,
         kernel_size=[5, 5],
         padding="same",
         activation=tf.nn.relu,
         name="conv2")

    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[2, 2],
        strides=2,
        name="pool2")

    # Flatten
    pool2_flat = layers.Flatten()(pool2)

    # Add a Dense layer
    dense = tf.layers.dense(inputs=pool2_flat,
                            units=1024,
                            activation=tf.nn.relu,
                            name="Dense1")

    # Add a dropout layer
    dropout = tf.layers.dropout(
        inputs=dense,
        rate=0.4, # Dropout rate
        training=True,
        name="dropout")

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout,
                             units=10,
                             name="Logits_layer")

    onehot_labels = tf.one_hot(tf.cast(label, tf.int32), 10, 1, 0)

    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels,
                                           logits=logits)

    correct_prediction = tf.equal(tf.cast(tf.argmax(logits,1),
                                  tf.float32), label)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    """
    Print out TensorBoard Summaries
    """
    tf.summary.scalar("loss", loss)
    tf.summary.histogram("loss", loss)
    tf.summary.scalar("training_accuracy", accuracy)
    tf.summary.image("images", input_layer, max_outputs=3)
    summary_op = tf.summary.merge_all()

    return tf.argmax(logits, 1), loss, accuracy


def main(_):

    # Initialize Horovod.
    hvd.init()

    # Download MNIST dataset.
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

    # Input tensors
    with tf.name_scope("input"):
        image = tf.placeholder(tf.float32, [None, 784], name="image")
        label = tf.placeholder(tf.float32, [None], name="label")

    # Define model
    predict, loss, accuracy = get_model(image, label)

    # Horovod: adjust learning rate based on number workers
    opt = tf.train.RMSPropOptimizer(0.001 * hvd.size())

    # Wrap optimizer with Horovod Distributed Optimizer.
    opt = hvd.DistributedOptimizer(opt)

    global_step = tf.train.get_or_create_global_step()
    train_op = opt.minimize(loss, global_step=global_step)

    last_step = FLAGS.total_steps // hvd.size()

    hooks = [
        # Horovod: BroadcastGlobalVariablesHook broadcasts
        # initial variable states from rank 0 to all other
        # processes. This is necessary to ensure consistent
        # initialization of all workers when training is
        # started with random weights
        # or restored from a checkpoint.
        hvd.BroadcastGlobalVariablesHook(0),

        # Horovod: adjust number of steps based on number of GPUs.
        tf.train.StopAtStepHook(last_step=last_step),

        # Prints the loss and step every 10 steps
        tf.train.LoggingTensorHook(tensors={"step": global_step,
                                    "loss": loss,
                                    "accuracy": accuracy},
                                   every_n_iter=FLAGS.log_steps),
    ]

    # Horovod: save checkpoints only on worker 0 to prevent other workers from
    # corrupting them.
    if hvd.rank() == 0:
        checkpoint_dir = "./checkpoints/{}-workers".format(hvd.size())
    else:
        checkpoint_dir = None

    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint,
    # and closing when done or an error occurs.
    with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,
                                           hooks=hooks,
                                           save_summaries_steps=FLAGS.log_steps,
                                           log_step_count_steps=FLAGS.log_steps,
                                           config=config) as mon_sess:
        while not mon_sess.should_stop():

            # Run a training step synchronously.
            image_, label_ = mnist.train.next_batch(FLAGS.batch_size)
            mon_sess.run(train_op, feed_dict={image: image_, label: label_})


if __name__ == "__main__":

    start_time = datetime.now()
    print("Starting at: {}".format(start_time))
    tf.app.run()
    stop_time = datetime.now()
    print("Stopping at: {}".format(stop_time))
    print("Elapsed time was: {}".format(stop_time-start_time))
