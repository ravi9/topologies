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

"""

Runs U-Net training on BraTS dataset

"""
import tensorflow as tf
from tensorflow import layers

from model import define_model, validate_model
from data import load_datasets, get_batch

import os
import psutil
from datetime import datetime

import settings


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("num_inter_threads", 2,
                            "# inter op threads")

tf.app.flags.DEFINE_integer("num_threads", psutil.cpu_count(logical=True),
                            "# intra op threads")

tf.app.flags.DEFINE_integer("epochs", settings.EPOCHS,
                            "Number of epochs to train")

tf.app.flags.DEFINE_integer("log_steps", 5,
                            "Number of steps between logs")

tf.app.flags.DEFINE_integer("batch_size", settings.BATCH_SIZE,
                            "Batch Size for Training")

tf.app.flags.DEFINE_string("output_path",
                            settings.OUTPUT_PATH,
                            "Output log directory")

tf.app.flags.DEFINE_string("data_path",
                            settings.DATA_PATH,
                            "Data directory")

tf.app.flags.DEFINE_boolean("no_horovod", False,
                            "Don't use Horovod. Single node training only.")
tf.app.flags.DEFINE_float("learningrate", settings.LEARNING_RATE,
                            "Learning rate")
tf.app.flags.DEFINE_boolean("use_upsampling", settings.USE_UPSAMPLING,
                        "True = Use upsampling; False = Use transposed convolution")

os.environ["KMP_BLOCKTIME"] = "0"
os.environ["KMP_AFFINITY"] = "granularity=thread,compact,1,0"
os.environ["OMP_NUM_THREADS"] = str(FLAGS.num_threads)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Get rid of the AVX, SSE warnings

config = tf.ConfigProto(intra_op_parallelism_threads=FLAGS.num_threads,
                        inter_op_parallelism_threads=FLAGS.num_inter_threads)

tf.logging.set_verbosity(tf.logging.INFO)

if not FLAGS.no_horovod:
    import horovod.tensorflow as hvd


def main(_):

    start_time = datetime.now()
    tf.logging.info("Starting at: {}".format(start_time))
    tf.logging.info("Batch size: {} images per step".format(FLAGS.batch_size))

    last_epoch_start_time = start_time

    # Load datasets
    imgs_train, msks_train, imgs_test, msks_test = load_datasets(FLAGS)

    if not FLAGS.no_horovod:
        # Initialize Horovod.
        hvd.init()

    # Define model
    model = define_model(imgs_train.shape, msks_train.shape, FLAGS)

    if not FLAGS.no_horovod:
        # Horovod: adjust learning rate based on number workers
        opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learningrate,
                                      epsilon=tf.keras.backend.epsilon())
        #opt = tf.train.RMSPropOptimizer(0.0001 * hvd.size())
        # tf.logging.info("HOROVOD: New learning rate is {}".\
        #         format(FLAGS.learningrate * hvd.size()))
    else:
        opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learningrate,
                                      epsilon=tf.keras.backend.epsilon())
        #opt = tf.train.RMSPropOptimizer(0.0001)

    # Wrap optimizer with Horovod Distributed Optimizer.
    if not FLAGS.no_horovod:
        tf.logging.info("HOROVOD: Wrapped optimizer")
        opt = hvd.DistributedOptimizer(opt)

    global_step = tf.train.get_or_create_global_step()
    train_op = opt.minimize(model["loss"], global_step=global_step)

    train_length = len(imgs_train)
    total_steps = (FLAGS.epochs * train_length) // FLAGS.batch_size
    if not FLAGS.no_horovod:
        last_step = total_steps // hvd.size()
        validation_steps = train_length // FLAGS.batch_size // hvd.size()
    else:
        last_step = total_steps
        validation_steps = train_length // FLAGS.batch_size

    def formatter_log(tensors):
        """
        Format the log output
        """
        if FLAGS.no_horovod:
            logstring = "Step {} of {}: " \
               " training Dice loss = {:.4f}," \
               " training Dice = {:.4f}".format(tensors["step"],
               last_step,
               tensors["loss"], tensors["dice"])
        else:
            logstring = "HOROVOD (Worker #{}), Step {} of {}: " \
               " training Dice loss = {:.4f}," \
               " training Dice = {:.4f}".format(
               hvd.rank(),
               tensors["step"],
               last_step,
               tensors["loss"], tensors["dice"])

        return logstring

    hooks = [

        tf.train.StopAtStepHook(last_step=last_step),

        # Prints the loss and step every log_steps steps
        tf.train.LoggingTensorHook(tensors={"step": global_step,
                                    "loss": model["loss"],
                                    "dice": model["metric_dice"]},
                                   every_n_iter=FLAGS.log_steps,
                                   formatter=formatter_log),
    ]

    # Horovod: BroadcastGlobalVariablesHook broadcasts
    # initial variable states from rank 0 to all other
    # processes. This is necessary to ensure consistent
    # initialization of all workers when training is
    # started with random weights
    # or restored from a checkpoint.
    if not FLAGS.no_horovod:
        hooks.append(hvd.BroadcastGlobalVariablesHook(0))

        # Horovod: save checkpoints only on worker 0 to prevent other workers from
        # corrupting them.
        if hvd.rank() == 0:
            checkpoint_dir = os.path.join(FLAGS.output_path,
                            "{}-workers".format(hvd.size()),
                            "{}".format(datetime.now().strftime("%Y%m%d-%H%M%S")))
        else:
            checkpoint_dir = None

    else:
        checkpoint_dir = os.path.join(FLAGS.output_path, "no_hvd",
                        "{}".format(datetime.now().strftime("%Y%m%d-%H%M%S")))

    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint,
    # and closing when done or an error occurs.
    current_step = 0
    startidx = 0
    epoch_idx = 0

    with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,
                                           hooks=hooks,
                                           save_summaries_steps=FLAGS.log_steps,
                                           log_step_count_steps=FLAGS.log_steps,
                                           config=config) as mon_sess:

        while not mon_sess.should_stop():

            # Run a training step synchronously.
            image_, mask_ = get_batch(imgs_train,
                                      msks_train,
                                      FLAGS.batch_size)

            # Do batch in order
            # stopidx = startidx + FLAGS.batch_size
            # if (stopidx > train_length):
            #     stopidx = train_length
            #
            # image_ = imgs_train[startidx:stopidx]
            # mask_  = msks_train[startidx:stopidx]

            mon_sess.run(train_op, feed_dict={model["input"]: image_,
                                              model["label"]: mask_})

            current_step += 1
            # # Get next batch (loop around if at end)
            # startidx += FLAGS.batch_size
            # if (startidx > train_length):
            #     startidx = 0

    stop_time = datetime.now()
    tf.logging.info("Stopping at: {}".format(stop_time))
    tf.logging.info("Elapsed time was: {}".format(stop_time-start_time))

if __name__ == "__main__":

    os.system("hostname")
    os.system("lscpu")
    tf.logging.info("{}".format(FLAGS.flag_values_dict()))
    tf.app.run()
