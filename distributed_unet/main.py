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


# These are the only things you need to change.
# Just replace the IP addresses with whatever machines you want to distribute over
# Then run this script on each of those machines.
"""
Usage:  python main.py --ip=10.100.68.816 --is_sync=0
		for asynchronous TF
		python main.py --ip=10.100.68.816 --is_sync=1
		for synchronous updates
		The IP address must match one of the ones in the list below. If not passed,
		then we"ll default to the current machine"s IP (which is usually correct unless you use OPA)
"""
import settings

ps_hosts = settings.PS_HOSTS
ps_ports = settings.PS_PORTS
worker_hosts = settings.WORKER_HOSTS
worker_ports = settings.WORKER_PORTS

ps_list = ["{}:{}".format(x, y) for x, y in zip(ps_hosts, ps_ports)]
worker_list = [
	"{}:{}".format(x, y) for x, y in zip(worker_hosts, worker_ports)
]
print("Distributed TensorFlow training")
print("Parameter server nodes are: {}".format(ps_list))
print("Worker nodes are {}".format(worker_list))

####################################################################

import numpy as np
import tensorflow as tf
import os
import socket

# Fancy progress bar
from tqdm import tqdm
from tqdm import trange

from model import define_model, validate_model
from data import load_datasets, get_epoch
import multiprocessing
import subprocess
import signal

# Unset proxy env variable to avoid gRPC errors
del os.environ["http_proxy"]
del os.environ["https_proxy"]

# You can turn on the gRPC messages by setting the environment variables below
#os.environ["GRPC_VERBOSITY"]="DEBUG"
#os.environ["GRPC_TRACE"] = "all"

# Define parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float("learning_rate", settings.LEARNINGRATE,
						  "Initial learning rate.")
tf.app.flags.DEFINE_integer("is_sync", 1, "Synchronous updates?")
tf.app.flags.DEFINE_integer("inter_op_threads", settings.NUM_INTER_THREADS,
							"# inter op threads")
tf.app.flags.DEFINE_integer("intra_op_threads", settings.NUM_INTRA_THREADS,
							"# intra op threads")
tf.app.flags.DEFINE_string("ip", socket.gethostbyname(socket.gethostname()),
						   "IP address of this machine")
tf.app.flags.DEFINE_integer("batch_size", settings.BATCH_SIZE,
							"Batch size of input data")
tf.app.flags.DEFINE_integer("epochs", settings.EPOCHS,
							"Number of epochs to train")

tf.app.flags.DEFINE_boolean("use_upsampling", settings.USE_UPSAMPLING,
							"True = Use upsampling; False = Use transposed convolution")

tf.app.flags.DEFINE_integer("LOG_SUMMARY_STEPS", settings.LOG_SUMMARY_STEPS,
							"How many steps per writing summary log")

if (FLAGS.ip in ps_hosts):
	job_name = "ps"
	task_index = ps_hosts.index(FLAGS.ip)
elif (FLAGS.ip in worker_hosts):
	job_name = "worker"
	task_index = worker_hosts.index(FLAGS.ip)
else:
	print(
		"Error: IP {} not found in the worker or ps node list.\nUse --ip= to specify which machine this is.".
		format(FLAGS.ip))
	exit()

os.environ["KMP_BLOCKTIME"] = str(settings.BLOCKTIME)
os.environ["KMP_AFFINITY"] = "granularity=thread,compact,1,0"
os.environ["OMP_NUM_THREADS"] = str(FLAGS.intra_op_threads)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Get rid of the AVX, SSE warnings

CHECKPOINT_DIRECTORY = settings.CHECKPOINT_DIRECTORY
if FLAGS.use_upsampling:
	method_up = "upsample2D"
else:
	method_up = "conv2DTranspose"
# CHECKPOINT_DIRECTORY = CHECKPOINT_DIRECTORY + "unet," + \
# 			"lr={},{},intra={},inter={}".format(FLAGS.learning_rate,
# 			method_up, num_intra_op_threads,
# 			num_inter_op_threads)

def main(_):

	config = tf.ConfigProto(
		inter_op_parallelism_threads=FLAGS.inter_op_threads,
		intra_op_parallelism_threads=FLAGS.intra_op_threads)

	run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
	run_metadata = tf.RunMetadata()  # For Tensorflow trace

	cluster = tf.train.ClusterSpec({"ps": ps_list, "worker": worker_list})

	server = tf.train.Server(cluster,
		  job_name=job_name,
		  task_index=task_index,
		  config=config)

	is_chief = (task_index == 0)  # Am I the chief node (always task 0)

	if job_name == "ps":

		print("*" * 30)
		print("\nParameter server #{} on {}.\n\n" \
		 "Waiting for the workers to finish.\n\n" \
		 "Press CTRL-\\ to terminate early.\n"  \
		 .format(task_index, ps_hosts[task_index]))
		print("*" * 30)

		server.join()

	elif job_name == "worker":

		if is_chief:
			print("I am the chief worker {} with task #{}".format(
				worker_hosts[task_index], task_index))
		else:
			print("I am worker {} with task #{}".format(
				worker_hosts[task_index], task_index))

		if len(ps_list) > 0:

			setDevice = tf.train.replica_device_setter(
					 worker_device="/job:worker/task:{}".format(task_index),
					 ps_tasks=len(ps_hosts),
					 cluster=cluster)
		else:
			setDevice = "/cpu:0" # No parameter server so put variables on chief worker

		with tf.device(setDevice):

			"""
			BEGIN: DEFINE DATA LOADER
			Loading the data into 2 dictionaries: training and validation
			"""
			training_data, validation_data = load_datasets()

			print("Loading epoch")
			epoch = get_epoch(FLAGS.batch_size, training_data["input"],
							  training_data["label"])
			training_data["num_batches"] = len(epoch)
			print("Loaded")

			validation_data["num_batches"] = validation_data["length"] // FLAGS.batch_size

			"""
			END: DEFINE DATA LOADER
			"""

			"""
			BEGIN: Define our model
			All of the model definitions are in the file model.py
			In this case, model is a dictionary containing
			the model, operations, and summaries.
			"""
			model = define_model(FLAGS, training_data["input"].shape,
								 training_data["input"].shape,
								 len(worker_hosts))
			"""
			END: Define our model
			"""


		# Session
		# The StopAtStepHook handles stopping after running given steps.
		# We'll just set the number of steps to be the # of batches * epochs
		hooks = [tf.train.StopAtStepHook(last_step=training_data["num_batches"] * FLAGS.epochs)]

		# Only the chief does the summary
		if is_chief:
			summary_op = tf.summary.merge_all()
			summary_hook = tf.train.SummarySaverHook(output_dir=CHECKPOINT_DIRECTORY,
							summary_op=summary_op, save_steps=FLAGS.LOG_SUMMARY_STEPS)
			chief_only_hooks=[summary_hook]
		else:
			chief_only_hooks = None

		# For synchronous SGD training.
		# This creates the hook for the MonitoredTrainingSession
		# So that the worker nodes will wait for the next training step
		# (which is signaled by the chief worker node)
		if FLAGS.is_sync:
			sync_replicas_hook = model["optimizer"].make_session_run_hook(is_chief)
			hooks.append(sync_replicas_hook)

		with tf.train.MonitoredTrainingSession(master = server.target,
				is_chief=is_chief,
				config=config,
				hooks=hooks,
				chief_only_hooks=chief_only_hooks,
				save_summaries_steps=FLAGS.LOG_SUMMARY_STEPS,
				log_step_count_steps=FLAGS.LOG_SUMMARY_STEPS,
				checkpoint_dir=CHECKPOINT_DIRECTORY) as sess:

			progressbar = trange(training_data["num_batches"] * FLAGS.epochs)
			step = 0

			while not sess.should_stop():

				batch_idx = step // training_data["num_batches"] # Which batch is the epoch?

				data = epoch[batch_idx, 0]
				labels = epoch[batch_idx, 1]

				# For n workers, break up the batch into n sections
				# Send each worker a different section of the batch
				data_range = int(FLAGS.batch_size / len(worker_hosts))
				start = data_range * task_index

				end = start + data_range
				# Make sure we don't go over
				if (end >= training_data["length"]):
					end = training_data["length"] - 1

				feed_dict = {model["input"]: data[start:end],
							 model["label"]: labels[start:end]}

				history, loss, dice, step = sess.run(
					[model["train_op"], model["loss"], model["metric_dice"],
					model["global_step"]],
					feed_dict=feed_dict)

				# Print the loss and dice metric in the progress bar.
				progressbar.set_description(
					"(loss={:.3f}, dice={:.3f})".format(loss, dice))
				progressbar.n = step

				"""
				Validation
				"""
				# Calculate metric on test dataset every epoch
				if ((step % training_data["num_batches"]) == 0) and (step > 1):

					if is_chief: # Only valiate on the chief worker
						validate_model(FLAGS, sess, model, validation_data,
									   batch_idx)

					print("Shuffling epoch")
					epoch = get_epoch(FLAGS.batch_size,
									  training_data["input"],
									  training_data["label"])


		print("\n\nFinished work on this node.")


if __name__ == "__main__":

	tf.app.run()