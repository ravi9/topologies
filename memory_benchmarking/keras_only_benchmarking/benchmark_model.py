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

import numpy as np
import os
import argparse
import psutil
import time
import datetime
import tensorflow as tf
from model import *

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model.signature_def_utils import predict_signature_def
from tensorflow.python.saved_model import tag_constants

parser = argparse.ArgumentParser(
	description="Benchmark 3D and 2D Convolution Models",add_help=True)
parser.add_argument("--dim_length",
					type = int,
					default=16,
					help="Tensor cube length of side")
parser.add_argument("--num_channels",
					type = int,
					default=1,
					help="Number of channels")
parser.add_argument("--num_outputs",
					type = int,
					default=1,
					help="Number of outputs")

parser.add_argument("--bz",
					type = int,
					default=1,
					help="Batch size")

parser.add_argument("--lr",
					type = float,
					default=0.001,
					help="Learning rate")

parser.add_argument("--num_datapoints",
					type = int,
					default=1024,
					help="Number of datapoints")
parser.add_argument("--epochs",
					type = int,
					default=3,
					help="Number of epochs")
parser.add_argument("--intraop_threads",
					type = int,
					default=psutil.cpu_count(logical=False),
					help="Number of intraop threads")
parser.add_argument("--interop_threads",
					type = int,
					default=2,
					help="Number of interop threads")
parser.add_argument("--blocktime",
					type = int,
					default=0,
					help="Block time for CPU threads")
parser.add_argument("--print_model",
					action="store_true",
					default=False,
					help="Print the summary of the model layers")
parser.add_argument("--use_upsampling",
					action="store_true",
					default=False,
					help="Use upsampling instead of transposed convolution")
parser.add_argument("--D2",
					action="store_true",
					default=False,
					help="Use 2D model and images instead of 3D.")
parser.add_argument("--single_class_output",
					action="store_true",
					default=False,
					help="Use binary classifier instead of U-Net")
parser.add_argument("--mkl_verbose",
					action="store_true",
					default=False,
					help="Print MKL debug statements.")
parser.add_argument("--inference",
					action="store_true",
					default=False,
					help="Test inference speed. Default=Test training speed")

args = parser.parse_args()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Get rid of the AVX, SSE warnings
if args.mkl_verbose:
	os.environ["MKL_VERBOSE"] = "1"  # Print out messages from MKL operations
	os.environ["MKLDNN_VERBOSE"] = "1"  # Print out messages from MKL-DNN operations
os.environ["OMP_NUM_THREADS"] = str(args.intraop_threads)
os.environ["KMP_BLOCKTIME"] = str(args.blocktime)
os.environ["KMP_AFFINITY"] = "granularity=thread,compact,1,0"


print("Started script on {}".format(datetime.datetime.now()))

print("args = {}".format(args))
print("OS: {}".format(os.system("uname -a")))
print("TensorFlow version: {}".format(tf.__version__))

#from tensorflow import keras as K
import keras as K

print("Keras API version: {}".format(K.__version__))

def get_model_memory_usage(batch_size, model):

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.backend.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.backend.count_params(p) for p in set(model.non_trainable_weights)])

    total_memory = 4.0*batch_size*(shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes

if args.D2:  # Define shape of the tensors (2D)
	dims = (1,2)
	tensor_shape = (args.dim_length,
					args.dim_length,
					args.num_channels)
	out_shape = (args.dim_length,
					args.dim_length,
					args.num_outputs)
else:        # Define shape of the tensors (3D)
	dims=(1,2,3)
	tensor_shape = (args.dim_length,
					args.dim_length,
					args.dim_length,
					args.num_channels)
	tensor_shape = (args.dim_length,
					args.dim_length,
					args.dim_length,
					args.num_outputs)

# Optimize CPU threads for TensorFlow
config = tf.ConfigProto(
		inter_op_parallelism_threads=args.interop_threads,
		intra_op_parallelism_threads=args.intraop_threads)

# Configure only as much GPU memory as needed during runtime
# Default is to use the entire GPU memory
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)

K.backend.set_session(sess)

if args.single_class_output:
	if args.D2:    # 2D convnet model
		pred, model = conv2D(tensor_shape,
					   print_summary=args.print_model, n_out=args.num_outputs,
					   return_model=True)
	else:			# 3D convet model
		pred, model = conv3D(tensor_shape,
					   print_summary=args.print_model, n_out=args.num_outputs,
					   return_model=True)
else:

	if args.D2:    # 2D U-Net model
		pred, model = unet2D(tensor_shape,
					   use_upsampling=args.use_upsampling,
					   print_summary=args.print_model, n_out=args.num_outputs,
					   return_model=True)
	else:			# 3D U-Net model
		pred, model = unet3D(tensor_shape,
					   use_upsampling=args.use_upsampling,
					   print_summary=args.print_model, n_out=args.num_outputs,
					   return_model=True)

# Freeze layers
if args.inference:
   for layer in model.layers:
       layer.trainable = False

#  Performance metrics for model
if args.single_class_output:
	model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

else:
	model.compile(loss=dice_coef_loss,
              optimizer="adam",
              metrics=[dice_coef, "accuracy"])


def get_imgs():

	# Just feed completely random data in for the benchmark testing
	sh = [args.bz] + list(tensor_shape)
	imgs = np.random.rand(*sh)

	while True:
		yield imgs

def get_batch():

	# Just feed completely random data in for the benchmark testing
	sh = [args.bz] + list(tensor_shape)

	imgs = np.random.rand(*sh)

	if args.single_class_output:
		truths = np.random.rand(args.bz, args.num_outputs)
	else:
		truths = np.random.rand(*sh)


	while True:
		yield imgs, truths

# Same number of sample to process regardless of batch size
# So if we have a larger batch size we can take fewer steps.
total_steps = args.num_datapoints//args.bz

print("Using random data.")
if args.inference:
	print("Testing inference speed.")
else:
	print("Testing training speed.")

print("Estimated memory for model = {} GB".format(get_model_memory_usage(args.bz, model)))

start_time = time.time()
if args.inference:
   for _ in range(args.epochs):
       model.predict_generator(get_imgs(), steps=total_steps, verbose=1)
else:
  	model.fit_generator(get_batch(), steps_per_epoch=total_steps,
					    epochs=args.epochs, verbose=1)

if args.inference:
   import shutil
   dirName = "./tensorflow_serving_model"
   if args.single_class_output:
      dirName += "_VGG16"
   else:
      dirName += "_UNET"
   if args.D2:
      dirName += "_2D"
   else:
      dirName += "_3D"

   shutil.rmtree(dirName, ignore_errors=True)
   # Save TensorFlow serving model
   builder = saved_model_builder.SavedModelBuilder(dirName)
   # Create prediction signature to be used by TensorFlow Serving Predict API
   signature = predict_signature_def(inputs={"images": model.input},
                                      outputs={"scores": model.output})
   # Save the meta graph and the variables
   builder.add_meta_graph_and_variables(sess=K.backend.get_session(), tags=[tag_constants.SERVING],
                                        signature_def_map={"predict": signature})

   builder.save()
   print("Saved TensorFlow Serving model to: {}".format(dirName))

stop_time = time.time()

print("\n\nTotal time = {:,.3f} seconds".format(stop_time - start_time))
print("Total images = {:,}".format(args.epochs*args.num_datapoints))
print("Speed = {:,.3f} images per second".format( \
			(args.epochs*args.num_datapoints)/(stop_time - start_time)))
