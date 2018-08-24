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
from multiprocessing import cpu_count

BASE = "/home/dir1/data/"
DATA_PATH = BASE
OUT_PATH  = BASE
IMG_HEIGHT = 128
IMG_WIDTH = 128

IN_CHANNEL_NO = 1
OUT_CHANNEL_NO = 1
CHANNELS_LAST = True

EPOCHS = 10

BLOCKTIME = 1
NUM_INTRA_THREADS = cpu_count() - 2
NUM_INTER_THREADS = 2
BATCH_SIZE = 1024

LEARNINGRATE = 0.0005  #0.0005

USE_UPSAMPLING = False  # True = Use upsampling; False = Use transposed convolution

#Use flair to identify the entire tumor: test reaches 0.78-0.80: MODE=1
#Use T1 Post to identify the active tumor: test reaches 0.65-0.75: MODE=2
#Use T2 to identify the active core (necrosis, enhancing, non-enh): test reaches 0.5-0.55: MODE=3
MODE=1

# Important that these are ordered correctly: [0] = chief node, [1] = worker node, etc.
PS_HOSTS = ["10.30.0.151"]
PS_PORTS = ["2222"]
WORKER_HOSTS = ["10.30.0.152","10.30.0.153","10.30.0.154"]
WORKER_PORTS = ["2222", "2222", "2222", "2222"]

CHECKPOINT_DIRECTORY = "/nfsshare/checkpoints/unet"
SAVED_MODEL_DIRECTORY = "/nfsshare/saved_models/unet"
TENSORBOARD_IMAGES = 3  # How many images to display on TensorBoard
LOG_SUMMARY_STEPS = 3 # Log summaries after these many steps

# TensorBoard
# To run TensorBoard you must log into the chief worker (first one in the WORKER_HOSTS list).
# Start your Tensorflow virtual environment and run `tensorboard --logdir=checkpoints`
# where checkpoints is whatever directory holds your log files.
# On your local machine (the one where you can run Chrome web browser), run
# the command: `ssh -f user1@123.45.67.890 -L 6006:localhost:6006 -N`
# where the `user1@123.45.67.890` is replaced with the username and IP of the chief worker.
# Then on the local machine start Chrome webbrowser and go to url  http://localhost:6006

# RegEx to exclude plots that match "test", "step", and "complete":  ^((?!test)(?!step)(?!complete).)*$
