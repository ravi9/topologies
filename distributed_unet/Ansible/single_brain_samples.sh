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

# Usage: ./single_brain_samples.sh
# Generates predicted masks on the test set with dim 128x128 and
# raw brain images at full resolution.


data_dir='/home/bduser/data_test/MICCAI_BraTS17_Data_Training/'
section='HGG/'
sample_dir='/home/bduser/data_test/'
train_test_split=0.80 # percent of the dataset to use for training
declare -i fin_msk_size=128 # must be 128 to work with current U-Net
declare -i fin_img_size=128 # can be larger than 128 if desired, is re-made after inference

source activate tf

# Add scans in list below to generate single brain samples
for sample in 'Brats17_TCIA_296_1' 'Brats17_TCIA_607_1'; do

        cp -r $data_dir$section$sample $sample_dir

        mkdir $sample_dir$sample/data

        time python sample_converter.py $data_dir $sample $train_test_split $fin_msk_size $sample_dir$sample/data/

        time python sample_inference.py $sample_dir$sample/data/

        rm $sample_dir$sample/data/*test.npy

        rm $sample_dir$sample/data/*train.npy

        time python sample_converter.py $data_dir $sample $train_test_split $fin_img_size $sample_dir$sample/data/

done

source deactivate
