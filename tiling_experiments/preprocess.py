
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

import os
import numpy as np
import settings


def load_data(data_path, prefix="_train"):
	imgs = np.load(os.path.join(data_path, "imgs"+prefix+".npy"),
						 mmap_mode="r", allow_pickle=False)
	msks = np.load(os.path.join(data_path, "msks"+prefix+".npy"),
						 mmap_mode="r", allow_pickle=False)

	return imgs, msks


def update_channels(imgs, msks, args):
	"""
	mode: int between 1-4
	"""

	imgs = imgs.astype("float32")
	msks = msks.astype("float32")

	shp = imgs.shape
	new_imgs = np.zeros((shp[0], shp[1], shp[2], args.num_input_channels))
	new_msks = np.zeros((shp[0], shp[1], shp[2], args.num_output_channels))

	if args.mode == 1:
		# Entire tumor (all 4 modalities combined)
		new_imgs[:, :, :, 0] = imgs[:, :, :, 2]
		new_msks[:, :, :, 0] = msks[:, :, :, 0] + \
			msks[:, :, :, 1]+msks[:, :, :, 2]+msks[:, :, :, 3]

	elif args.mode == 2:
		# Active tumor
		new_imgs[:, :, :, 0] = imgs[:, :, :, 0]
		new_msks[:, :, :, 0] = msks[:, :, :, 3]

	elif args.mode == 3:
		# Active core (necrosis, enchancing, non-enchancing)
		new_imgs[:, :, :, 0] = imgs[:, :, :, 1]
		new_msks[:, :, :, 0] = msks[:, :, :, 0] + \
			msks[:, :, :, 2]+msks[:, :, :, 3]

	elif args.mode == 4:
		# Entire tumor (all 4 modalities combined)
		# Use all input channels
		new_imgs[:, :, :, :] = imgs[:, :, :, :]
		new_msks[:, :, :, 0] = msks[:, :, :, 0] + \
			msks[:, :, :, 1]+msks[:, :, :, 2]+msks[:, :, :, 3]

	else:
		print(
			"Error mode must be 1, 2, or 3, 4 for "
			"entire tumor, active tumor, or active core")


	return new_imgs, new_msks
