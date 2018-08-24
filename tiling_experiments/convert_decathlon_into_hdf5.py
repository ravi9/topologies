
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

import numpy as np
import nibabel as nib   # Needed for reading Nifti files
import os
import glob
from tqdm import tqdm
import h5py

import argparse

parser = argparse.ArgumentParser(
               description="Convert Medicathlon data "
               "(http://medicaldecathlon.com/) "
               "files to HDF5 data file",
               add_help=True)

parser.add_argument("--data_dir",
                    default=os.path.join("/mnt/data/medical/decathlon/",
                                         "Task01_BrainTumour/"),
                    help="Root directory for the Medicathlon data files")

parser.add_argument("--save_dir",
                    default=os.path.join("/mnt/data/medical/decathlon/",
                                         "Task01_BrainTumour/",
                                         "decathlon_brain"),
                    help="Directory to save HDF5 data files")

parser.add_argument("--save_filename",
                    default=os.path.join("brain.h5"),
                    help="Save filename")

parser.add_argument("--split_ratio", type=int, default=0.85,
                    help="Split ratio train/test")

args = parser.parse_args()

def transform_mask_channels(msk):
    """
    Convert mask into separate channels.
    Otherwise, the value is the class ID for the mask.
    """
    newMsk = np.zeros(list(np.shape(msk))+[4])
    for classId in [1, 2, 3]:
         idx, idy, idz = np.where(msk==classId)
         for i in range(len(idx)):
             newMsk[idx[i],idy[i],idz[i],classId] = 1.0

    return np.rot90(newMsk)

def normalize_img(img):
    """
    Normalize images between 0 and 1
    """

    for idx in range(img.shape[3]):
        img[:,:,:,idx] = img[:,:,:,idx] / np.max(img[:,:,:,idx])

    return np.rot90(img)

def convert_files(dataDir, saveDir, saveFileName, split_ratio):

    """
    Find filenames for Medicathlon data.
    These are all listed under the subdirectories.
    Directory structure should be something like:
        Task01_BrainTumour/
             imagesTr          # Raw images (training set)
             imagesTs          # Raw images (testing set)
             labelsTr          # Masks (training set)
             labelsTs          # Masks (testing set)

    """
    imgFileNames = glob.glob(os.path.join(dataDir,
                             "imagesTr",
                             "BRATS*.nii.gz"))
    mskFileNames = [filename.replace("imagesTr", "labelsTr")
                    for filename in imgFileNames]

    fileNames = np.array([imgFileNames, mskFileNames])

    """
    First train image/mask
    """
    imgfilename = nib.load(fileNames[0,0])
    image_array = normalize_img(imgfilename.get_data())

    mskfilename = nib.load(fileNames[1,0])
    mask_array = transform_mask_channels(mskfilename.get_data())

    shapeImage = image_array.shape
    shapeMask = mask_array.shape

    maxshapeImage = (shapeImage[0],shapeImage[1],None,shapeImage[3])
    maxshapeMask = (shapeImage[0], shapeImage[1], None, 4)

    import pathlib
    pathlib.Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    saveFileName = os.path.join(saveDir, saveFileName)
    print("Saving HDF5 file to: {}".format(saveFileName))

    hdfFile = h5py.File(saveFileName, "w-")
    imgStackTrain = hdfFile.create_dataset("images/train",
                                      data=image_array,
                                      dtype=float,
                                      maxshape=maxshapeImage)
    mskStackTrain = hdfFile.create_dataset("masks/train",
                                      data=mask_array,
                                      dtype=float,
                                      maxshape=maxshapeMask)

    """
    First test image/mask
    """
    imgfilename = nib.load(fileNames[0,1])
    image_array = normalize_img(imgfilename.get_data())

    mskfilename = nib.load(fileNames[1,1])
    mask_array = transform_mask_channels(mskfilename.get_data())

    shapeImage = image_array.shape
    shapeMask = mask_array.shape

    maxshapeImage = (shapeImage[0],shapeImage[1],None,shapeImage[3])
    maxshapeMask = (shapeImage[0], shapeImage[1], None, 4)

    imgStackTest = hdfFile.create_dataset("images/test",
                                      data=image_array,
                                      dtype=float,
                                      maxshape=maxshapeImage)
    mskStackTest = hdfFile.create_dataset("masks/test",
                                      data=mask_array,
                                      dtype=float,
                                      maxshape=maxshapeMask)

    """
    Go through remaining files in directory and append to stack
    """
    for idx in tqdm(range(2,fileNames.shape[1])):

        imgfilename = nib.load(fileNames[0,idx])
        image_array = normalize_img(imgfilename.get_data())
        mskfilename = nib.load(fileNames[1,idx])
        mask_array = transform_mask_channels(mskfilename.get_data())

        # Assert that the array shape doesn't change.
        # Otherwise, dstack won't work
        assert(image_array.shape == shapeImage), \
            "File {}: Mismatch shape {}".format(fileNames[0,idx],
            image_array.shape)

        if (np.random.rand() < split_ratio):

            """
            Train cases
            """
            row = imgStackTrain.shape[2]
            extent = image_array.shape[2]
            imgStackTrain.resize(row+extent, axis=2) # Add new image
            imgStackTrain[:,:,row:(row+extent),:] = image_array

            row = mskStackTrain.shape[2]
            extent = mask_array.shape[2]
            mskStackTrain.resize(row+extent, axis=2) # Add new mask
            mskStackTrain[:,:,row:(row+extent),:] = mask_array

        else:

            """
            Test cases
            """
            row = imgStackTest.shape[2]
            extent = image_array.shape[2]
            imgStackTest.resize(row+extent, axis=2) # Add new image
            imgStackTest[:,:,row:(row+extent),:] = image_array

            row = mskStackTest.shape[2]
            extent = mask_array.shape[2]
            mskStackTest.resize(row+extent, axis=2) # Add new mask
            mskStackTest[:,:,row:(row+extent),:] = mask_array

    imgStackTrain.attrs["lshape"] = np.shape(imgStackTrain)
    mskStackTrain.attrs["lshape"] = np.shape(mskStackTrain)
    imgStackTest.attrs["lshape"] = np.shape(imgStackTest)
    mskStackTest.attrs["lshape"] = np.shape(mskStackTest)

if __name__ == "__main__":

    print("\n\n\nConverting Medical Decathlon raw "
            "Nifti files to HDF5 data files.")
    print("http://medicaldecathlon.com/")
    print("Looking for decathlon files in: {}".format(args.data_dir))
    print("\nConverting the training files.")
    # Convert the training data
    convert_files(args.data_dir, args.save_dir,
                  args.save_filename, args.split_ratio)
