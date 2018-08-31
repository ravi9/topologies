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

FROM continuumio/anaconda 

# Update the installed packages
RUN apt-get -y update

# Add gcc cmake
RUN apt-get -y install cmake gcc build-essential

# Update conda 
RUN conda update -n base conda

# Make sure some packages are installed
RUN apt-get install -y bzip2 git vim nano

# Update pip
RUN pip install --upgrade pip

# Install Intel optimized frameworks

# TensorFlow with MKL
RUN conda install -y -c anaconda tensorflow 

# MX-Net
RUN pip install mxnet-mkl

# pyTorch
# Instructions from Ma, Mingfei <mingfei.ma@intel.com>
WORKDIR /opt
RUN conda install -y numpy pyyaml mkl mkl-include setuptools cmake cffi typing
RUN conda install -c mingfeima mkldnn
RUN git clone --recursive https://github.com/pytorch/pytorch 
RUN cd /opt/pytorch && CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" python setup.py install 

# Keras (no optimizations since FW is optimized)
RUN conda install -c anaconda -y keras

# Environment variables
ENV KMP_AFFINITY=granularity=fine,compact,1,0 KMP_BLOCKTIME=0 KMP_SETTINGS=0 

# Create working directory and start container there
WORKDIR /home/intel
CMD [ "/bin/bash" ]

