# UNet with Horovod

This repository contains code for multimodal brain tumor segmentation with U-Net. Horovod is utilized to specify the number of workers per node and coordinate single or multi-node training.  

## Setup

Horovod requires MPI to be installed globally on all nodes. Install it by running `sudo yum install openmpi` on each node.

Use conda to setup a virtual environment called 'tf' on all the nodes:
```
conda config --add channels intel
conda create -n tf -c intel python=2 pip numpy
```
Use `source activate tf` to enter the virtual environment, and install the following packages, beginning with `conda install -c anaconda tensorflow-mkl` to install the latest release of Intel Optimized TensorFlow with MKL-DNN. Additional packages can be installed using pip:
```
cython
psutil
SimpleITK
opencv-python
h5py
tqdm
keras
horovod
```

## Required data

Data files are not included in this public repo but can accessed by registering (using your institutional email address) at the following link: http://braintumorsegmentation.org/. Once access has been granted, you may download the raw data. To convert those datasets into numpy arrays having shape [num_images, x_dimension (128), y_dimension (128), num_channels] run `python converter.py` after changing its root_dir variable to point to the location your MICCAI_BraTS... folder (processing will take a few minutes). Once complete, the following four files will be saved in the directory specified by the OUT_PATH variable in `settings.py`.

```
-rw-r----- 1 bduser bduser  3250585696 Nov 14 11:42 imgs_test.npy
-rw-r----- 1 bduser bduser 13002342496 Nov 14 12:08 imgs_train.npy
-rw-r----- 1 bduser bduser   406323296 Nov 14 11:36 msks_test.npy
-rw-r----- 1 bduser bduser  1625292896 Nov 14 11:47 msks_train.npy
```

Copy these files to that same directory on all nodes.

## Customization

Once environments are constructed which meets the above requirements, clone this repo anywhere on the chief node.

Within the cloned directory, open `hosts.txt` and replace the current addresses with the appropriate addresses for your cluster.

Depending on your hardware, you may need to modify the NUM_INTRA_THREADS value. Set NUM_INTRA_THREADS to the number of physical cores available to the process, and NUM_INTER_THREADS threads to the number of disjoint branches in the computation graph (usually in the single digits).

Note that a natural consequence of synchronizing updates across several workers is a proportional decrease in the number of weight updates per epoch and slower convergence. To combat this slowdown and reduce the training time in multi-node execution, we use a warm-up strategy at the outset of training. The initial learning rate is defined in `settings.py`.

## Multi-Node Execution

Training is initiated with the `./run_multiworker_hvd.sh` command which:

```
1. Defines logdir, node_ips, num_workers_per_node, and num_inter_threads variables for passing to the execute script
2. Pulls hardware information and calculates values for ppr (processes per resource) and total num_processes 
3. Synchronizes the existing working directory with the corresponding directories on all other worker nodes
4. Sends the MPI command to initiate training on all the nodes via exec_multiworker.sh
```
Optionally, the following arguments may be passed to  `run_multiworker_hvd.sh` which will override the defaults: `<logidr> <hosts_file> <workers_per_node> <inter_op_threads>`.

The `exec_multiworker.sh` script then executes the following on each node:

```
1. Activates the tf virtual environment
2. Queries the core count and calculates the number of threads to pass to the TensorFlow script
3. Executes hvd_train.py
```

`hvd_train.py` references `settings.py` for its default learning rate and inter_op threads. 

When training completes, logs will be saved in the directory defined by the `logidir` argument passed into the `./run_multiworker_hvd.sh` script. If no `logidir` was specified, it will default to `tensorboard_multiworker`.

## Citations

Cite the following papers whenever using and/or refering to the BraTS datasets in your publications:

1. https://www.ncbi.nlm.nih.gov/pubmed/25494501
2. https://www.ncbi.nlm.nih.gov/pubmed/28872634
