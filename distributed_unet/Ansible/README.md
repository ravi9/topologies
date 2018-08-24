# UNet

UNet architecture for Multimodal Brain Tumor Segmentation, built with TensorFlow and optimized for multi-node execution on Intel KNL and Skylake servers.

## Overview

This repo contains code for multi-node execution:

	main.py: Multi-node implementation for synchronous weight updates, optimized for use on Intel KNL servers.

## Setup

The following virtual environment must be present on all worker and parameter server (PS) nodes. Use conda to setup a virtual environment called 'tf' with the following command:

```
conda create -n tf -c intel python=2 pip numpy
```

This will default the conda environment to use the Intel Python distribution. Use `source activate tf` to enter the virtual environment, then install the following packages:

```
Tensorflow 1.4.0
SimpleITK
opencv-python
h5py
tqdm
```

Outside of the virtual environment, you'll need Ansible to be installed on your parameter server. (e.g. `sudo yum install ansible -y`)

We use Intel optimized TensorFlow 1.4.0 for Python 2.7. Install instructions can be found at https://software.intel.com/en-us/articles/intel-optimized-tensorflow-wheel-now-available.

## Required Data

Data files are not included in this public repo but can accessed by registering (using your institutional email address) at the following link: http://braintumorsegmentation.org/. Once access has been granted, you may download the raw data. To convert those datasets into numpy arrays having shape [num_images, x_dimension (128), y_dimension (128), num_channels] run `python converter.py` after changing its root_dir variable to point to the location your MICCAI_BraTS... folder (processing will take a few minutes). Once complete, the following four files will be saved in the `/home/unet/data/slices/Results/` folder on the parameter server.

```
-rw-r----- 1 bduser bduser  3250585696 Nov 14 11:42 imgs_test.npy
-rw-r----- 1 bduser bduser 13002342496 Nov 14 12:08 imgs_train.npy
-rw-r----- 1 bduser bduser   406323296 Nov 14 11:36 msks_test.npy
-rw-r----- 1 bduser bduser  1625292896 Nov 14 11:47 msks_train.npy
```

Copy these files to `/home/unet/data/slices/Results/` on all the worker nodes as well.

## Modifications to settings file

Once environments are constructed which meets the above requirements, clone this repo anywhere on the parameter server.

Within the cloned directory 'unet', open `settings.py` and replace the current addresses:ports in the `ps_hosts` and `worker_hosts` lists with the appropriate addresses:ports for your cluster.

Depending on your hardware, you may need to modify the NUM_INTRA_THREADS value. This code was developed on Intel KNL and SKL servers having 68 and 56 cores each, so intra-op thread values of 57 and 50 were most ideal. Please note that maxing out the NUM_INTRA_THREADS value may result in segmentation faults or other memory issues. It is recommended to turn off hyper-threading to prevent resource exhaustion.

Note that a natural consequence of synchronizing updates across several workers is a proportional decrease in the number of weight updates per epoch and slower convergence. To combat this slowdown and reduce the training time in multi-node execution, we default to a large initial learning rate which decays as the model trains. This learning rate is also contained in `settings.py`.

We provide the following flags for modifying system variables in Multi-Node execution:

```
--use_upsampling    # Boolean, Use the UpSampling2D method in place of Conv2DTranspose (default: False)
--num_threads       # Int, Number of intra-op threads (default: # cores - 2)
--num_inter_threads # Int, Number of inter-op threads (default: 2)
--batch_size        # Int, Images per batch (default: 1024)
--blocktime         # Int, Set KMP_BLOCKTIME environment variable (default: 0)
--epochs            # Int, Number of epochs to train (default: 10)
--learningrate      # Float, Learning rate (default: 0.0005)
```

## Multi-Node Execution

We use an Ansible's playbook function to automate Multi-Node execution. This playbook will be run from the parameter server.
To initiate training, enter the command `./run_distributed_training.sh` in this cloned repo.

This command will run the `distributed_train.yml` playbook and initiate the following:

1. Create the `inv.yml` file from the addresses listed in `settings.py`.
2. Synchronize all files from the `unet` directory on the parameter server to the `unet` directories on the workers.
3. Start the parameter server with the following command:

```
Parameter Server:	numactl -p 1 python main.py
```

4. Run the `Distributed.sh` bash script on all the workers, which executes a run command on each worker:

```
Worker 0:	numactl -p 1 python main.py
Worker 1:	numactl -p 1 python main.py
Worker 2:	numactl -p 1 python main.py
Worker 3:	numactl -p 1 python main.py
```

5. While these commands are running, ansible registers their outputs (global step, training loss, dice score, etc.) and saves that to `training.log`.

To view training progress, as well as sets of images, predictions, and ground truth masks, direct your chrome browser to `http://your_chief_worker_address:6006/`. After a few moments, the webpage will populate and a series of training visualizations will become available. Explore the Scalars, Images, Graphs, Distributions, and Histograms tabs for detailed visualizations of training progress. You may need to create a SSH tunnel if port 6006 is not visible on your chief worker.

## Inference

To perform inference in TensorFlow with a model trained in a distributed setting, the model must be saved to a common location during training. We recommend saving to an NFS drive visible to all nodes in the cluster.

To perform inference on the test set (`imgs_test.npy` and `msks_test.npy`) update the `CHECKPOINT_DIRECTORY` variable in `settings.py` to reflect the location of the saved `.pb` model. Execute the following command to perform inference using the trained model:

```
python inference.py
```

When `inference.py` completes, the final dice coefficient will be printed to stdout and the predicted masks will be saved in the directory, specified by `OUT_PATH` in `settings.py`, under the name `msks_test_predictions.npy`. These prediction masks are the same dimensions as the input data and serve as helpful sanity checks when evaluating the quality of a trained model.

If performing inference on smaller subsets of the test data, or smaller test sets in general, the `batch_size` variable can be adjusted within `inference.py`.

## Sample Generation/Validation

If individual brain scans (each with 155 slices) are desired, use the `single_brain_samples.sh` bash script. Before running, modify the `data_dir`, `section`, `sample_dir` and `train_test_split` variables. To pick out individual samples, simply list them by name in the for loop, then run:

```
bash single_brain_samples.sh
```

This will output numpy files in an identical format to the original `msks/imgs_train.npy` and `msks/imgs_test.npy`, which may be used for training, testing or inference in the same manner.

## Citations

Whenever using and/or refering to the BraTS datasets in your publications, please make sure to cite the following papers.

1. https://www.ncbi.nlm.nih.gov/pubmed/25494501
2. https://www.ncbi.nlm.nih.gov/pubmed/28872634
