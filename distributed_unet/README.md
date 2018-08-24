# Distributed UNet

This repository contains code for running multi-node training in U-Net via Ansible or Horovod.

## Ansible Implementation

We use an Ansible's playbook function to run multi-node training with synchronous weight updates using a parameter server approach. To initiate training, follow instructions in the 'Ansible' directory.

## Horovod Implementation

We use Uber's Horovod MPI library to run multi-node training with multiple workers per node. We also provide functionality for running a single worker per node. To initiate training, follow instructions in the 'Horovod' directory.

## Citations

Whenever using and/or refering to the BraTS datasets in your publications, please make sure to cite the following papers.

1. https://www.ncbi.nlm.nih.gov/pubmed/25494501
2. https://www.ncbi.nlm.nih.gov/pubmed/28872634
