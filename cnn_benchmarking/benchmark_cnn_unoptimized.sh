#!/bin/bash

sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

set -x

git clone https://github.com/tensorflow/benchmarks.git
   
cd benchmarks/scripts/tf_cnn_benchmarks

date > start_benchmark.txt

for network in googlenet inception3 resnet50 resnet152 vgg16 ; do
    for bz in 1 32 64 96 128; do
        echo -e "\n\n #### Starting $network and BZ=$bz ####\n\n"
        time python tf_cnn_benchmarks.py --forward_only \
        --data_format NHWC --model $network --batch_size $bz \
        --optimizer rmsprop --num_batches 100 2>&1 | \
        tee net_${network}_bz_${bz}.log
       echo -e "#### Finished $network and BZ=$bz ####"
 
  done
done

date > stop_benchmark.txt

echo "No Intel optimizations"
bash ../../../print_bench.sh
