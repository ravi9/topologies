#!/bin/bash

# Runs the TF CNN Benchmarks using Intel optimizations

git clone -b mkl_experiment https://github.com/tensorflow/benchmarks.git
   
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1
export KMP_SETTINGS=1
export OMP_PROC_BIND=true

export bz1_num_cores=8
export num_cores=`grep -c ^processor /proc/cpuinfo`

cd benchmarks/scripts/tf_cnn_benchmarks

sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
set -x

date > start_benchmark.txt

for network in googlenet inception3 resnet50 resnet152 ; do

    inter=2
    bz=1
    OMP_NUM_THREADS=$bz1_num_cores
    echo -e "\n\n #### Starting $network and BZ=$bz ####\n\n"

    python tf_cnn_benchmarks.py --device cpu --forward_only True \
    --data_format NCHW --cpu skl --data_name synthetic \
    --model $network --learning_rate 0.001 --batch_size $bz \
    --optimizer rmsprop --num_intra_threads $OMP_NUM_THREADS \
    --num_inter_threads $inter --num_omp_threads $OMP_NUM_THREADS \
    --num_batches 100 2>&1 | \
    tee net_${network}_bz_${bz}.log

    echo -e "#### Finished $network and BZ=$bz ####"

    OMP_NUM_THREADS=$num_cores

    for bz in 32 64 96 128; do
        echo -e "\n\n #### Starting $network and BZ=$bz ####\n\n"

	python tf_cnn_benchmarks.py --device cpu --forward_only True \
        	--data_format NCHW --cpu skl --data_name synthetic \
        	--model $network --learning_rate 0.001 --batch_size $bz \
        	--optimizer rmsprop --num_intra_threads $OMP_NUM_THREADS \
        	--num_inter_threads $inter --num_omp_threads $OMP_NUM_THREADS \
        	--num_batches 100 2>&1 | \
        	tee net_${network}_bz_${bz}.log

        echo -e "#### Finished $network and BZ=$bz ####"

    done
done

for network in vgg16 ; do

    inter=1
    num_cores=8
    bz=1
    OMP_NUM_THREADS=$bz1_num_cores
    echo -e "\n\n #### Starting $network and BZ=$bz ####\n\n"

    python tf_cnn_benchmarks.py --device cpu --forward_only True \
    --data_format NCHW --cpu skl --data_name synthetic \
    --model $network --learning_rate 0.001 --batch_size $bz \
    --optimizer rmsprop --num_intra_threads $OMP_NUM_THREADS \
    --num_inter_threads $inter --num_omp_threads $OMP_NUM_THREADS \
    --num_batches 100 2>&1 | \
    tee net_${network}_bz_${bz}.log

    echo -e "#### Finished $network and BZ=$bz ####"

    OMP_NUM_THREADS=$num_cores

    for bz in 32 64 96 128; do
        echo -e "\n\n #### Starting $network and BZ=$bz ####\n\n"

        python tf_cnn_benchmarks.py --device cpu --forward_only True \
        --data_format NCHW --cpu skl --data_name synthetic \
        --model $network --learning_rate 0.001 --batch_size $bz \
        --optimizer rmsprop --num_intra_threads $OMP_NUM_THREADS \
        --num_inter_threads $inter --num_omp_threads $OMP_NUM_THREADS \
        --num_batches 100 2>&1 | \
        tee net_${network}_bz_${bz}.log

        echo -e "#### Finished $network and BZ=$bz ####"

    done
done

date > stop_benchmark.txt

bash ../../../print_bench.sh
