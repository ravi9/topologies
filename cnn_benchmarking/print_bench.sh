#!/bin/bash
#Prints FPS from from the logs emitted from benchmark_cnn.sh script.

echo -e "\n Net BZ FPS \n"

for network in googlenet inception3 resnet50 resnet152 vgg16 ; do
  for bz in 1 32 64 96 128; do
    fps=$(grep  "total images/sec:"  net_${network}_bz_${bz}.log | cut -d ":" -f2 | xargs)
    echo "$network $bz $fps"
  done
    echo -e "\n"
done

echo Benchmark started  at: `cat start_benchmark.txt`
echo Benchmark finished at: `cat stop_benchmark.txt`
