#!/bin/bash

pip install memory_profiler
rm *.dat 
rm *.log

using_gpu=${1:-True}

for dim_length in 32 56 64 80 128 184 200 256 320 400 480 512 600
do

   num=1000000  # Run for a long time because timeout will automatically stop the script after a certain number of seconds
   secs=600  # Number of seconds to record memory

   # Training batch size 1
   if [ $using_gpu == True ]; then
   	timeout 5 bash check_gpu_memory.sh > gpu_memory_${dim_length}_bz1_train.log
   
   	timeout $secs mprof run benchmark_model.py --dim_length $dim_length \
  		 --num_datapoints 5 --epochs $num --bz 1 \
        	 2>&1 | tee train_unet_${dim_length}_bz1.log &

   	timeout $secs bash check_gpu_memory.sh > gpu_memory_${dim_length}_bz1_train.log

   else

	timeout $secs mprof run benchmark_model.py --dim_length $dim_length \
               	 --num_datapoints 5 --epochs $num --bz 1 \
                 2>&1 | tee train_unet_${dim_length}_bz1.log
   fi

   pattern="mprofile_*.dat"
   files=( $pattern )
   mv ${files[0]}  unet3d_train_len${dim_length}_bz1.dat
 
   bash clear_caches.sh

   # Training batch size 2
   if [ $using_gpu == True ]; then
   	timeout 5 bash check_gpu_memory.sh > gpu_memory_${dim_length}_bz2_train.log
  
   	timeout $secs mprof run  benchmark_model.py --dim_length $dim_length \
		--num_datapoints 5 --epochs $num --bz 2 \
         	2>&1 | tee train_unet_${dim_length}_bz2.log &

   	timeout $secs bash check_gpu_memory.sh > gpu_memory_${dim_length}_bz2_train.log

   else
	timeout $secs mprof run  benchmark_model.py --dim_length $dim_length \
               	--num_datapoints 5 --epochs $num --bz 2 \
               	2>&1 | tee train_unet_${dim_length}_bz2.log
   fi

   pattern="mprofile_*.dat"
   files=( $pattern )
   mv ${files[0]} unet3d_train_len${dim_length}_bz2.dat

   bash clear_caches.sh

   # Inference batch size 1
   if [ $using_gpu == True ]; then
   	timeout 5 bash check_gpu_memory.sh > gpu_memory_${dim_length}_bz1_inference.log
   
   	timeout $secs mprof run benchmark_model.py --dim_length $dim_length \
		--inference --num_datapoints 5 --epochs $num --bz 1 \
        	 2>&1 | tee inference_unet_${dim_length}_bz1.log &

   	timeout $secs bash check_gpu_memory.sh > gpu_memory_${dim_length}_bz1_inference.log
   else
	timeout $secs mprof run benchmark_model.py --dim_length $dim_length \
               	--inference --num_datapoints 5 --epochs $num --bz 1 \
                 2>&1 | tee inference_unet_${dim_length}_bz1.log
   fi

   pattern="mprofile_*.dat"
   files=( $pattern )
   mv ${files[0]} unet3d_inference_len${dim_length}_bz1.dat

   bash clear_caches.sh

done

clear
echo "All done. Remember to copy the logs."
