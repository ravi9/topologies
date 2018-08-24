while true
do
   nvidia-smi -i 0 --query-gpu=timestamp,memory.total,memory.free,memory.used --format=csv | tail -n 1
   sleep 1
done

