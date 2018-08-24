#!/bin/bash
# To run: bash run_multiworker_hvd.sh <logdir> <hostfile> <workers per node> <inter op threads>
# Note: The total number of workers deployed will be the number of workers per node * number of nodes

if [ "$1" == "-h" ]; then
  echo "Usage: `basename $0` [<logdir>] [<hostfile>] [<workers per node>] [<inter op threads>]"
  echo "   where "
  echo "   [<logdir>]    = Directory for the TensorFlow/TensorBoard logs"
  echo "   [<hostfile>]  = File name for the node IP list (comma separated list)"
  echo "   [<workers per node>] = Number of workers per node"
  echo "   [<inter op threads>] = Number of inter-op-parallelism threads for TensorFlow"
  echo " "
  exit 0
fi

logdir=${1:-_multiworker}     # Default suffix is _multiworker
node_ips=${2:-hosts.txt}      # Default is the hosts.txt file
export num_workers_per_node=${3:-2}  # Default 4 workers per node
export num_inter_threads=${4:-2} # Default to 2 inter_op threads

export physical_cores=`lscpu | grep "Core(s) per socket" | cut -d':' -f2 | sed "s/ //g"` # Total number of physical cores per socket
export num_nodes=`awk -F, '{print NF}' ${node_ips} | head -1 ` # Hosts.txt should contain IP addresses separated by commas
export num_sockets=`lscpu | grep "Socket(s)" | cut -d':' -f2 | sed "s/ //g"`   # Number of sockets per node
export logical_cores=`nproc`

export num_processes=$(( $num_nodes * $num_workers_per_node )) # Total number of workers across all nodes
export ppr=$(( $num_workers_per_node / $num_sockets ))
export pe=$(( $physical_cores / $ppr ))

echo "Running $num_workers_per_node worker(s)/node on $num_nodes nodes..."

echo Using $num_processes total workers.
echo nodes are: `cat $node_ips`

mpirun --mca btl_tcp_if_include eth0 -np $num_processes -H `cat $node_ips` --map-by socket -cpus-per-proc $physical_cores --report-bindings --oversubscribe bash exec_multiworker.sh $logdir $ppr $num_inter_threads
#mpirun -np $num_processes -H `cat $node_ips` --map-by socket -cpus-per-proc $physical_cores --report-bindings --oversubscribe bash exec_multiworker.sh $logdir $ppr $num_inter_threads
