Running on ThetaGPU (ALCF)
**************************

Create a script ``SetUp.sh`` and adapt the path to activate your DeepHyper conda environment ``PATH_TO_ENV``::

    #!/bin/bash

    # Tensorflow optimized for A100 with CUDA 11
    source /lus/theta-fs0/software/thetagpu/conda/tf_master/2020-11-11/mconda3/setup.sh

    # Activate conda env
    conda activate $PATH_TO_ENV/dhgpu/


**If you want to run on a singe node**

Create a ``SingleNodeRayCluster.sh`` script and adapt the value of ``CURRENT_DIR`` which is the path of the folder containing ``SetUp.sh``::

    #!/bin/bash

    # USER CONFIGURATION
    CURRENT_DIR=...
    CPUS_PER_TASK=8
    GPUS_PER_TASK=8

    # Script to launch Ray cluster

    ACTIVATE_PYTHON_ENV="${CURRENT_DIR}SetUpEnv.sh"
    echo "Script to activate Python env: $ACTIVATE_PYTHON_ENV"

    head_node=$HOSTNAME
    head_node_ip=$(dig $head_node a +short | awk 'FNR==2')

    # if we detect a space character in the head node IP, we'll
    # convert it to an ipv4 address. This step is optional.
    if [[ "$head_node_ip" == *" "* ]]; then
    IFS=' ' read -ra ADDR <<<"$head_node_ip"
    if [[ ${#ADDR[0]} -gt 16 ]]; then
    head_node_ip=${ADDR[1]}
    else
    head_node_ip=${ADDR[0]}
    fi
    echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
    fi

    # Starting the Ray Head Node
    port=6379
    ip_head=$head_node_ip:$port
    export ip_head
    echo "IP Head: $ip_head"

    echo "Starting HEAD at $head_node"
    ssh -tt $head_node_ip "source $ACTIVATE_PYTHON_ENV; \
        ray start --head --node-ip-address=$head_node_ip --port=$port \
        --num-cpus $CPUS_PER_TASK --num-gpus $GPUS_PER_TASK --block" &

    # optional, though may be useful in certain versions of Ray < 1.0.
    sleep 10


**If you want to run on multiple nodes**

Create a ``MultiNodeRayCluster.sh`` script and adapt the value of ``CURRENT_DIR`` which is the path of the folder containing ``SetUp.sh``::

    #!/bin/bash

    # USER CONFIGURATION
    CURRENT_DIR=...
    CPUS_PER_TASK=8
    GPUS_PER_TASK=8

    # Script to launch Ray cluster

    ACTIVATE_PYTHON_ENV="${CURRENT_DIR}SetUpEnv.sh"
    echo "Script to activate Python env: $ACTIVATE_PYTHON_ENV"


    # Getting the node names
    mapfile -t nodes_array -d '\n' < $COBALT_NODEFILE

    head_node=${nodes_array[0]}
    head_node_ip=$(dig $head_node a +short | awk 'FNR==2')

    # if we detect a space character in the head node IP, we'll
    # convert it to an ipv4 address. This step is optional.
    if [[ "$head_node_ip" == *" "* ]]; then
    IFS=' ' read -ra ADDR <<<"$head_node_ip"
    if [[ ${#ADDR[0]} -gt 16 ]]; then
    head_node_ip=${ADDR[1]}
    else
    head_node_ip=${ADDR[0]}
    fi
    echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
    fi

    # Starting the Ray Head Node
    port=6379
    ip_head=$head_node_ip:$port
    export ip_head
    echo "IP Head: $ip_head"

    echo "Starting HEAD at $head_node"
    ssh -tt $head_node_ip "source $ACTIVATE_PYTHON_ENV; \
        ray start --head --node-ip-address=$head_node_ip --port=$port \
        --num-cpus $CPUS_PER_TASK --num-gpus $GPUS_PER_TASK --block" &

    # optional, though may be useful in certain versions of Ray < 1.0.
    sleep 10

    # number of nodes other than the head node
    worker_num=$((${#nodes_array[*]} - 1))
    echo "$worker_num workers"

    for ((i = 1; i <= worker_num; i++)); do
        node_i=${nodes_array[$i]}
        node_i_ip=$(dig $node_i a +short | awk 'FNR==1')
        echo "Starting WORKER $i at $node_i with ip=$node_i_ip"
        ssh -tt $node_i_ip "source $ACTIVATE_PYTHON_ENV; \
            ray start --address $ip_head \
            --num-cpus $CPUS_PER_TASK --num-gpus $GPUS_PER_TASK" --block &
        sleep 5
    done


**Execution of the search**

Execute::

    deephyper nas random --evaluator ray --ray-address auto --problem deephyper.benchmark.nas.mnist1D.problem.Problem --max-evals 10 --num-cpus-per-task 1 --num-gpus-per-task 1