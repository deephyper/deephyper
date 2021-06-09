Running on ThetaGPU (ALCF)
**************************

It is important to run the following commands from the appropriate storage space because some features of DeepHyper can generate a consequante quantity of data such as model checkpointing. The storage spaces available at the ALCF are:

- ``/lus/grand/projects/...``
- ``/lus/eagle/projects/...``
- ``/lus/theta-fs0/projects/...``

Then create a script named ``SetUpEnv.sh`` and adapt the path to activate your installed DeepHyper conda environment ``PATH_TO_ENV``:

.. code-block:: bash
    :caption: SetUpEnv.sh

    #!/bin/bash

    . /etc/profile

    # Tensorflow optimized for A100 with CUDA 11
    module load conda/tensorflow/2020-11-11

    # Activate conda env
    conda activate $PATH_TO_ENV/dhgpu/

.. note::

    This ``SetUpEnv.sh`` script can be very useful to tailor the execution's environment to your needs. Here are a few tips that can be useful:

      - To activate XLA optimized compilation add  ``export TF_XLA_FLAGS=--tf_xla_enable_xla_devices``
      - To change the log level of Tensorflow add ``export TF_CPP_MIN_LOG_LEVEL=3``


Manually start a Ray cluster
============================

Single Node Cluster
-------------------

Create a ``SingleNodeRayCluster.sh`` script and adapt the value of ``CURRENT_DIR`` which is the path of the folder containing ``SetUpEnv.sh``:

.. code-block:: bash
    :caption: SingleNodeRayCluster.sh

    #!/bin/bash

    # USER CONFIGURATION
    CURRENT_DIR=...
    CPUS_PER_NODE=8
    GPUS_PER_NODE=8

    # Script to launch Ray cluster

    ACTIVATE_PYTHON_ENV="${CURRENT_DIR}/SetUpEnv.sh"
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
        --num-cpus $CPUS_PER_NODE --num-gpus $GPUS_PER_NODE --block" &

    # optional, though may be useful in certain versions of Ray < 1.0.
    sleep 10


Multiple Nodes Cluster
----------------------

Create a ``MultiNodeRayCluster.sh`` script and adapt the value of ``CURRENT_DIR`` which is the path of the folder containing ``SetUpEnv.sh``:

.. code-block:: bash
    :caption: MultiNodeRayCluster.sh

    #!/bin/bash

    # USER CONFIGURATION
    CURRENT_DIR=...
    CPUS_PER_NODE=8
    GPUS_PER_NODE=8

    # Script to launch Ray cluster

    ACTIVATE_PYTHON_ENV="${CURRENT_DIR}/SetUpEnv.sh"
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
        --num-cpus $CPUS_PER_NODE --num-gpus $GPUS_PER_NODE --block" &

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
            --num-cpus $CPUS_PER_NODE --num-gpus $GPUS_PER_NODE" --block &
        sleep 5
    done


Execution of the search
=======================

Manual Execution
----------------

Once a cluster is started you can execute the search with for example 1 GPU per evaluation with the following command:

.. code-block:: console

    deephyper nas random --evaluator ray --ray-address auto --problem deephyper.benchmark.nas.mnist1D.problem.Problem --max-evals 10 --num-cpus-per-task 1 --num-gpus-per-task 1

Automatic Execution
-------------------

DeepHyper provides the ``deephyper ray-submit`` command interface to automatically generate and submit a submission script for the COBALT scheduler of ThetaGPU. This interface follows some of the argument available with the ``qsub`` command such as ``-n`` (number of nodes), ``-t`` (time in minutes), ``-A`` (project name) and ``-q`` (queue name). An example command is:

.. code-block:: console

    deephyper ray-submit nas agebo -w mnist_1gpu_2nodes_60 -n 2 -t 60 -A $PROJECT_NAME -q full-node --problem deephyper.benchmark.nas.mnist1D.problem.Problem --run deephyper.nas.run.alpha.run --max-evals 10000 --num-cpus-per-task 1 --num-gpus-per-task 1 -as $PATH_TO_SETUP --n-jobs 16

.. warning::

    The ``--n-jobs`` argument is the number of parallel processes that the surrogate model of DeepHyper's Bayesian optimisation can use. This argument does not have any link with the number of available workers.