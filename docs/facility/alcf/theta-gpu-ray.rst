.. _tutorial-alcf-02:

Execution on the ThetaGPU supercomputer (with Ray)
**************************************************

In this tutorial we are going to learn how to use DeepHyper on the **ThetaGPU** supercomputer at the ALCF using Ray. `ThetaGPU <https://www.alcf.anl.gov/support-center/theta/theta-thetagpu-overview>`_ is a 3.9 petaflops system based on NVIDIA DGX A100.

Submission Script
=================

This section of the tutorial will show you how to submit script to the COBALT scheduler of ThetaGPU. To execute DeepHyper on ThetaGPU with a submission script it is required to:

1. Define a Bash script to initialize the environment (e.g., load a module, activate a conda environment).
2. Define a script composed of 3 steps: (1) launch a Ray cluster on available ressources, (2) execute a Python application which connects to the Ray cluster, and (3) stop the Ray cluster.

Start by creating a script named ``activate-dhenv.sh`` to initialize your environment. It will be used to initialize each used compute node. Replace the ``$CONDA_ENV_PATH`` by your personnal conda installation (e.g., it can be replaced by ``base`` if no virtual environment is used):


.. code-block:: bash
    :caption: **file**: ``activate-dhenv.sh``

    #!/bin/bash

    # Necessary for Bash shells
    . /etc/profile

    # Tensorflow optimized for A100 with CUDA 11
    module load conda/2022-07-01

    # Activate conda env
    conda activate $CONDA_ENV_PATH

.. tip::

    This ``activate-dhenv`` script can be very useful to tailor the execution's environment to your needs. Here are a few tips that can be useful:

    - To activate XLA optimized compilation add ``export TF_XLA_FLAGS=--tf_xla_enable_xla_devices``
    - To change the log level of Tensorflow add ``export TF_CPP_MIN_LOG_LEVEL=3``


Then create a new file named ``job-deephyper.sh`` and make it executable. It will correspond to your submission script.

.. code-block:: bash

    $ touch job-deephyper.sh && chmod +x job-deephyper.sh

Add the following content:

.. code-block:: bash
    :caption: **file**: ``job-deephyper.sh``

    #!/bin/bash
    #COBALT -q full-node
    #COBALT -n 2
    #COBALT -t 20
    #COBALT -A $PROJECT_NAME
    #COBALT --attrs filesystems=home,grand,eagle,theta-fs0

    # User Configuration
    EXP_DIR=$PWD
    INIT_SCRIPT=$PWD/activate-dhenv.sh
    CPUS_PER_NODE=8
    GPUS_PER_NODE=8

    # Initialization of environment
    source $INIT_SCRIPT

    # Getting the node names
    mapfile -t nodes_array -d '\n' < $COBALT_NODEFILE

    head_node=${nodes_array[0]}
    head_node_ip=$(dig $head_node a +short | awk 'FNR==2')

    # Starting the Ray Head Node
    port=6379
    ip_head=$head_node_ip:$port
    export ip_head
    echo "IP Head: $ip_head"

    echo "Starting HEAD at $head_node"
    ssh -tt $head_node_ip "source $INIT_SCRIPT; cd $EXP_DIR; \
        ray start --head --node-ip-address=$head_node_ip --port=$port \
        --num-cpus $CPUS_PER_NODE --num-gpus $GPUS_PER_NODE --block" &

    # Optional, though may be useful in certain versions of Ray < 1.0.
    sleep 10

    # Number of nodes other than the head node
    nodes_num=$((${#nodes_array[*]} - 1))
    echo "$nodes_num nodes"

    for ((i = 1; i <= nodes_num; i++)); do
        node_i=${nodes_array[$i]}
        node_i_ip=$(dig $node_i a +short | awk 'FNR==1')
        echo "Starting WORKER $i at $node_i with ip=$node_i_ip"
        ssh -tt $node_i_ip "source $INIT_SCRIPT; cd $EXP_DIR; \
            ray start --address $ip_head \
            --num-cpus $CPUS_PER_NODE --num-gpus $GPUS_PER_NODE --block" &
        sleep 5
    done

    # Check the status of the Ray cluster
    ssh -tt $head_node_ip "source $INIT_SCRIPT && ray status"

    # Run the search
    ssh -tt $head_node_ip "source $INIT_SCRIPT && cd $EXP_DIR && python myscript.py"

    # Stop de Ray cluster
    ssh -tt $head_node_ip "source $INIT_SCRIPT && ray stop"

.. note::

    About the *COBALT* directives :

    .. code-block:: bash

        #COBALT -q full-node

    The queue your job will be submitted to. For ThetaGPU it can either be ``single-gpu``, ``full-node``, or ``bigmem`` ; you can find here the `specificities of these queues <https://www.alcf.anl.gov/support-center/theta-gpu-nodes/job-and-queue-scheduling-thetagpu#gpu-queues>`_.

    .. code-block:: bash

        #COBALT -n 2

    The number of nodes your job will be submitted to.

    .. code-block:: bash

        #COBALT -t 20

    The duration of the job submission, in minutes.

    .. code-block:: bash

        #COBALT -A $PROJECT_NAME

    Your current project, e-g ``#COBALT -A datascience``:

    .. code-block:: bash

        #COBALT --attrs filesystems=home,grand,eagle,theta-fs0

    The filesystems your application should have access to, DeepHyper only requires ``home`` and ``theta-fs0``, and it is unnecessary to let in this list a filesystem your application (and DeepHyper) doesn't need.

Adapt the executed Python application depending on your needs. Here is an application example of ``CBO`` using the ``ray`` evaluator:

.. code-block:: python
    :caption: **file**: ``myscript.py``

    import pathlib
    import os

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    import numpy as np

    from deephyper.evaluator import Evaluator
    from deephyper.hpo import CBO
    from deephyper.evaluator.callback import ProfilingCallback

    from deephyper.hpo import HpProblem


    hp_problem = HpProblem()
    hp_problem.add_hyperparameter((-10.0, 10.0), "x")

    def run(config):
        return - config["x"]**2

    timeout = 10
    search_log_dir = "search_log/"
    pathlib.Path(search_log_dir).mkdir(parents=False, exist_ok=True)

    # Evaluator creation
    print("Creation of the Evaluator...")
    evaluator = Evaluator.create(
        run,
        method="ray",
        method_kwargs={
            "adress": "auto",
            "num_gpus_per_task": 1,
        }
    )
    print(f"Creation of the Evaluator done with {evaluator.num_workers} worker(s)")

    # Search creation
    print("Creation of the search instance...")
    search = CBO(
        hp_problem,
        evaluator,
    )
    print("Creation of the search done")

    # Search execution
    print("Starting the search...")
    results = search.search(timeout=timeout)
    print("Search is done")

    results.to_csv(os.path.join(search_log_dir, f"results.csv"))

Finally, submit the script using:

.. code-block:: bash

    qsub-gpu job-deephyper.sh

.. note::

    The ``ssh -tt $head_node_ip "source $INIT_SCRIPT && ray status"`` command is used to check the good initialization of the Ray cluster. Once the job starts running, check the ``*.output`` file and verify that the number of detected GPUs is correct.
