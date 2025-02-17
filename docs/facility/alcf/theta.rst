.. _tutorial-alcf-01:

Execution on the Theta supercomputer
************************************

In this tutorial we are going to learn how to use DeepHyper on the **Theta** supercomputer at the ALCF. `Theta <https://www.alcf.anl.gov/support-center/theta/theta-thetagpu-overview>`_ is  a Cray XC40, 11.7 petaflops system based on Intel® Xeon Phi processor. It is composed of:

1. Login nodes ``thetalogin*``
2. Head nodes ``thetamom*``
3. Compute nodes ``nid*``

When logging in **Theta** you are located on a **login node**. From this node you can set up your environment such as downloading your data or installing the software you will use for your experimentations. Then, you will setup an experiment using DeepHyper and finally submit an allocation request with the ``qsub`` command line to execute your code on the compute nodes. Once the allocation starts you will be located on a **head node** from which you can access compute nodes using MPI (e.g., with the ``aprun`` command line) or using SSH. However, using SSH requires a specific arguments to be present in your allocation request otherwise it will be blocked by the system.

When using DeepHyper, one can use two different strategies to distribute the computation of evaluations on the supercomputer:

1. :ref:`theta-n-evaluation-per-1-node`: many evaluations can be launched in parallel but each of them only uses the ressources of at most one node (e.g., one neural network training per node).
2. :ref:`theta-1-evaluation-per-n-node`: many evaluations can be launched in parallel and each of them can use the ressources of multiple nodes.

.. admonition:: About the Storage/File Systems
    :class: dropdown, important

    It is important to run DeepHyper from the appropriate storage space because some features can generate a consequante quantity of data such as model checkpointing. The storage spaces available at the ALCF are:

    - ``/lus/grand/projects/``
    - ``/lus/eagle/projects/``
    - ``/lus/theta-fs0/projects/``

    For more details refer to `ALCF Documentation <https://www.alcf.anl.gov/support-center/theta/theta-file-systems>`_.


Submission Script
=================

.. _theta-n-evaluation-per-1-node:

N-evaluation per 1-node
-----------------------

In this strategy, ``N``-evaluations can be launched per available compute node. Therefore, each evaluation uses the ressources of at most one node. For example, you can train one neural network per node, or two neural networks per node. In this case, we will (1) start by launching a Ray cluster accross all available compute nodes, then we will (2) execute the search on one of them and send tasks to previously instanciated workers.

Let us start by defining a toy hyperparameter search. Create a script named ``myscript.py`` and add the following content in it:

.. code-block:: python

    from deephyper.hpo import HpProblem

    # define the run-function to evaluation a given hyperparameter configuration
    # here we simply want to maximise a polynomial function
    def run(config: dict):
        return -config["x"]**2

    # define the variable(s) you want to optimize
    problem = HpProblem()
    problem.add_hyperparameter((-10.0, 10.0), "x")

Then, it is required to define a shell script which will initialize the environment of each compute node. One way to initialize your environment could be to use the ``~/.bashrc`` or ``~/.bash_profile`` which are called at the beginning of each session. However, if you want to have different installations depending on your experimentations it is preferable to avoid activating globally each installation but instead activate them only when necessary. To that end, we will create the ``init-dh-environment.sh`` script which will be called to initialize each compute node:

.. code-block:: console

    $ touch init-dh-environment.sh && chmod +x init-dh-environment.sh

Once created and executable you can add the following content in it (e.g. ``vim init-dh-environment.sh``) and do not forget to adapt the ``$PROJECT_NAME`` with your project's allocation:

.. code-block:: bash
    :caption: **file**: ``init-dh-environment.sh``

    #!/bin/bash

    export TF_CPP_MIN_LOG_LEVEL=3
    export TF_XLA_FLAGS=--tf_xla_enable_xla_devices

    module load miniconda-3

    # Activate installed conda environment with DeepHyper
    conda activate $PATH_TO_CONDA_WITH_DEEPHYPER

    # We had the directory where "myscript.py" is located to be able
    # to access it during the search
    export PYTHONPATH=$MYSCRIPT_DIR:$PYTHONPATH

Adapt the two variables ``$PATH_TO_CONDA_WITH_DEEPHYPER`` and ``$MYSCRIPT_DIR``. Once the ``init-dh-environment.sh`` is created we need to define a submission script. The goal of this script is to (1) request a given amount of ressources, (2) launch a Ray cluster accross all compute nodes, (3) execute a DeepHyper task which distribute the computation on the Ray workers. Create a folder ``exp/`` where to store your experiments. Then create a script named ``deephyper-job.qsub``:

.. code-block:: console

    $ mkdir exp && cd exp/
    $ touch deephyper-job.qsub && chmod +x deephyper-job.qsub

Then add the following content to ``deephyper-job.qsub``:

.. code-block:: bash

    #!/bin/bash
    #COBALT -A $PROJECT_NAME
    #COBALT -n 2
    #COBALT -q debug-flat-quad
    #COBALT -t 60
    #COBALT --attrs enable_ssh=1

    # User Configuration
    EXP_DIR=$PWD
    INIT_SCRIPT=$PWD/../init-dh-environment.sh
    CPUS_PER_NODE=2

    # Initialize environment
    source $INIT_SCRIPT

    # Getting the node names
    nodes_array=($(python -m deephyper.core.cli._nodelist theta $COBALT_PARTNAME | grep -P '\[.*\]' | tr -d '[],'))

    head_node=${nodes_array[0]}
    head_node_ip=$(eval "getent hosts $head_node"| awk {'print $1'})

    # Starting the Ray Head Node
    port=6379
    ip_head=$head_node_ip:$port
    export ip_head
    echo "IP Head: $ip_head"

    echo "Starting HEAD at $head_node"
    ssh -tt $head_node_ip "source $INIT_SCRIPT; cd $EXP_DIR; \
        ray start --head --node-ip-address=$head_node_ip --port=$port \
        --num-cpus $CPUS_PER_NODE --block" &

    # optional, though may be useful in certain versions of Ray < 1.0.
    sleep 10

    # number of nodes other than the head node
    worker_num=$((${#nodes_array[*]} - 1))
    echo "$worker_num workers"

    for ((i = 1; i <= worker_num; i++)); do
        node_i=${nodes_array[$i]}
        node_i_ip=$(eval "getent hosts $node_i"| awk {'print $1'})
        echo "Starting WORKER $i at $node_i with ip=$node_i_ip"
        ssh -tt $node_i_ip "source $INIT_SCRIPT; cd $EXP_DIR; \
            ray start --address $ip_head \
            --num-cpus $CPUS_PER_NODE --block" &
        sleep 5
    done

    # Execute the DeepHyper Task
    # Here the task is an hyperparameter search using the DeepHyper CLI
    # However it is also possible to call a Python script using different
    # Features from DeepHyper (see following notes)
    ssh -tt $head_node_ip "source $INIT_SCRIPT && cd $EXP_DIR && \
        deephyper hps ambs \
        --problem myscript.problem \
        --evaluator ray \
        --run-function myscript.run \
        --ray-address auto \
        --ray-num-cpus-per-task 1"


.. warning::

    The ``#COBALT --attrs enable_ssh=1`` is crucial otherwise ``ssh`` calls will be blocked by the system.

    Don't forget to adapt the ``COBALT`` variables to your needs:

    .. code-block:: console

            #COBALT -A $PROJECT_NAME
            #COBALT -n 2
            #COBALT -q debug-flat-quad
            #COBALT -t 60

.. tip::

    The different ``#COBALT`` arguments can also be passed through the command line:

    .. code-block:: console

        qsub -n 2 -q debug-flat-quad -t 60 -A $PROJECT_NAME \
            --attrs enable_ssh=1 \
            deephyper-job.qsub


.. admonition:: Use a Python script instead of DeepHyper CLI
    :class: dropdown

    Instead of calling ``deephyper hps ambs`` in ``deephyper-job.qsub`` it is possible to call a custom Python script with the following content:

    .. code-block:: python
        :caption: **file**: ``myscript.py``

        def run(hp):
            return hp["x"]

        if __name__ == "__main__":
            import os
            from deephyper.hpo import HpProblem
            from deephyper.hpo import CBO
            from deephyper.evaluator.evaluate import Evaluator

            problem = HpProblem()
            problem.add_hyperparameter((0.0, 10.0), "x")

            evaluator = Evaluator.create(
                run, method="ray", method_kwargs={
                    "address": "auto"
                    "num_cpus_per_task": 1
                }
            )

            search = CBO(problem, evaluator)

            search.search()

    Then replace the ``ssh`` call with:

    .. code-block:: bash

        ssh $HEAD_NODE_IP "source $INIT_SCRIPT; cd $EXP_DIR; \
            python myscript.py"

    This can be more practical to use this approach when integrating DeepHyper in a different workflow.


.. _theta-1-evaluation-per-N-node:

1-evaluation per N-node
-----------------------

.. important::

    **Section under construction!**

The Ray workers are launch on the head node this time. This will allow us to use MPI inside our run-function.

.. code-block:: bash
    :caption: **file**: ``deephyper-job.qsub``

    #!/bin/bash
    #COBALT -A datascience
    #COBALT -n 2
    #COBALT -q debug-flat-quad
    #COBALT -t 30

    # Initialize the head node
    EXP_DIR=$PWD
    INIT_SCRIPT=$PWD/SetUpEnv.sh
    source $INIT_SCRIPT

    # Start Ray workers on the head node
    for port in $(seq 6379 9000); do
        RAY_PORT=$port;
        ray start --head --num-cpus 2 --port $RAY_PORT;
        if [ $? -eq 0 ]; then
            break
        fi
    done

    # Execute the DeepHyper Task
    python myscript.py

In this case the ``run`` function can call MPI routines:

.. code-block:: python

    import os

    def run(config):

        os.system("aprun -n .. -N ..")

        return parse_result()

