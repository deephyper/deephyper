.. _create-new-nas-problem:

Neural Architecture Search (NAS)
***********************************************

A  neural architecture search (NAS) problem can be defined using three files with a NAS problem directory within a python package::

    nas_problems/
        setup.py
        nas_problems/
            __init__.py
            myproblem/
                __init__.py
                load_data.py
                problem.py
                search_space.py


We will illustrate the NAS problem definition using a regression example. We will use polynome function to generate training and test data and run a NAS to find the best search_space for this experiment.

Create a python package
=======================

Init a new nas project:

.. code-block:: console
    :caption: bash

    deephyper start-project nas_problems

The project was created and installed in your current python environment. Then go to the ``nas_problems`` package and create a new problem:

.. code-block:: console
    :caption: bash

    cd nas_problems/nas_problems/
    deephyper new-problem nas polynome2

The problem was created. Then go to the problem ``polynome2`` folder:

.. code-block:: console
    :caption: bash

    cd nas_problems/nas_problems/polynome2

Create load_data.py
===================

Fist, we will look at the ``load_data.py`` file that loads and returns the
training and validation data. The ``load_data`` function generates data from
a function :math:`f` where :math:`X \in [a, b]^n`  such as
:math:`f(X) = -\sum_{i=0}^{n-1} {x_i ^2}`:

.. literalinclude:: polynome2/load_data.py
    :linenos:
    :caption: polynome2/load_data.py
    :name: polynome2-load_data-nas

Test the ``load_data`` function:

.. code-block:: console
    :caption: bash

    python load_data.py

The expected output is:

.. code-block:: python
    :caption: [Out]

    train_X shape: (8000, 10)
    train_y shape: (8000, 1)
    valid_X shape: (2000, 10)
    valid_y shape: (2000, 1)

Create search_space.py
======================

Then, we will take a look at ``search_space.py`` which contains the code for
the neural network search_space definition.

.. literalinclude:: polynome2_nas/search_space.py
    :linenos:
    :caption: polynome2/search_space.py
    :name: polynome2-search_space


Create problem.py
==================

Now, we will take a look at ``problem.py`` which contains the code for the
problem definition.

.. literalinclude:: polynome2_nas/problem.py
    :linenos:
    :caption: polynome2_nas/problem.py
    :name: polynome2-nas-problem

You can look at the representation of your problem:

.. code-block:: console
    :caption: bash

    python problem.py

The expected output is:

.. code-block:: console
    :caption: [Out]

    Problem is:
    * SEED = 2019 *
        - search space   : nas_problems.polynome2.search_space.create_search_space
        - data loading   : nas_problems.polynome2.load_data.load_data
        - preprocessing  : deephyper.nas.preprocessing.minmaxstdscaler
        - hyperparameters:
            * verbose: 1
            * batch_size: 32
            * learning_rate: 0.01
            * optimizer: adam
            * num_epochs: 20
            * callbacks: {'EarlyStopping': {'monitor': 'val_r2', 'mode': 'max', 'verbose': 0, 'patience': 5}}
        - loss           : mse
        - metrics        :
            * r2
        - objective      : val_r2__last
        - post-training  : None


Running the search locally
==========================

Everything is ready to run. Let's remember the search_space of our experiment::

    polynome2/
        __init__.py
        load_data.py
        problem.py
        search_space.py

Each of these files have been tested one by one on the local machine.
Next, we will run a random search (RDM).

.. code-block:: console
    :caption: bash

    deephyper nas random --evaluator ray --problem nas_problems.polynome2.problem.Problem

.. note::

    In order to run DeepHyper locally and on other systems we are using :mod:`deephyper.evaluator`. For local evaluations we can use the :class:`deephyper.evaluator.RayEvaluator` or the :class:`deephyper.evaluator.SubProcessEvaluator`.


After the search is over, you will find the following files in your current folder:

.. code-block:: console

    deephyper.log

You can now use ``deephyper-analytics`` to plot some information about the search

.. code-block:: console
    :caption: bash

    deephyper-analytics parse deephyper.log

A JSON file should have been generated. We will now create a juyter notebook (replace ``$MY_JSON_FILE`` by the name of the json file created with ``parse``:

.. code-block:: console
    :caption: bash

    deephyper-analytics single -p $MY_JSON_FILE

    jupyter notebook dh-analytics-single.ipynb


Running the search on ALCF's Theta and Cooley
==============================================

Now let's run the search on an HPC system such as Theta or Cooley. First create a Balsam database:

.. code-block:: console
    :caption: bash

    balsam init polydb

Start and connect to the ``polydb`` database:

.. code-block:: console
    :caption: bash

    source balsamactivate polydb

Create a Balsam ``AE`` application:

.. code-block:: console
    :caption: bash

    balsam app --name AE --exe "$(which python) -m deephyper.search.nas.regevo"

.. code-block:: console
    :caption: [Out]

    Application 1:
    -----------------------------
    Name:           PPO
    Description:
    Executable:     /lus/theta-fs0/projects/datascience/regele/dh-opt/bin/python -m deephyper.search.nas.regevo
    Preprocess:
    Postprocess:

.. code-block:: console
    :caption: bash

    balsam job --name poly_exp --workflow poly_exp --app PPO --num-nodes 2 --args "--evaluator balsam --problem nas_problems.polynome2.problem.Problem"

.. code-block:: console
    :caption: [Out]

    BalsamJob 575dba96-c9ec-4015-921c-abcb1f261fce
    ----------------------------------------------
    workflow:                       poly_exp
    name:                           poly_exp
    description:
    lock:
    parents:                        []
    input_files:                    *
    stage_in_url:
    stage_out_files:
    stage_out_url:
    wall_time_minutes:              1
    num_nodes:                      2
    coschedule_num_nodes:           0
    ranks_per_node:                 1
    cpu_affinity:                   none
    threads_per_rank:               1
    threads_per_core:               1
    node_packing_count:             1
    environ_vars:
    application:                    PPO
    args:                           --evaluator balsam --problem nas_problems.polynome2.problem.Problem
    user_workdir:
    wait_for_parents:               True
    post_error_handler:             False
    post_timeout_handler:           False
    auto_timeout_retry:             True
    state:                          CREATED
    queued_launch_id:               None
    data:                           {}
    *** Executed command:         /lus/theta-fs0/projects/datascience/regele/dh-opt/bin/python -m deephyper.search.nas.regevo --evaluator balsam --problem nas_problems.polynome2.problem.Problem
    *** Working directory:        /lus/theta-fs0/projects/datascience/regele/polydb/data/poly_exp/poly_exp_575dba96

    Confirm adding job to DB [y/n]: y

Submit the search to the Cobalt scheduler:

.. code-block:: console
    :caption: bash

    balsam submit-launch -n 6 -q debug-cache-quad -t 60 -A datascience --job-mode mpi --wf-filter poly_exp

.. code-block:: console
    :caption: [Out]

    Submit OK: Qlaunch {   'command': '/lus/theta-fs0/projects/datascience/regele/polydb/qsubmit/qlaunch1.sh',
        'from_balsam': True,
        'id': 1,
        'job_mode': 'serial',
        'nodes': 6,
        'prescheduled_only': False,
        'project': 'datascience',
        'queue': 'debug-flat-quad',
        'scheduler_id': 347124,
        'state': 'submitted',
        'wall_minutes': 60,
        'wf_filter': 'poly_exp'}

Now the search is done. You will find results at ``polydb/data/poly_exp/poly_exp_575dba96``.
