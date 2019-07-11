.. _create-new-hps-problem:

Create a new hyperparameter search problem
******************************************


A  hyperparameter search (HPS) problem can be defined using three files with a HPS problem directory::

      hps_problem_directory/
            load_data.py
            model_run.py
            problem.py


We will illustrate the HPS problem definition using a regression example. We will use polynome function to generate training and test data
and run a HPS to tune the hyperparameters of a simple neural network.




Create a problem directory
==========================
First, we will create a hps_problem_directory ``polynome2``.

.. code-block:: console
    :caption: bash

    mkdir polynome2
    cd polynome2

Create load_data.py
===================

Second, we will create ``load_data.py`` file that loads and returns the training and testing data.

.. code-block:: console
    :caption: bash

    touch load_data.py

We will generate data from a function :math:`f` where :math:`X \in [a, b]^n`  such as :math:`f(X) = -\sum_{i=0}^{n-1} {x_i ^2}`:

.. literalinclude:: polynome2/load_data.py
    :linenos:
    :caption: polynome2/load_data.py
    :name: polynome2-load_data

Test the ``load_data`` function:

.. code-block:: console
    :caption: bash

    python load_data.py

The expected output is:

.. code-block:: python
    :caption: [Out]

    train_X shape: (8000, 10)
    train_y shape: (8000, 1)
    test_X shape: (2000, 10)
    test_y shape: (2000, 1)

Create model_run.py
===================

Third, we will create ``model_run.py`` that contains the code for the neural network.
We will use Keras for the neural network definition.

.. code-block:: console
    :caption: bash

    touch model_run.py

We will create a neural network model and run it to make sure that the model runs without any error:

.. literalinclude:: polynome2/model_run_step_0.py
    :linenos:
    :caption: Step 0: polynome2/model_run.py
    :name: polynome2-model_run_step_0

.. note::

    Adding an ``EarlyStopping(...)`` callback is a good idea to stop the training of your model as soon as it is stops to improve.

    .. code-block:: python3

        ...
        callbacks=[EarlyStopping(
                            monitor='val_r2',
                            mode='max',
                            verbose=1,
                            patience=10
                        )]
        ...

Train this model and look at it accuracy:

.. code-block:: console
    :caption: bash

    python model_run.py

.. code-block:: console
    :caption: [Out]

    objective: -0.00040728187561035154

.. image:: polynome2/model_step_0_val_r2.png

.. note::
    When defining a new HPS problem, ``model_run.py``
    must be runnable from any arbitrary working directory. This means that Python
    modules simply located in the same directory as the ``model_run.py`` will not be
    part of the default Python import path, and importing them will cause an ``ImportError``.
    For example in :ref:`polynome2-model_run_step_0` we are doing ``import load_data``.

    To ensure that modules located alongside the ``model_run.py`` script are
    always importable, a quick workaround is to explicitly add the problem
    folder to ``sys.path`` at the top of the script:

    .. code-block:: python
        :linenos:

        import os
        import sys
        here = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, here)

Next, we modify :ref:`polynome2-model_run_step_0` for HPS with DeepHyper. The ``run`` function will be used by DeepHyper and a ``point`` Python dictionary will be passed
to this function.
In the ``point`` dictionary, each key is a hyperparameter.
The tunable hyperparameters are the number of units of the Dense layer (i.e. ``point['units']``),
the activation function of the Dense layer (i.e. ``point['activation'])``, and
the learning rate of the optimizer (i.e. ``point['lr']``).
These modifications are given in :ref:`polynome2-model_run_step_1`:

.. literalinclude:: polynome2/model_run_step_1.py
    :linenos:
    :caption: Step 1: polynome2/model_run.py
    :name: polynome2-model_run_step_1

Create problem.py
==================

The ``run`` function in ``model_run.py`` accepts arguments: ``units, activation, and lr``.
Next, we will define the ranges for the hyperparameters using ``problem.py`` file.
Each hyperparameter range is defined using the following notation.
The integer and real hyperparameter range is given by a tuple (lower_bound, upper_bound).
The categorical or nonordinal hyperparameter range is given by a list of possible values.

.. code-block:: console
    :linenos:

    touch problem.py

You can also add starting points to your problem if you already know good starting points in the search space.
Just use the ``add_starting_point(...)`` method.

.. note::
    Several starting points can be defined with ``Problem.add_starting_point(**dims)``.
    All starting points will be evaluated before generating other evaluations.

.. literalinclude:: polynome2/problem_step_1.py
    :linenos:
    :caption: polynome2/problem.py
    :name: polynome2-problem-step-1

You can look at the representation of your problem:

.. code-block:: console
    :caption: bash

    python problem.py

The expected output is:

.. code-block:: console
    :caption: [Out]

    Problem
    { 'activation': [None, 'relu', 'sigmoid', 'tanh'],
    'lr': (0.0001, 1.0),
    'units': (1, 100)}

    Starting Point
    {0: {'activation': None, 'lr': 0.01, 'units': 10}}


Running the search locally
==========================

Everything is ready to run. Let's remember the structure of our experiment::

      polynome2/
            load_data.py
            model_run.py
            problem.py

All the three files have been tested one by one on the local machine. Next, we will run asynchronous model-based search (AMBS).

.. code-block:: console
    :caption: bash

    python -m deephyper.search.hps.ambs --problem problem.py --run model_run.py

.. note::

    In order to run DeepHyper locally and on other systems we are using :ref:`evaluators`. For local evaluations we use the :ref:`subprocess-evaluator`.

.. WARNING::

    When a path to python scripts is given to ``--problem, --run`` arguments you have to make sure that the problem script contains a ``Problem`` attribute and the run script contains a ``run`` attribute.
    Another way to use these arguments is to give a python import path such as ``mypackage.mymodule.myattribute``, where ``myattribute`` should be an ``HpProblem`` instance for the problem argument and
    it should be a callable object with one parameter for the run argument. In order to do so ``mypackage`` should be installed in your current python environment.
    A package structure look like:

    .. code-block:: console

        myfolder/
            setup.py
            mypackage/
                __init__.py
                mymodule.py

    where ``setup.py`` contains the following:

    .. code-block:: python3
        :caption: setup.py

        from setuptools import setup

        setup(
        name='mypackage',
        packages=['mypackage'],
        install_requires=[])

    To install ``mypackage`` just do:

    .. code-block:: console
        :caption: bash

        cd myfolder

    Then:

    .. code-block:: console
        :caption: bash

        python setup.py install

After the search is over, you will find the following files in your current folder:

.. code-block:: console

    deephyper.log
    results.csv
    results.json


.. include:: polynome2/dh-analytics-hps.rst


The best point the search found::

    point = {
        'activation': 'relu',
        'lr': 0.8820413612862609,
        'units': 21
    }

Just pass this ``point`` to your run function

.. literalinclude:: polynome2/model_run_step_1_point.py
    :linenos:
    :caption: Step 1: polynome2/model_run.py
    :name: polynome2-model_run_step_1_point

And run the script:

.. code-block:: console
    :caption: bash

    python model_run.py

.. code-block:: console
    :caption: [Out]

    objective:  0.47821942329406736

.. image:: polynome2/model_step_1_val_r2.png

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

Create a Balsam ``AMBS`` application:

.. code-block:: console
    :caption: bash

    balsam app --name AMBS --exe "$(which python) -m deephyper.search.hps.ambs"

.. code-block:: console
    :caption: [Out]

    Application 1:
    -----------------------------
    Name:           AMBS
    Description:
    Executable:     /lus/theta-fs0/projects/datascience/regele/dh-opt/bin/python -m deephyper.search.hps.ambs
    Preprocess:
    Postprocess:

Add a Balsam job to the ``polydb`` database:

.. code-block:: console
    :caption: bash
    
    balsam job --name step_2 --workflow step_2 --app AMBS --args "--evaluator balsam --problem $PWD/polynome2/problem.py --run $PWD/polynome2/model_run.py"

.. code-block:: console
    :caption: [Out]
    
    BalsamJob 575dba96-c9ec-4015-921c-abcb1f261fce
    ----------------------------------------------
    workflow:                       step_2
    name:                           step_2
    description:
    lock:
    parents:                        []
    input_files:                    *
    stage_in_url:
    stage_out_files:
    stage_out_url:
    wall_time_minutes:              1
    num_nodes:                      1
    coschedule_num_nodes:           0
    ranks_per_node:                 1
    cpu_affinity:                   none
    threads_per_rank:               1
    threads_per_core:               1
    node_packing_count:             1
    environ_vars:
    application:                    AMBS
    args:                           --evaluator balsam --problem /projects/datascience/regele/polynome2/problem.py --run /projects/datascience/regele/polynome2/model_run.py
    user_workdir:
    wait_for_parents:               True
    post_error_handler:             False
    post_timeout_handler:           False
    auto_timeout_retry:             True
    state:                          CREATED
    queued_launch_id:               None
    data:                           {}
    *** Executed command:         /lus/theta-fs0/projects/datascience/regele/dh-opt/bin/python -m deephyper.search.hps.ambs --evaluator balsam --problem /projects/datascience/regele/polynome2/problem.py --run /projects/datascience/regele/polynome2/model_run.py
    *** Working directory:        /lus/theta-fs0/projects/datascience/regele/polydb/data/step_2/step_2_575dba96

    Confirm adding job to DB [y/n]: y

Submit the search to the Cobalt scheduler:

.. code-block:: console
    :caption: bash

    balsam submit-launch -n 4 -q debug-cache-quad -t 60 -A datascience --job-mode serial --wf-filter step_2

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
        'wf_filter': 'step_2'}

Now the search is done. You will find results at ``polydb/data/step_2/step_2_575dba96``.
