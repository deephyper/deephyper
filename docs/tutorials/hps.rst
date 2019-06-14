.. _create-new-hps-problem:

Create a new HPS Problem
************************


For HPS a new experiment is defined by a problem definition and a function
that runs the model::

      problem_folder/
            load_data.py
            model_run.py
            problem.py

For this tutorial we will work on a regression experiment. We will start by defining how to load data generated from a polynome function. Then we will set up a function to run our learning model as well as returning the objective we want to maximize (i.e. it will be :math:`R^2` for our regression). In a third part we will define our search space. Finally we will run our experiment and study its results. Let's start and create a new directory for our ``polynome2`` experiment:

.. code-block:: console
    :caption: bash

    mkdir polynome2

Then go to this directory:

.. code-block:: console
    :caption: bash

    cd polynome2

Load your data
==============

Now we can define a function to generate our data. It is a good practice to do it in a specific file, hence we can create a ``load_data.py`` file:

.. code-block:: console
    :caption: bash

    touch load_data.py

We are generating data from a function :math:`f` where :math:`X \in [a, b]^n`  such as :math:`f(X) = -\sum_{i=0}^{n-1} {x_i ^2}`:

.. literalinclude:: polynome2/load_data.py
    :linenos:
    :caption: polynome2/load_data.py
    :name: polynome2-load_data

You are encouraged to test localy your ``load_data`` function. For example you can run it and take a look at the shape of your data:

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

Define your model
=================

Now we can define how to run our machine learning model. In order to do so create a ``model_run.py`` file:

.. code-block:: console
    :caption: bash

    touch model_run.py

Let's now write a first draft of our model without any hyperparameter tunning:

.. literalinclude:: polynome2/model_run_step_0.py
    :linenos:
    :caption: Step 0: polynome2/model_run.py
    :name: polynome2-model_run_step_0

.. note::

    Adding an ``EarlyStopping(...)`` callback is a good idea to stop the training of your model as soon as it is stopping to improve.

    .. code-block:: python3

        ...
        callbacks=[EarlyStopping(
                            monitor='val_r2',
                            mode='max',
                            verbose=1,
                            patience=10
                        )]
        ...

Let's train this model and look at its performance:

.. code-block:: console
    :caption: bash

    python model_run.py

.. code-block:: console
    :caption: [Out]

    objective: -0.00040728187561035154

.. image:: polynome2/model_step_0_val_r2.png

.. WARNING::
    When designing a new optimization experiment, keep in mind ``model_run.py``
    must be runnable from an arbitrary working directory. This means that Python
    modules simply located in the same directory as the ``model_run.py`` will not be
    part of the default Python import path, and importing them will cause an ``ImportError``! For example in :ref:`polynome2-model_run_step_0` we are doing ``import load_data``.

    To ensure that modules located alongside the ``model_run.py`` script are
    always importable, a quick workaround is to explicitly add the problem
    folder to ``sys.path`` at the top of the script:

    .. code-block:: python
        :linenos:

        import os
        import sys
        here = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, here)

Let's now modify our previous :ref:`polynome2-model_run_step_0`. The ``run`` function will be used with DeepHyper and a ``point`` parameter will be passed to this function. The ``point`` is a ``dict`` where each key is a dimension of our search space. For our model we want to tune the number of units of our Dense layer (i.e. ``point['units']``), the activation function of our Dense layer (i.e. ``point['activation']) and the learning of our optimizer (i.e. ``point['lr']``). Now look at :ref:`polynome2-model_run_step_1` and find this modifications:

.. literalinclude:: polynome2/model_run_step_1.py
    :linenos:
    :caption: Step 1: polynome2/model_run.py
    :name: polynome2-model_run_step_1

Define your search space
========================

The model is now accepting 3 tunnable parameters: ``units, activation, lr``. Let's define a search space for these parameters. Create a ``problem.py`` file:

.. code-block:: console
    :linenos:

    touch problem.py

The ``Problem``defined in ``problem.py`` contains the parameters you want to search over. They are defined by their name, their space. DeepHyper is using the `Skopt package <https://scikit-optimize.github.io/optimizer/index.html>`_, hence it recognizes three types of parameters:

- a (lower_bound, upper_bound) tuple (for Real or Integer dimensions),
- a (lower_bound, upper_bound, "prior") tuple (for Real dimensions),
- as a list of categories (for Categorical dimensions)

You can also add starting points to your problem if you already now good points in your search space. Just use the ``add_starting_point(...)`` method.

.. note::
    Many starting points can be defined with ``Problem.add_starting_point(**dims)``. All starting points will be evaluated before generating other evaluations. The starting
    point help the user to bring actual knowledge of the current search space. For
    instance if you know a good set of hyperparameter for your current models.

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


Run the search locally
======================

Everything is ready to run. Let's remember the structure of our experiment::

      problem_folder/
            load_data.py
            model_run.py
            problem.py

All these pieces have now been tested one by one, locally and we are sure they are working. Let's try out an Asynchronous Model Based Search. On your local machine do:

.. code-block:: console
    :caption: bash

    python -m deephyper.search.hps.ambs --problem problem.py --run model_run.py

.. note::

    In order to run DeepHyper locally and on other systems we are using :ref:`evaluators`. For local evaluations we use the :ref:`subprocess-evaluator`.

.. WARNING::

    When a path to python scripts is given to ``--problem, --run`` arguments you have to make sure that the problem script contains a ``Problem`` attribute and the run script contains a ``run`` attribute. An other way to use these arguments is to give a python import path such as ``mypackage.mymodule.myattribute``, where ``myattribute`` should be an ``HpProblem`` instance for the problem argument and it should be a Callable object with one parameter for the run argument. In order to do so ``mypackage`` should be installed in your current python environment. A package structure look like:

    .. code-block:: console

        myfolder/
            setup.py
            mypackage/
                __init__.py
                mymodule.py

    Where ``setup.py`` contains the following:

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

Now the search is done. You should find the following files in your current folder:

.. code-block:: console

    deephyper.log
    results.csv
    results.json

Let's use the ``deephyper-analytics` command line to see what are these results looking like:

.. include:: polynome2/dh-analytics-hps.rst

Let's look at the best candidate the search found which is::

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

Run the search at ALCF
======================

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

    balsam job --name step_2 --workflow step_2 --app AMBS --args "--evaluator balsam --problem $PWD/polynome2/problem_step_2.py --run $PWD/polynome2/model_run_step_2.py"

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
    args:                           --evaluator balsam --problem /projects/datascience/regele/polynome2/problem_step_2.py --run /projects/datascience/regele/polynome2/model_run_step_2.py
    user_workdir:
    wait_for_parents:               True
    post_error_handler:             False
    post_timeout_handler:           False
    auto_timeout_retry:             True
    state:                          CREATED
    queued_launch_id:               None
    data:                           {}
    *** Executed command:         /lus/theta-fs0/projects/datascience/regele/dh-opt/bin/python -m deephyper.search.hps.ambs --evaluator balsam --problem /projects/datascience/regele/polynome2/problem_step_2.py --run /projects/datascience/regele/polynome2/model_run_step_2.py
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