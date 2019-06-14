.. _create-new-hps-problem:

Create a new HPS Problem
************************


For HPS a new experiment is defined by a problem definition and a function
that runs the model::

      problem_folder/
            problem.py
            model_run.py
            load_data.py

The ``Problem``defined in ``problem.py`` contains the parameters you want to search over. They are defined
by their name, their space and a default value for the starting point.
DeepHyper is using the `Skopt <https://scikit-optimize.github.io/optimizer/index.html>`_,
hence it recognizes three types of parameters:

- a (lower_bound, upper_bound) tuple (for Real or Integer dimensions),
- a (lower_bound, upper_bound, "prior") tuple (for Real dimensions),
- as a list of categories (for Categorical dimensions)
.. note::
    Many starting points can be defined with ``Problem.add_starting_point(**dims)``. All starting points will be evaluated before generating other evaluations. The starting
    point help the user to bring actual knowledge of the current search space. For
    instance if you know a good set of hyperparameter for your current models.

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

You are encouraged to test localy your ``load_data``. For example you can run it and look at the shape of your data:

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

Run our model
=============

Now we can define how to run our machine learning model. In order to do so create a ``model_run.py`` file:

.. code-block:: console
    :caption: bash

    touch model_run.py

.. literalinclude:: polynome2/model_run_step_0.py
    :linenos:
    :caption: polynome2/model_run.py
    :name: polynome2-model_run_step_0


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

.. literalinclude:: polynome2/model_run_step_1.py
    :linenos:
    :caption: polynome2/model_run_step_1.py
    :name: polynome2-model_run_step_1

.. literalinclude:: polynome2/problem_step_1.py
    :linenos:
    :caption: polynome2/problem_step_1.py
    :name: polynome2-problem-step-1

.. code-block:: console
    :caption: [Out]

    Problem
    {'num_units': (1, 100)}

    Starting Point
    {0: {'num_units': 10}}

.. code-block:: console
    :caption: bash

    python -m deephyper.search.hps.ambs --problem problem.py --run model_run_step_1.py

.. include:: polynome2/dh-analytics-hps.rst

.. literalinclude:: polynome2/model_run_step_2.py
    :linenos:
    :caption: polynome2/model_run_step_2.py
    :name: polynome2-model_run_step_2

.. literalinclude:: polynome2/problem_step_2.py
    :linenos:
    :caption: polynome2/problem_step_2.py
    :name: polynome2-problem-step-2

.. code-block:: console
    :caption: [Out]

    Problem
    { 'activation_l1': [None, 'relu', 'sigmoid', 'tanh'],
    'activation_l2': [None, 'relu', 'sigmoid', 'tanh'],
    'batch_size': (32, 512),
    'dropout_l1': (0.0, 1.0),
    'dropout_l2': (0.0, 1.0),
    'lr': (0.0001, 1.0),
    'units_l1': (1, 100),
    'units_l2': (1, 100)}

    Starting Point
    {0: {'activation_l1': None,
        'activation_l2': None,
        'batch_size': 64,
        'dropout_l1': 0.1,
        'dropout_l2': 0.1,
        'lr': 0.01,
        'units_l1': 10,
        'units_l2': 10}}

.. code-block:: console
    :caption: bash

    balsam init polydb

.. code-block:: console
    :caption: bash

    source balsamactivate polydb

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

.. code-block:: console
    :caption: bash

    balsam submit-launch -n 6 -q debug-flat-quad -t 60 -A datascience --job-mode serial --wf-filter step_2

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

