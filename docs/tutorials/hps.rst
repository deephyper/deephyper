.. _create-new-hps-problem:

How to Create/Run a new HPS Problem?
************************************

.. automodule:: deephyper.benchmark.hps

Create the Problem & the model to run
=====================================

For HPS a benchmark is defined by a problem definition and a function
that runs the model::

      deephyper/.../problem_folder/
            __init__.py
            problem.py
            model_run.py
            load_data.py

The problem contains the parameters you want to search over. They are defined
by their name, their space and a default value for the starting point.
Deephyper is using the `Skopt <https://scikit-optimize.github.io/optimizer/index.html>`_,
hence it recognizes three types of parameters:

- a (lower_bound, upper_bound) tuple (for Real or Integer dimensions),
- a (lower_bound, upper_bound, "prior") tuple (for Real dimensions),
- as a list of categories (for Categorical dimensions)

For example if we want to create an hyperparameter search problem for Mnist
with a given starting point:

.. note::
    Many starting points can be defined with ``Problem.add_starting_point(**dims)``. All starting points will be evaluated before generating other evaluations. The starting
    point help the user to bring actual knowledge of the current search space. For
    instance if you know a good set of hyperparameter for your current models.



.. literalinclude:: ../../deephyper/benchmark/hps/mnistmlp/problem.py
    :linenos:
    :caption: deephyper/benchmark/hps/mnistmlp/problem.py
    :name: benchmark-hps-mnistmlp-problem-py

.. note::
    You can notice the ``if __name__ == '__main__'`` at the end of
    :ref:`benchmark-hps-mnistmlp-problem-py` with a ``print(Problem)`` statement.
    Indeed it is well adviced to print the problem after defining it in order to
    make sure the definition is correct. For :ref:`benchmark-hps-mnistmlp-problem-py`
    the output is:
    ::

        Problem
        { 'activation_l1': ['relu', 'elu', 'selu', 'tanh'],
        'activation_l2': ['relu', 'elu', 'selu', 'tanh'],
        'batch_size': (8, 1024),
        'dropout_l1': (0.0, 1.0),
        'dropout_l2': (0.0, 1.0),
        'epochs': (5, 500),
        'nunits_l1': (1, 1000),
        'nunits_l2': (1, 1000)}

        Starting Point
        {0: {'activation_l1': 'relu',
            'activation_l2': 'relu',
            'batch_size': 8,
            'dropout_l1': 0.0,
            'dropout_l2': 0.0,
            'epochs': 5,
            'nunits_l1': 1,
            'nunits_l2': 2}}



and that's it, we just defined a problem with 8 dimensions: ``activation_l1,
activation_l2, batch_size, dropout_l1, dropout_l2, epochs, nunits_l1,
nunits_l2``. Now the problem is defined the next step is to define a function
which will run our Mnist model while taking in account the parameters chosen by
the search. This function is returning an objective scalar value which is
minimized by the hyperparameter search algorithm. For our Mnist problem we want
to maximize the accuracy so the return value is ``return -score[1]``.


.. literalinclude:: ../../deephyper/benchmark/hps/mnistmlp/mnist_mlp.py
    :linenos:
    :caption: deephyper/benchmark/hps/mnistmlp/mnist_mlp.py
    :name: benchmark-hps-mnistmlp-mnist_mlp-py

.. note::
    A ``dict`` is passed to the function running the model. In :ref:`benchmark-hps-mnistmlp-mnist_mlp-py` the function running the model is ``run``. This ``dict`` will be similar to the starting point ``dict`` of :ref:`benchmark-hps-mnistmlp-problem-py` but with values corresponding to the search choices.

.. note::
    It is well adviced to test the function running the model locally by giving it an
    example ``dict`` corresponding to a point included in the problem definition.
    For example you can use a starting point of your problem ::

        run(Problem.starting_point_asdict[0])

.. WARNING::
    When designing a new optimization experiment, keep in mind ``model_run.py``
    must be runnable from an arbitrary working directory. This means that Python
    modules simply located in the same directory as the ``model_run.py`` will not be
    part of the default Python import path, and importing them will cause an ``ImportError``! For example in :ref:`benchmark-hps-mnistmlp-mnist_mlp-py` we are doing ``import load_data``.

    To ensure that modules located alongside the ``model_run.py`` script are
    always importable, a quick workaround is to explicitly add the problem
    folder to ``sys.path`` at the top of the script:

    .. code-block:: python
        :linenos:

        import os
        import sys
        here = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, here)
        # import user modules below here

    .. literalinclude:: ../../deephyper/benchmark/hps/mnistmlp-script/mnist_mlp.py
        :linenos:
        :caption: deephyper/benchmark/hps/mnistmlp-script/mnist_mlp.py
        :name: benchmark-hps-mnistmlp-script-mnist_mlp-py

.. highlight:: console

Assuming you have installed ``deephyper`` in your environment we will show you
how to run an asynchronous model-based search (AMBS) on a benchmark included
within deephyper. To print the arguments of a search like AMBS just run::

    python -m deephyper.search.hps.ambs --help

The expected output is::

    usage: ambs.py [-h] [--problem PROBLEM] [--run RUN] [--backend BACKEND]
               [--max-evals MAX_EVALS]
               [--eval-timeout-minutes EVAL_TIMEOUT_MINUTES]
               [--evaluator {balsam,subprocess,processPool,threadPool}]
               [--learner {RF,ET,GBRT,DUMMY,GP}]
               [--liar-strategy {cl_min,cl_mean,cl_max}]
               [--acq-func {LCB,EI,PI,gp_hedge}]

    optional arguments:
    -h, --help            show this help message and exit
    --problem PROBLEM     Module path to the Problem instance you want to use
                            for the search (e.g.
                            deephyper.benchmark.hps.polynome2.Problem).
    --run RUN             Module path to the run function you want to use for
                            the search (e.g.
                            deephyper.benchmark.hps.polynome2.run).
    --backend BACKEND     Keras backend module name
    --max-evals MAX_EVALS
                            maximum number of evaluations
    --eval-timeout-minutes EVAL_TIMEOUT_MINUTES
                            Kill evals that take longer than this
    --evaluator {balsam,subprocess,processPool,threadPool}
                            The evaluator is an object used to run the model.
    --learner {RF,ET,GBRT,DUMMY,GP}
                            type of learner (surrogate model)
    --liar-strategy {cl_min,cl_mean,cl_max}
                            Constant liar strategy
    --acq-func {LCB,EI,PI,gp_hedge}
                            Acquisition function type


Run an Hyperparameter Search locally
====================================

Now you can run AMBS with custom arguments.

.. WARNING::
    By default a ``subprocess`` evaluator is used.

The python package way
----------------------

This method is used when the *problem* and *run* are installed in the current
python environment. With ``setuptools`` you can easily create a new package
and install it in your current python environment::

    python -m deephyper.search.hps.ambs --problem deephyper.benchmark.hps.mnistmlp.problem.Problem --run deephyper.benchmark.hps.mnistmlp.mnist_mlp.run

.. note::
    ``deephyper.benchmark.hps.mnistmlp.problem.Problem`` and ``deephyper.benchmark.hps.mnistmlp.mnist_mlp.run`` are python imports. Where ``Problem`` and ``run`` are attributes. This is working because the package ``deephyper`` is installed in ``dh-opt`` our current virtual environment.

The python script way
---------------------

::

    python -m deephyper.search.hps.ambs --problem deephyper/benchmark/hps/mnistmlp-script/problem.py --run deephyper/benchmark/hps/mnistmlp-script/mnist_mlp.py

.. note::
    When using a path to a python script for the ``--problem`` and ``--run`` argument. This is assuming that there is a ``Problem`` attribute in the *problem* script and a
    ``run`` attribute in the *run* script.


Run an Hyperparameter Search on Theta
=====================================

Now we are going to run an *AMBS* search on ``deephyper.benchmark.hps.mnistmlp``
benchmark.

.. WARNING::
    We are assuming *deephyper* is already installed on Theta. If not please go to :ref:`theta-user-installation`.

Then you can create a new postgress database in the current directory, this
database is used by the balsam software::

    balsam init testdb

.. note::
    To see a list of accessible databses do::

        balsam which

Once the database has been created you can start it or link to it if it is
already running::

    source balsamactivate testdb

The database is now running, let's now create our first balsam application
in order to run an Asynchronous Model-Based Search (AMBS)::

    [BalsamDB: testdb] dhuser $ balsam app --name AMBS --exec 'python -m deephyper.search.hps.ambs'

.. WARNING::
    The ``python`` has to be the python interpretor where *deephyper* is currently installed. If I am using a virtual environment such as ``dh-opt`` the *exec* argument should be something like ``~/dh-opt/bin/python -m deephyper.search.hps.ambs``.

You can run the following command to print all the applications available in
your current balsam environment::

    balsam ls apps

.. note::
    If you want to see more information about your apps use the ``--verb`` argument. You can also configure the ``BALSAM_LS_FIELDS`` env var such as::

        export BALSAM_LS_FIELDS=TODO

Now you can create a new job::

    balsam job --name test --application AMBS --workflow TEST --args '--evaluator balsam --problem deephyper.benchmark.hps.polynome2.Problem --run deephyper.benchmark.hps.polynome2.run'

.. note::
    Each balsam job creates a directory starting by its *name* then *pk* (primary key, a database id) located at ``testdb/data/$workflow/``. The created directory will be the working directory (``$PWD``) of the job.

You can check the configuration of your jobs by using ``balsam ls``::

    balsam ls jobs --name test

.. note::
    If you want to see more information about your apps use the ``--verb`` argument. So that you can see the full command line which will be run to start the job. It can be useful to check the good python interpretor is used.

Define your ``PROJECT_NAME``, for us it is ``datascience``::

    export PROJECT_NAME=datascience

Finally you can submit a cobalt job to Theta::

    balsam submit-launch -n 128 -q default -t 180 -A $PROJECT_NAME --job-mode serial --wf-filter TEST


Now if you want to look at the logs, go to ``testdb/data/TEST``. You'll see
one directory prefixed with ``test``. Inside this directory you will find the
logs of you search. All the other directories prefixed with ``task`` correspond
to the logs of your ``--run`` function, here the run function is corresponding
to the training of a neural network.

.. note::

    In case of failure the job *state* will be set to ``FAILED`` in the balsam database (do ``balsam ls jobs`` to see jobs states). To find why the job failed you can look at the ``testdb/data/TEST/test_$jobid`` or at the ``testdb/logs`` folder.


.. note::
    The ``--wf-filter $workflow`` arguments tell to balsam to only execute jobs with this *workflow*. If you don't specify this filter balsam will pull and start all available jobs in the database. The ``-n``, ``-q``, ``-t`` and ``-A`` are Cobalt arguments, hence if you want to start a job in a debug queue you can do::

        balsam submit-launch -n 8 -q debug-cache-quad -t 30 -A $PROJECT_NAME --job-mode serial --wf-filter TEST

    The ``balsam submit-launch`` command generates a cobalt script using a Jinja template located at ``~/.balsam/job-templates/theta.cobaltscheduler.tmpl``. You can edit this template if required.

.. note::

    The ``--job-mode serial`` will use one compute node for a launcher to start jobs on 1 compute node. This feature help to reduce overhead and limits of the ``aprun`` command. Hence with AMBS 1 compute node will be used by the search and 1 compute nodes will be used by the launcher which means you can't ask less than 3 nodes in the debug queue.


Run an Hyperparameter Search on Cooley
======================================

On Cooley two GPUs are available per node. By default one evaluation per node
will be executed which means you can use 2 GPUs for one model. If you want to
use 1 GPU per evaluation with deephyper please follow these steps.

.. note::
    It means 2 evaluations per node will happened in parallel. In sum you will have twice the number of deephyper workers.

1. Use the Cooley Job template of Balsam::

    vim ~/.balsam/settings.json

The following default settings are expected

.. literalinclude:: balsam_settings_example.json
    :linenos:
    :caption: Balsam settings
    :name: balsam-settings-json

and set ``JOB_TEMPLATE`` to ``job-templates/cooley.cobaltscheduler.tmpl``.

2. then add the line ``export DEEPHYPER_WORKERS_PER_NODE=2`` to the job
template::

    vim ~/.balsam/job-templates/cooley.cobaltscheduler.tmpl
