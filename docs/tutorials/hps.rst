.. _create-new-hps-problem:

Create & Run an Hyperparameter Search Problem
*********************************************

.. automodule:: deephyper.benchmark.hps

Create the Problem & the model to run
=====================================

The python package way
----------------------

For HPS a benchmark is defined by a problem definition and a function that runs the model.

::

      deephyper/.../problem_folder/
            __init__.py
            problem.py
            model_run.py
            load_data.py

The problem contains the parameters you want to search over. They are defined
by their name, their space and a default value for the starting point. Deephyper
recognizes three types of parameters:
- continuous (tuple of 2 floats such as (0.0, 1.0))
- discrete ordinal (for instance integers) (tuple of 2 integers (0, 10))
- discrete non-ordinal (for instance a list of tokens) (list of Objects)
For example if we want to create an hyperparameter search problem for Mnist dataset:


.. literalinclude:: ../../deephyper/benchmark/hps/mnistmlp/problem.py
    :linenos:
    :caption: deephyper/benchmark/hps/mnistmlp/problem.py
    :name: benchmark-hps-mnistmlp-problem-py


and that's it, we just defined a problem with 8 dimensions: epochs, nhidden, nunits,
activation, batch_size, dropout, optimizer and learning_rate. Now we need to define a
function which will run our mnist model while taking in account the parameters chosen by
the search.


.. literalinclude:: ../../deephyper/benchmark/hps/mnistmlp/mnist_mlp.py
    :linenos:
    :caption: deephyper/benchmark/hps/mnistmlp/mnist_mlp.py
    :name: benchmark-hps-mnistmlp-mnist_mlp-py:

The python script way
---------------------

::

      deephyper/.../problem_folder/
            problem.py
            model_run.py
            load_data.py

.. literalinclude:: ../../deephyper/benchmark/hps/mnistmlp-script/mnist_mlp.py
    :linenos:
    :caption: deephyper/benchmark/hps/mnistmlp-script/mnist_mlp.py
    :name: benchmark-hps-mnistmlp-mnist_mlp-py:

.. WARNING::
    When designing a new optimization experiment, keep in mind ``model_run.py``
    must be runnable from an arbitrary working directory. This means that Python
    modules simply located in the same directory as the ``model_run.py`` will not be
    part of the default Python import path, and importing them will cause an ``ImportError``!

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

Run an Hyperparameter Search locally
====================================

Assuming you have installed deephyper on your local environment we will show you how to run an asynchronous model-based search (AMBS) on a benchmark included within deephyper. To print the arguments of a search like AMBS just run:


.. highlight:: console

::

    [BalsamDB: testdb] (dh-opt) dhuser $ python -m deephyper.search.hps.ambs --help

Now you can run AMBS with custom arguments:

The python package way
----------------------

::

    [BalsamDB: testdb] (dh-opt) dhuser $ python -m deephyper.search.hps.ambs --problem deephyper.benchmark.hps.mnistmlp.problem.Problem --run deephyper.benchmark.hps.mnistmlp.mnist_mlp.run

The python script way
---------------------

::

    [BalsamDB: testdb] (dh-opt) dhuser $ python -m deephyper.search.hps.ambs --problem deephyper/benchmark/hps/mnistmlp-script/problem.py --run deephyper/benchmark/hps/mnistmlp-script/mnist_mlp.py


Run an Hyperparameter Search on Theta
=====================================

First we are going to run a search on ``deephyper.benchmark.hps.polynome2`` benchmark. In order to achieve this goal we first have to load the deephyper module on Theta. This module is bringing deephyper in you current environment but also the cray python distribution and the balsam software:
::

    [BalsamDB: testdb] (dh-opt) dhuser $ module load deephyper

Then you can create a new postgress database in the current directory, this database is used by the balsam software:
::

    [BalsamDB: testdb] (dh-opt) dhuser $ balsam init testdb

Once the database has been created you can start it, or link to it if it is already running:
::

    [BalsamDB: testdb] (dh-opt) dhuser $ source balsamactivate testdb

The database is now running, let's create our first balsam application in order to run a Asynchronous Model-Based Search (AMBS):
::

    [BalsamDB: testdb] (dh-opt) dhuser $ balsam app --name AMBS --exec $DH_AMBS

You can run the following command to print all the applications available in your current balsam environment:
::

    [BalsamDB: testdb] (dh-opt) dhuser $ balsam ls apps

    # Create a new job and Print jobs referenced
    [BalsamDB: testdb] (dh-opt) dhuser $ balsam job --name test --application AMBS --workflow TEST --args '--evaluator balsam --problem deephyper.benchmark.hps.polynome2.Problem --run deephyper.benchmark.hps.polynome2.run'
    [BalsamDB: testdb] (dh-opt) dhuser $ balsam ls jobs


Finally you can submit a cobalt job to Theta which will start by running your master job named test:
::

    [BalsamDB: testdb] (dh-opt) dhuser $ balsam submit-launch -n 128 -q default -t 180 -A PROJECT_NAME --job-mode serial --wf-filter TEST


Now if you want to look at the logs, go to ``testdb/data/TEST``. You'll see one directory prefixed with ``test``. Inside this directory you will find the logs of you search. All the other directories prefixed with ``task`` correspond to the logs of your ``--run`` function, here the run function is corresponding to the training of a neural network.