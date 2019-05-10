Running locally
***************

This section will show you how to run Hyperparameter or neural architecture on your local machine. All search can be run throw command line or using python.

Hyperparameter search
=====================

Command Line
------------

Assuming you have installed deephyper on your local environment we will show you how to run an asynchronous model-based search (AMBS) on a benchmark included within deephyper. To print the arguments of a search like AMBS just run:

::

    python -m deephyper.search.hps.ambs --help

Now you can run AMBS with custom arguments:

::

    python -m deephyper.search.hps.ambs --problem deephyper.benchmark.hps.polynome2.Problem --run deephyper.benchmark.hps.polynome2.run

Python
------

You can also use our hyperparameter searches directly from a python file by importing its corresponding class, for more details see :ref:`SearchDH`.

Neural Architecture Search
==========================

Command Line
------------

::

    python -m deephyper.search.nas.ppo_a3c_sync --problem deephyper.benchmark.nas.mnist1D.problem.Problem --run deephyper.search.nas.model.run.alpha.run

Python
------

You can also use our hyperparameter searches directly from a python file by importing its corresponding class, for more details see :ref:`SearchDH`.
