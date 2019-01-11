Running locally
***************

This section will show you how to run Hyperparameter or neural architecture on your local machine. All search can be run throw command line or using python.

Hyperparameter search
=====================

Command Line
------------

Assuming you have installed deephyper on your local environment.

::

    cd deephyper/deephyper/search/hps
    python ambs.py

This is going to run an asynchronous model-based search (AMBS) with default parameters. To print the arguments of a search like AMBS just run :

::

    python ambs.py --help

Now you can run AMBS with custom arguments :

::

    python ambs.py --problem deephyper.benchmark.hps.b2.problem.Problem --run deephyper.benchmark.hps.b2.babi_memnn.run

Python
------

.. todo:: use hps inside python

Neural Architecture Search
==========================

Command Line
------------

::

    cd deephyper/deephyper/search/nas
    python ppo_a3c_sync.py --problem deephyper.benchmark.nas.mnist1D.problem.Problem --run deephyper.search.nas.model.run.alpha.run

Python
------

.. todo:: use nas inside python
