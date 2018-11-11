Running locally
***************

This section will show you how to run Hyperparameter or neural architecture on your local machine. All searches can be run throw command line or using python.

Hyperparameter search (HPS)
===========================

Command Line
------------

Assuming you have installed deephyper on your local environment.

::

    cd deephyper/deephyper/searches/hps
    python ambs.py

This is going to run an asynchronous model-based search (AMBS) with default parameters. To print the arguments of a search like AMBS just run :

::

    python ambs.py --help

Now you can run AMBS with custom arguments :

::

    python ambs.py --problem deephyper.benchmarks.hps.b2.problem.Problem --run deephyper.benchmarks.hps.b2.babi_memnn.run

Python
======

TODO

Neural Architecture Search (NAS)
================================

Command Line
------------

::

    cd deephyper/deephyper/searches/nas
    python nas_a3c_sync.py --problem deephyper.benchmarks.nas.mnist1D.problem.Problem --run deephyper.searches.nas.run.nas_structure_raw.run

Python
------

TODO
