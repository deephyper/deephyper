Running locally
***************

Command line
============

Assuming you have installed deephyper on your local environment.

::

    cd deephyper/deephyper/search
    python ambs.py

This is going to run an asynchronous model-based search (AMBS) with default parameters. To print the arguments of a search like AMBS just run :

::

    python ambs.py --help

Now you can run AMBS with custom arguments :

::

    python ambs.py --problem deephyper.benchmarks.b2.problem.Problem --run deephyper.benchmarks.b2.babi_memnn.run

Inside a python script
======================

TODO
