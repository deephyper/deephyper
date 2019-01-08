Quick Start
***********

Local
================================================

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


Theta
==================================

First we are going to run a search on b2 benchmark.

::

    # Load deephyper on theta
    module load deephyper

    # Create a new postgress database in current directory
    balsam init testdb

    # Start or Link to the database
    source balsamactivate testdb

    # Create a new application and Print applications referenced
    balsam app --name AMBS --exec $DH_AMBS
    balsam ls apps

    # Create a new job and Print jobs referenced
    balsam job --name test --application AMBS --workflow TEST --args '--problem deephyper.benchmarks.b2.problem.Problem --run deephyper.benchmarks.b2.babi_memnn.run'
    balsam ls jobs

    # Submit a Theta job that will run the balsam job corresponding to the search
    balsam submit-launch -n 128 -q default -t 180 -A PROJECT_NAME --job-mode serial --wf-filter TEST


Now if you want to look at the logs, go to ``testdb/data/TEST``. You'll see one directory prefixed with ``test``. Inside this directory you will find the logs of you search. All the other directories prefixed with ``task`` correspond to the logs of your ``--run`` function, here the run function is corresponding to the training of a neural network.


