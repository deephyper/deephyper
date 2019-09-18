Managing Experiments with Balsam
**********************************

The Balsam command line
=======================

One way to discover the Balsam command line is to look at its help
menu by running::

    balsam --help

Then the following output should be expected::

    usage: balsam [-h]
                {app,job,dep,ls,modify,rm,killjob,mkchild,launcher,submit-launch,init,service,make_dummies,which,log,server}
                ...

    Balsam command line interface

    optional arguments:
    -h, --help            show this help message and exit

    Subcommands:
    {app,job,dep,ls,modify,rm,killjob,mkchild,launcher,submit-launch,init,service,make_dummies,which,log,server}
        app                 add a new application definition
        job                 add a new Balsam job
        dep                 add a dependency between two existing jobs
        ls                  list jobs, applications, or jobs-by-workflow
        modify              alter job or application
        rm                  remove jobs or applications from the database
        killjob             Kill a job without removing it from the DB
        mkchild             Create a child job of a specified job
        launcher            Start a local instance of the balsam launcher
        submit-launch       Submit a launcher job to the batch queue
        init                Create new balsam DB
        service             Start Balsam auto-scheduling service
        which               Get info on current/available DBs
        log                 Quick view of Balsam log files
        server              Control Balsam server at BALSAM_DB_PATH

The Balsam launcher
===================


The Balsam submit launch command
================================


Jobs states
===========

Let's say we just ran the polynome2 problem from ``deephyper.benchmark.hps``.

.. note::
    As a reminder you can create the job with::

        balsam job --name test --application AMBS --workflow TEST --args '--evaluator balsam --problem deephyper.benchmark.hps.polynome2.Problem --run deephyper.benchmark.hps.polynome2.run'

    And submit the job with::

        balsam submit-launch -n 4 -q debug-cache-quad -t 60 -A datascience --job-mode serial --wf-filter TEST

    The ``polynome2`` benchmark is a fast way to test the behavior of deephyper.


If we look at the current state of our balsam database with::

    balsam ls --wf TEST

.. note::

    If you want to print other fields from Balsam jobs ou can configure the ``BALSAM_LS_FIELDS`` env var such as::

        export BALSAM_LS_FIELDS=num_nodes:args

We should expect something like this::

    8bf2e5a1-ff11-4c96-8f32-f45fbfd80612 | task92  | TEST     | deephyper.benchmark.hps.polynome2.problem.run | JOB_FINISHED | 1         | '{"e0": 1, "e1": 9, "e2": 5, "e3": 6, "e4": 2, "e5": 6, "e6": -10, "e7": -6, "e8": 1, "e9": -9}'
    3cb219ef-750c-4861-ad7b-441a6e01145d | task93  | TEST     | deephyper.benchmark.hps.polynome2.problem.run | JOB_FINISHED | 1         | '{"e0": 5, "e1": -10, "e2": 6, "e3": 8, "e4": 10, "e5": 5, "e6": -10, "e7": 2, "e8": -1, "e9": -7}'
    d84ab601-6410-4173-a259-096723ca2e83 | task94  | TEST     | deephyper.benchmark.hps.polynome2.problem.run | JOB_FINISHED | 1         | '{"e0": 8, "e1": 9, "e2": 6, "e3": 8, "e4": -10, "e5": 9, "e6": -2, "e7": 10, "e8": 1, "e9": -9}'
    65b605f0-98f8-4551-bcca-20a6c41b6be7 | task95  | TEST     | deephyper.benchmark.hps.polynome2.problem.run | JOB_FINISHED | 1         | '{"e0": 9, "e1": -5, "e2": 6, "e3": 10, "e4": -8, "e5": -8, "e6": -10, "e7": 0, "e8": 5, "e9": 7}'
    4258d34b-8e1c-4a5f-8242-217158ba28a6 | task96  | TEST     | deephyper.benchmark.hps.polynome2.problem.run | JOB_FINISHED | 1         | '{"e0": 8, "e1": 9, "e2": 6, "e3": 10, "e4": 1, "e5": -1, "e6": -6, "e7": 3, "e8": 8, "e9": -9}'
    72df1858-661a-4a12-901a-40e85c3efe36 | task98  | TEST     | deephyper.benchmark.hps.polynome2.problem.run | RESTART_READY | 1         | '{"e0": 10, "e1": -9, "e2": 9, "e3": 8, "e4": 3, "e5": -10, "e6": -8, "e7": -7, "e8": 9, "e9": -2}'
    f8314971-c81c-404d-9863-d169ae147f08 | task99  | TEST     | deephyper.benchmark.hps.polynome2.problem.run | RESTART_READY | 1         | '{"e0": 10, "e1": 7, "e2": 6, "e3": 8, "e4": -7, "e5": 1, "e6": -9, "e7": 6, "e8": 9, "e9": -6}'
    de03094c-df47-4ce6-9426-df3f2e63a40a | test    | TEST     | AMBS                                          | JOB_FINISHED | 1         | --evaluator balsam --problem deephyper.benchmark.hps.polynome2.Problem --run deephyper.benchmark.hps.polynome2.run


As you can see some jobs of our *TEST* workflow are in state ``RESTART_READY``. According to the `Balsam documentation <https://balsam.readthedocs.io/en/latest/index.html>`_ the job will be run again if a new launcher with the *TEST* workflow is executed.

.. image:: https://balsam.readthedocs.io/en/latest/_images/state-flow.png


This is why you should delete all jobs of the same workflow if you want
to execute the same experiment again. To do so you can use::

    balsam rm jobs --name $name | --id $id

For example here I would do::

    balsam rm jobs --name test

Then::

    balsam rm jobs --name task

.. note::

    The ``--name`` argument is used to query jobs with a name including it.

You can also use balsam django models directly. First create a new
python script::

    vim script_with_django_model.py

Then add these code::

    import sys
    from balsam.launcher.dag import BalsamJob

    BalsamJob.objects.filter(name__contains=sys.argv[2], workflow=sys.argv[1]).delete()

then execute::

    python script_with_django_model.py TEST task

this previous command will delete all jobs with a name containing ``task``
from the ``TEST`` workflow. Indeed the previous command ``balsam rm jobs
--name $name`` was not filtering with respect to a specific workflow.
Hence if you have jobs with similar names such as ``task_$id``
(generic name for evaluations generated by search algorithms) they will
all be deleted.

