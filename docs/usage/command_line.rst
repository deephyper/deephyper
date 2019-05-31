Analytics
*********

The Analytics command line is a set of tools which has been created to help you analyse deephyper data.

.. highlight:: console

::

    [BalsamDB: testdb] (dh-opt) dhuser $ deephyper-analytics --help
    usage: deephyper-analytics [-h] {parse,json} ...

    Run some analytics for deephyper.

    positional arguments:
    {parse,json}  Kind of analytics.
        parse       Tool to parse "deephyper.log" and produce a JSON file.
        json        Tool to analyse a JSON file produced by the "parse" tool.

    optional arguments:
    -h, --help    show this help message and exit


Parsing logs
============

The parsing tool helps you to parse the ``deephyper.log`` file of your master job. For now this tool is used for neural architecture search logs.

::

    [BalsamDB: testdb] (dh-opt) dhuser $ deephyper-analytics parse -h
    Module: 'balsam' was not found!
    usage: deephyper-analytics parse [-h] path

    positional arguments:
    path        The parsing script takes only 1 argument: the relative path to
                the log file. If you want to compute the workload data with
                'balsam' you should specify a path starting at least from the
                workload parent directory, eg.
                'nas_exp1/nas_exp1_ue28s2k0/deephyper.log' where 'nas_exp1' is
                the workload.

    optional arguments:
    -h, --help  show this help message and exit

::

    [BalsamDB: testdb] (dh-opt) dhuser $ deephyper-analytics parse ../nasdb/data/combo_async_exp4/combo_async_exp4_b0c432c7/deephyper.log
    Module: 'balsam' has been loaded successfully!
    Path to deephyper.log file: ../nasdb/data/combo_async_exp4/combo_async_exp4_b0c432c7/deephyper.log
    File has been opened
    File closed
    Computing workload!
    Workload has been computed successfuly!
    Create json file: combo_async_exp4_2019-02-28_08.json
    Json dumped!
    len raw_rewards: 5731
    len max_rewards: 5731
    len id_worker  : 5731
    len arch_seq   : 5731

Transformations from JSON file
==============================

::

    [BalsamDB: testdb] (dh-opt) dhuser $ deephyper-analytics json -h
    usage: deephyper-analytics json [-h] {best} ...

    positional arguments:
    {best}      Kind of analytics.
        best      Select the best n architectures and save them into a JSON file.

    optional arguments:
    -h, --help  show this help message and exit

Single study
============

.. automodule:: deephyper.core.plot.single

Multiple study
==============

.. automodule:: deephyper.core.plot.multi