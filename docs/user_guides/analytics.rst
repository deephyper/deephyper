DeepHyper Analytics Tools
*************************

The Analytics command line is a set of tools which has been created to help you analyse DeepHyper outputs.

.. highlight:: console

Let's look at the help menu of ``deephyper-analytics``::

    deephyper-analytics --help

The following output is expected::

    Command line to analysis the outputs produced by DeepHyper.

    positional arguments:
    {notebook,parse,quickplot,topk}
                            Kind of analytics.
        notebook            Generate a notebook with different types of analysis
        parse               Tool to parse "deephyper.log" and produce a JSON file.
        quickplot           Tool to generate a quick 2D plot from file.
        topk                Print the top-k configurations.
        balsam              Extract information from Balsam jobs.

    optional arguments:
    -h, --help            show this help message and exit


Notebooks
=========

Hyperparameter search
---------------------

.. automodule:: deephyper.core.plot.hps

Neural architecture search
--------------------------

.. automodule:: deephyper.core.plot.single

.. automodule:: deephyper.core.plot.multi

Parsing Logs
============

.. warning::

    This tool is deprecated and is replaced by the ``save`` folder and the ``results.csv`` file.

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

Quick Plots
===========

.. automodule:: deephyper.core.plot.quick_plot

Top-k Configuration
===================

.. automodule:: deephyper.core.logs.topk

Balsam
======

.. automodule:: deephyper.core.logs.balsam