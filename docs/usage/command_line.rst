Command Line
************

Analytics
=========

::

    $ deephyper-analytics --help
    usage: deephyper-analytics [-h] {parse,json} ...

    Run some analytics for deephyper.

    positional arguments:
    {parse,json}  Kind of analytics.
        parse       Tool to parse "deephyper.log" and produce a JSON file.
        json        Tool to analyse a JSON file produced by the "parse" tool.

    optional arguments:
    -h, --help    show this help message and exit


Parsing
-------

::

    $ deephyper-analytics parse -h
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

Json
----

::
    $ deephyper-analytics json -h
    usage: deephyper-analytics json [-h] {best} ...

    positional arguments:
    {best}      Kind of analytics.
        best      Select the best n architectures and save them into a JSON file.

    optional arguments:
    -h, --help  show this help message and exit

