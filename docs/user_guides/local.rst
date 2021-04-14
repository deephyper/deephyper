Running locally
***************

This section will show you how to run Hyperparameter or neural architecture on your client machine, like a laptop.
All searches can be invoked from either the shell or the Python API.

.. note::

    To invoke HPS or NAS from Python, use DeepHyper's ``Search`` API (see :class:`deephyper.search.Search`).


.. note::

    You can use the ``--help`` argument to see valid options for any DeepHyper subcommand. For
    instance::

        $ deephyper hps ambs --help

Hyperparameter search
=====================

Let's run asynchronous model-based search (AMBS) on the ``polynome2`` benchmark installed with DeepHyper: ::

    deephyper hps ambs --problem deephyper.benchmark.hps.polynome2.Problem --run deephyper.benchmark.hps.polynome2.run



Neural Architecture Search
==========================

::

    deephyper nas regevo --problem deephyper.benchmark.nas.linearReg.Problem