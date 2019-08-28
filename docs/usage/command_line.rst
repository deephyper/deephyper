Command line
************

For hyperparameter search use ``deephyper hps ...``. If you want to run the
polynome2 hyperparameter search benchmark with the asynchronous model-based
search do:

.. code-block:: console
    :caption: bash

    deephyper hps ambs --problem deephyper.benchmark.hps.polynome2.Problem --run deephyper.benchmark.hps.polynome2.run


For neural architecture search use ``deephyper nas ...``. If you want to run
the linearReg neural architecture search benchmark with regularized evolution
do:

.. code-block:: console
    :caption: bash

    deephyper nas regevo --problem deephyper.benchmark.nas.linearReg.Problem

If you want to initialize an hyperparameter or neural architecture search
problem folder you can use ``deephyper hps-init ...`` or
``deephyper nas-init ...`` respectively.

Use commands such as ``deephyper --help``, ``deephyper nas --help`` or
``deephyper nas regevo --help`` to find out more about the command line
interface.
