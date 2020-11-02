Algorithms
============

Regularized Evolution
---------------------

.. autoclass:: deephyper.search.nas.regevo.RegularizedEvolution
    :members:

If you want to run the aging evolution search:

.. code-block:: console
    :caption: bash

    deephyper nas regevo --problem deephyper.benchmark.nas.mnist1D.Problem


Asynchronous Model Based Neural Architecture Search (AMBNAS)
------------------------------------------------------------

.. autoclass:: deephyper.search.nas.ambs.AMBNeuralArchitectureSearch
    :members:

Random Search
-------------

.. autoclass:: deephyper.search.nas.random.Random
    :members:

If you want to run the random search:

.. code-block:: console
    :caption: bash

    deephyper nas random --problem deephyper.benchmark.nas.linearReg.Problem
