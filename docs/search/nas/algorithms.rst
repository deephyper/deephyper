Algorithms
============

Proximal Policy Optimization
------------------------------

.. autoclass:: deephyper.search.nas.ppo.Ppo
    :members:

If you want to run the proximal policy optimization search without MPI:

.. code-block:: console
    :caption: bash

    deephyper nas ppo --problem deephyper.benchmark.nas.mnist1D.Problem


and with MPI (i.e. several agents):

.. code-block:: console
    :caption: bash

    mpirun -np 2 deephyper nas ppo --problem deephyper.benchmark.nas.mnist1D.Problem

Regularized Evolution
------------------------------

.. autoclass:: deephyper.search.nas.regevo.RegularizedEvolution
    :members:

If you want to run the aging evolution search:

.. code-block:: console
    :caption: bash

    deephyper nas regevo --problem deephyper.benchmark.nas.mnist1D.Problem


Asynchronous Model Based Neural Architecture Search (AMBNAS)
------------------------------

.. autoclass:: deephyper.search.nas.ambs.AMBNeuralArchitectureSearch
    :members:

Random Search
------------------------------

.. autoclass:: deephyper.search.nas.full_random.Random
    :members:

If you want to run the random search:

.. code-block:: console
    :caption: bash

    deephyper nas random --problem deephyper.benchmark.nas.linearReg.Problem
