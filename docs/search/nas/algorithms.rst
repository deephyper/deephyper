Algorithms
**********

.. autoclass:: deephyper.search.nas.rl.ReinforcementLearningSearch
    :members:

NAS (PPO) Asynchronous
======================

.. autoclass:: deephyper.search.nas.ppo.Ppo
    :members:

Run locally
-----------

Without MPI:

::

    $ python -m deephyper.search.nas.ppo --problem deephyper.benchmark.nas.mnist1D.Problem --run deephyper.search.nas.model.run.alpha.run


With MPI (i.e. several agents):

::

    $ mpirun -np 2 python -m deephyper.search.nas.ppo --problem deephyper.benchmark.nas.mnist1D.Problem --run deephyper.search.nas.model.run.alpha.run

Asynchronous Model Based Neural Architecture Search (AMBNAS)
============================================================

.. autoclass:: deephyper.search.nas.ambs.AMBNeuralArchitectureSearch
    :memeber:

NAS Full Random
===============

.. autoclass:: deephyper.search.nas.full_random.Random
   :members:


Run locally
-----------

There isn't any MPI implementation for the full random search.

::

    $ python -m deephyper.search.nas.full_random --problem deephyper.benchmark.nas.linearReg.Problem --run deephyper.search.nas.model.run.alpha.run
