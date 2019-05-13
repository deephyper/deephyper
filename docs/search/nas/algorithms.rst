Algorithms
**********

.. autoclass:: deephyper.search.nas.NeuralArchitectureSearch
   :members:

NAS (PPO) Asynchronous
======================

.. autoclass:: deephyper.search.nas.ppo.Ppo
   :members:

Run locally
-----------

Without MPI:

::

    python -m deephyper.search.nas.ppo --problem deephyper.benchmark.nas.mnist1D.Problem --run deephyper.search.nas.model.run.alpha.run


With MPI (i.e. several agents):

::

    mpirun -np 2 python -m deephyper.search.nas.ppo --problem deephyper.benchmark.nas.mnist1D.Problem --run deephyper.search.nas.model.run.alpha.run


NAS Full Random
===============

.. autoclass:: deephyper.search.nas.full_random.Random
   :members:
