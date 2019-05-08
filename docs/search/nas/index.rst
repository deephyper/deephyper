Neural Architecture Search (NAS)
********************************


.. toctree::
   :maxdepth: 2

   env/index
   model/index

NAS (PPO) Asynchronous
======================

.. autoclass:: deephyper.search.nas.ppo.Ppo
   :members:

Run locally
-----------

With n agent where n = np - 1, because 1 mpi process is used for the parameter server.

::

    mpirun -np 2 python python -m deephyper.search.nas.ppo --problem deephyper.benchmark.nas.mnist1D.problem.Problem --run deephyper.search.nas.model.run.alpha.run --evaluator subprocess


NAS Full Random
===============

.. autoclass:: deephyper.search.nas.full_random.Random
   :members:
