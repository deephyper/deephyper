Data Parallelism
****************

The `Horovod software <https://github.com/horovod/horovod>`_ is used to do data parallel traininig with deep neural neworks. Data parallelism consists in spliting the original dataset in multiple parts and then performing an distributed computation of gradients as shown in the following image.

.. image:: ../_static/img/user_guides/data_parallelism/data-parallelism.png
   :scale: 100 %
   :alt: data parallelism with horovod
   :align: center

To use this feature the :class:`deephyper.evaluator.BalsamEvaluator` should be set for the search with ``--evaluator balsam``. Also, the ``-job-mode mpi`` has to be used when submitting the task with ``balsam submit-launch --job-mode mpi``. Then, 3 command line arguments are exposed to choose how to distribute the computation: ``--num-nodes-per-eval, --num-ranks-per-node, --num-threads-per-rank`` where:

- ``num-nodes-per-eval`` is the number of nodes used for each evaluation.
- ``num-ranks-per-node`` is the number of MPI ranks used for each evaluation.
- ``num-threads-per-rank`` is the number of threads per rank for each evaluation.

Neural Architecture Search (NAS)
================================

The available pipeline to use Horovod with NAS algorithms is ``deephyper.nas.run.horovod.run`` which has to be specified with the ``--run`` argument such as:

.. code-block:: console
    :caption: bash

    $ python -m deephyper.search.nas.regevo --problem deephyper.benchmark.nas.linearReg.Problem --evaluator balsam --max-evals 5 --num-nodes-per-eval 2 --num-ranks-per-node 1 --num-threads-per-rank 64 --run deephyper.nas.run.horovod.run

.. note::

    In the previous example we used a typical choice for Theta compute nodes by distributing the computation of each evalution on 2 nodes where each nodes has 64 threads.

Hyperparameter Search (HPS)
===========================

For HPS algorithm you simply have to follow the `Horovod documentation <https://horovod.readthedocs.io/>`_ to use it in the content of your ``run(...)`` function.