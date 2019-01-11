Introduction
************

.. automodule:: deephyper.benchmark

Problem
*******

This class describe the most generic aspect of a problem. Basically we are using a python ``dict`` and adding key-values. It is mostly used for neural architecture search problems, see :ref:`create-new-nas-problem` for more details.

.. autoclass:: deephyper.benchmark.problem.Problem
   :members:


Hyperparameter Search Problem
*****************************

Use this class to define a hyperparameter search problem, see :ref:`create-new-hps-problem` for more details.

.. autoclass:: deephyper.benchmark.problem.HpProblem
   :members:
