.. _evaluators:

**********************
Evaluator Interface
**********************

.. automodule:: deephyper.evaluator

.. image:: ../_static/img/software/evaluators/evaluators-diag.png
   :scale: 25 %
   :alt: evaluator diag
   :align: center


Evaluator
=========

.. autoclass:: deephyper.evaluator.evaluate.Evaluator

.. _balsam-evaluator:

BalsamEvaluator
===============

.. autoclass:: deephyper.evaluator._balsam.BalsamEvaluator

.. _ray-evaluator:

RayEvaluator
=============

.. autoclass:: deephyper.evaluator.ray_evaluator.RayEvaluator


.. _subprocess-evaluator:

SubprocessEvaluator
===================

.. autoclass:: deephyper.evaluator._subprocess.SubprocessEvaluator


ProcessPoolEvaluator
====================

.. autoclass:: deephyper.evaluator._processPool.ProcessPoolEvaluator


ThreadPoolEvaluator
===================

.. autoclass:: deephyper.evaluator._threadPool.ThreadPoolEvaluator

.. warning::
    For ThreadPoolEvaluator, note that this does not mean that they are executed on different CPUs. Python threads will NOT make your program faster if it already uses 100 % CPU time. Python threads are used in cases where the execution of a task involves some waiting. One example would be interaction with a service hosted on another computer, such as a webserver. Threading allows python to execute other code while waiting; this is easily simulated with the sleep function. (from: https://en.wikibooks.org/wiki/Python_Programming/Threading)
