Introduction
************

.. automodule:: deephyper.evaluator

.. image:: ../_static/img/evaluator/evaluators-diag.png
   :scale: 25 %
   :alt: evaluator diag
   :align: center


BalsamEvaluator
***************

.. autoclass:: deephyper.evaluator._balsam.BalsamEvaluator


SubprocessEvaluator
*******************

.. autoclass:: deephyper.evaluator._subprocess.SubprocessEvaluator


ProcessPoolEvaluator
********************

.. autoclass:: deephyper.evaluator._processPool.ProcessPoolEvaluator


ThreadPoolEvaluator
*******************

.. autoclass:: deephyper.evaluator._threadPool.ThreadPoolEvaluator

.. warning::
    For ThreadPoolEvaluator, note that this does not mean that they are executed on different CPUs. Python threads will NOT make your program faster if it already uses 100 % CPU time. Python threads are used in cases where the execution of a task involves some waiting. One example would be interaction with a service hosted on another computer, such as a webserver. Threading allows python to execute other code while waiting; this is easily simulated with the sleep function. (from: https://en.wikibooks.org/wiki/Python_Programming/Threading)
