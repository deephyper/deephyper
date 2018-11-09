Evaluators
==========

The goal off the evaluator module is to have a set of objects which can helps us to run our task on different environments and with different system settings/properties.

.. image:: ../_static/img/evaluators/evaluators-diag.png
   :scale: 25 %
   :alt: evaluators diag
   :align: center

.. autoclass:: deephyper.evaluators._balsam.BalsamEvaluator

.. autoclass:: deephyper.evaluators._subprocess.SubprocessEvaluator

.. autoclass:: deephyper.evaluators._processPool.ProcessPoolEvaluator

.. autoclass:: deephyper.evaluators._threadPool.ThreadPoolEvaluator

.. warning::
    For ThreadPoolEvaluator, note that this does not mean that they are executed on different CPUs. Python threads will NOT make your program faster if it already uses 100 % CPU time. Python threads are used in cases where the execution of a task involves some waiting. One example would be interaction with a service hosted on another computer, such as a webserver. Threading allows python to execute other code while waiting; this is easily simulated with the sleep function. (from: https://en.wikibooks.org/wiki/Python_Programming/Threading)
