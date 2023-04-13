Software Architecture
*********************

The architecture of DeepHyper is based on the following components:

.. image:: ../_static/figures/deephyper-generic-components-architecture.png
    :scale: 25%
    :alt: deephyper generic components
    :align: center

The blue boxes are components defined by the user. The orange boxes are the components which can be configured by the user but not necessarily. The arrows represents the data flow. 

First, the user must provides a **search space** and an **objective function**. The search space is defined through the :mod:`deephyper.problem` module (with :class:`deephyper.problem.HpProblem` for hyperparameter optimization and :class:`deephyper.problem.NaProblem` for neural architecture search). The objective function is simply a Python function which returns the objective to maximize during optimization. This is where the logic to evaluate a suggested configuration is happening. It is commonly named the ``run``-function accross the documentation. This ``run``-function must follow some standards which are detailed in the :mod:`deephyper.evaluator` module.

Then, the user can choose how to **distribute the computation** of suggested tasks in parallel. This distributed computation is abstracted through the :class:`deephyper.evaluator.Evaluator` interface which provides the ``evaluator.submit(configurations)`` and ``results = evaluator.gather(...)`` methods. A panel of different backends is provided: serial (similar to sequential execution in local process), threads, process, MPI and Ray. This interface to evaluate tasks in parallel is possibly synchronous or asynchronous by batch. Also, the :class:`deephyper.evaluator.Evaluator` uses the :class:`deephyper.evaluator.storage.Storage` interface to record and retrieve jobs metadata. A panel of different storage is provided: local memory, Redis and Ray.

Finally, the user can choose a **search strategy** to suggest new configurations to evaluate. These strategies are defined in the :mod:`deephyper.search` module and vary depending if the problem is for hyperparameter optimization (:mod:`deephyper.search.hps`) or neural architecture search (:mod:`deephyper.search.hps`).