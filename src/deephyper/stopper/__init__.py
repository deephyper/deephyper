"""Multi-fidelity and early discarding subpackage.

This module provides features to observe intermediate performances of
iterative algorithm and decide dynamically if its evaluation should be
stopped or continued.

This module was inspired from the Pruner interface and implementation of
`Optuna <https://optuna.readthedocs.io/en/stable/reference/pruners.html>`_.

The ``Stopper`` class is the base class for all stoppers. It provides the
interface for the ``observe`` and ``stop`` methods that should be implemented
by all stoppers. The ``observe`` method is called at each iteration of the
iterative algorithm and the ``stop`` method is called at the end of each
iteration to decide if the evaluation should be stopped or continued. The
stopper object is not used directly but through the ``RunningJob`` received
by the ``run``-function. In the following example we demonstrate with a
simulation how it can be used:

.. code-block:: python

    import time

    from deephyper.hpo import HpProblem
    from deephyper.hpo import CBO
    from deephyper.stopper import SuccessiveHalvingStopper


    def run(job):

        x = job.parameters["x"]

        # Simulation of iteration
        cum = 0
        for i in range(100):
            cum += x
            time.sleep(0.01) # each iteration cost 0.1 secondes

            # Record the intermediate performance
            # Calling stopper.observe(budget, objective) under the hood
            job.record(budget=i + 1, objective=cum)

            # Check if the evaluation should be stopped
            # Calling stopper.stop() under the hood
            if job.stopped():
                break

        # Return objective and metadata to save what is the maximum step reached
        return {"objective": cum, "metadata": {"i_stopped": i}}


    problem = HpProblem()
    problem.add_hyperparameter((0.0, 100.0), "x")

    stopper = SuccessiveHalvingStopper(min_steps=1, max_steps=100)
    search =  CBO(problem, run, stopper=stopper, log_dir="multi-fidelity-exp")
    results = search.search(timeout=10)


As it can be observed in the following results many evaluation stopped after
the first iteration which saved a lot of computation time. If evaluated
fully, each configuration would take about 1 seconds and we would be able to
compute only a maximum of 10 configurations (because we set a timeout of 10).
However, with the stopper we managed to perform 15 evaluations instead.

.. code-block:: console

              p:x    objective  job_id  m:timestamp_submit  m:timestamp_gather  m:i_stopped
    0   79.654299  7965.429869       0            0.016269            1.234227           99
    1   74.266072    74.266072       1            1.256349            1.269175            0
    2   74.491125    74.491125       2            1.281712            1.294496            0
    3   10.245385    10.245385       3            1.305979            1.317513            0
    4    4.229917     4.229917       4            1.417226            1.430005            0
    5   53.690895    53.690895       5            1.437582            1.450419            0
    6   54.902216    54.902216       6            1.458042            1.470806            0
    7   22.945529    22.945529       7            1.478365            1.491140            0
    8   94.051310  9405.130978       8            1.498538            2.733619           99
    9   23.024237    23.024237       9            2.753319            2.766194            0
    10  97.121528  9712.152792      10            2.884685            4.114600           99
    11  97.192445  9719.244491      11            4.241939            5.467425           99
    12  98.844525  9884.452486      12            5.598530            6.833938           99
    13  99.722437  9972.243688      13            6.946300            8.172941           99
    14  99.988566  9998.856623      14            8.376363            9.615355           99
"""

from deephyper.stopper._stopper import Stopper
from deephyper.stopper._asha_stopper import SuccessiveHalvingStopper
from deephyper.stopper._median_stopper import MedianStopper
from deephyper.stopper._idle_stopper import IdleStopper
from deephyper.stopper._const_stopper import ConstantStopper


__all__ = [
    "IdleStopper",
    "Stopper",
    "SuccessiveHalvingStopper",
    "MedianStopper",
    "ConstantStopper",
]


try:
    from deephyper.stopper._lcmodel_stopper import LCModelStopper  # noqa: F401

    __all__.append("LCModelStopper")
except ImportError:
    pass
