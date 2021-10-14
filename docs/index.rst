.. deephyper documentation master file, created by
    sphinx-quickstart on Thu Sep 27 13:32:19 2018.
    You can adapt this file completely to your liking, but it should at least
    contain the root `toctree` directive.

******************************************************************************************
DeepHyper: Scalable Neural Architecture and Hyperparameter Search for Deep Neural Networks
******************************************************************************************

.. image:: _static/logo/medium.png
    :scale: 100%
    :alt: logo
    :align: center


.. automodule:: deephyper


Quick Start
-----------

The black-box function named ``run`` is defined by taking an input dictionnary named ``config`` which contains the different variables to optimize. Then the run-function is binded to an ``Evaluator`` in charge of distributing the computation of multiple evaluations. Finally, a Bayesian search named ``AMBS`` is created and executed to find the values of config which maximize the return value of ``run(config)``.

.. code-block:: python

    def run(config: dict):
        return -config["x"]**2


    # Necessary IF statement otherwise it will enter in a infinite loop
    # when loading the 'run' function from a subprocess
    if __name__ == "__main__":
        from deephyper.problem import HpProblem
        from deephyper.search.hps import AMBS
        from deephyper.evaluator.evaluate import Evaluator

        # define the variable you want to optimize
        problem = HpProblem()
        problem.add_hyperparameter((-10.0, 10.0), "x")

        # define the evaluator to distribute the computation
        evaluator = Evaluator.create(
            run,
            method="subprocess",
            method_kwargs={
                "num_workers": 2,
            },
        )

        # define your search and execute it
        search = AMBS(problem, evaluator)

        results = search.search(max_evals=100)
        print(results)

Which outputs the following where the best ``x`` found is clearly around ``0``.

.. code-block:: console

               x  id  objective  elapsed_sec  duration
    0   1.667375   1  -2.780140     0.124388  0.071422
    1   9.382053   2 -88.022911     0.124440  0.071465
    2   0.247856   3  -0.061433     0.264603  0.030261
    3   5.237798   4 -27.434527     0.345482  0.111113
    4   5.168073   5 -26.708983     0.514158  0.175257
    ..       ...  ..        ...          ...       ...
    94  0.024265  95  -0.000589     9.261396  0.117477
    95 -0.055000  96  -0.003025     9.367814  0.113984
    96 -0.062223  97  -0.003872     9.461532  0.101337
    97 -0.016222  98  -0.000263     9.551584  0.096401
    98  0.009660  99  -0.000093     9.638016  0.092450


.. toctree::
    :maxdepth: 2
    :titlesonly:
    :caption: Get Started

    install/index
    tutorials/index
    research


.. toctree::
    :maxdepth: 2
    :caption: API Reference

    Core <api/deephyper.core>
    Ensemble <api/deephyper.ensemble>
    Evaluator <api/deephyper.evaluator>
    NAS <api/deephyper.nas>
    Problem <api/deephyper.problem>
    Search <api/deephyper.search>
    Sklearn <api/deephyper.sklearn>

.. toctree::
    :maxdepth: 2
    :caption: Developer Guides
    :glob:

    developer_guides/dev
    developer_guides/doc
    developer_guides/tests_link

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
