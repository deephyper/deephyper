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
===========

.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/deephyper/tutorials/blob/main/tutorials/colab/DeepHyper_101.ipynb
   :alt: Open In Colab
   :align: center

The black-box function named ``run`` is defined by taking an input dictionnary named ``config`` which contains the different variables to optimize. Then the run-function is binded to an ``Evaluator`` in charge of distributing the computation of multiple evaluations. Finally, a Bayesian search named ``CBO`` is created and executed to find the values of config which maximize the return value of ``run(config)``.

.. code-block:: python

    def run(config: dict):
        return -config["x"]**2


    # Necessary IF statement otherwise it will enter in a infinite loop
    # when loading the 'run' function from a subprocess
    if __name__ == "__main__":
        from deephyper.problem import HpProblem
        from deephyper.search.hps import CBO
        from deephyper.evaluator import Evaluator

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
        search = CBO(problem, evaluator)

        results = search.search(max_evals=100)
        print(results)

Which outputs the following where the best ``x`` found is clearly around ``0``.

.. code-block:: console

            x  job_id     objective  timestamp_submit  timestamp_gather
    0  -7.744105       1 -5.997117e+01          0.011047          0.037649
    1  -9.058254       2 -8.205196e+01          0.011054          0.056398
    2  -1.959750       3 -3.840621e+00          0.049750          0.073166
    3  -5.150553       4 -2.652819e+01          0.065681          0.089355
    4  -6.697095       5 -4.485108e+01          0.082465          0.158050
    ..       ...     ...           ...               ...               ...
    95 -0.034096      96 -1.162566e-03         26.479630         26.795639
    96 -0.034204      97 -1.169901e-03         26.789255         27.155481
    97 -0.037873      98 -1.434366e-03         27.148506         27.466934
    98 -0.000073      99 -5.387088e-09         27.460253         27.774704
    99  0.697162     100 -4.860350e-01         27.768153         28.142431


Table of Contents
=================

.. toctree::
    :maxdepth: 2
    :titlesonly:
    :caption: Get Started

    install/index
    tutorials/index
    examples/index
    research
    Authors <authors>


.. toctree::
    :caption: API Reference

    Core <_autosummary/deephyper.core>
    Ensemble <_autosummary/deephyper.ensemble>
    Evaluator <_autosummary/deephyper.evaluator>
    Keras <_autosummary/deephyper.keras>
    NAS <_autosummary/deephyper.nas>
    Problem <_autosummary/deephyper.problem>
    Search <_autosummary/deephyper.search>
    Sklearn <_autosummary/deephyper.sklearn>

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