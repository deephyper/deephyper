***********************************************************************************************
DeepHyper: Distributed Neural Architecture and Hyperparameter Optimization for Machine Learning
***********************************************************************************************

DeepHyper is a powerful Python package for automating machine learning tasks, particularly focused on optimizing hyperparameters, searching for optimal neural architectures, and quantifying uncertainty through the deep ensembles. With DeepHyper, users can easily perform these tasks on a single machine or distributed across multiple machines, making it ideal for use in a variety of environments. Whether you're a beginner looking to optimize your machine learning models or an experienced data scientist looking to streamline your workflow, DeepHyper has something to offer. So why wait? Start using DeepHyper today and take your machine learning skills to the next level!

DeepHyper is specialized for machine learning tasks but it can also be used for generic black-box and gray-box optimization problems of expensive functions.

DeepHyper's software architecture is designed to be modular and extensible.

It is organized around the following subpackages:

* :mod:`deephyper.analysis`: To analyse your results.
* :mod:`deephyper.ensemble`: To build ensembles of predictive models possibly with disentangled uncertainty quantification.
* :mod:`deephyper.evaluator`: To distribute the evaluation of tasks (e.g., training or inference).
* :mod:`deephyper.hpo`: To perform hyperparameter optimization (HPO) and neural architecture search (NAS).
* :mod:`deephyper.predictor`: To wrap predictive models from different libraries.
* :mod:`deephyper.stopper` : To apply multi-fidelity or early discarding strategies for hyperparameter optimization (HPO) and neural architecture search (NAS).

DeepHyper installation requires **Python >= 3.10**.

Install instructions
====================
Install with ``pip``

.. code-block:: python

    # For the most basic set of features (hyperparameter search)
    pip install deephyper

    # For the default set of features including:
    # - hyperparameter search with transfer-learning
    # - neural architecture search
    # - deep ensembles
    # - Ray-based distributed computing
    # - Learning-curve extrapolation for multi-fidelity hyperparameter search
    pip install "deephyper[default]"

More details about the installation process can be found at `DeepHyper Installations <https://deephyper.readthedocs.io/en/latest/install/index.html>`_.


Quick Start
===========

.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/deephyper/tutorials/blob/main/tutorials/colab/DeepHyper_101.ipynb
   :alt: Open In Colab
   :align: center

The black-box function named ``run`` is defined by taking an input dictionnary named ``config`` which contains the different variables to optimize. Then the run-function is binded to an ``Evaluator`` in charge of distributing the computation of multiple evaluations. Finally, a Bayesian search named ``CBO`` is created and executed to find the values of config which **MAXIMIZE** the return value of ``run(config)``.

.. code-block:: python

    def run(job):
        # The suggested parameters are accessible in job.parameters (dict)
        x = job.parameters["x"]
        b = job.parameters["b"]

        if job.parameters["function"] == "linear":
            y = x + b
        elif job.parameters["function"] == "cubic":
            y = x**3 + b

        # Maximization!
        return y


    # Necessary IF statement otherwise it will enter in a infinite loop
    # when loading the 'run' function from a new process
    if __name__ == "__main__":
        from deephyper.hpo import HpProblem, CBO
        from deephyper.evaluator import Evaluator

        # define the variable you want to optimize
        problem = HpProblem()
        problem.add_hyperparameter((-10.0, 10.0), "x") # real parameter
        problem.add_hyperparameter((0, 10), "b") # discrete parameter
        problem.add_hyperparameter(["linear", "cubic"], "function") # categorical parameter

        # define the evaluator to distribute the computation
        evaluator = Evaluator.create(
            run,
            method="process",
            method_kwargs={
                "num_workers": 2,
            },
        )

        # define your search and execute it
        search = CBO(problem, evaluator, random_state=42)

        results = search.search(max_evals=100)
        print(results)

Which outputs the following results where the best parameters are with ``function == "cubic"``, ``x == 9.99`` and ``b == 10``.

.. code-block:: console

        p:b p:function       p:x    objective  job_id  m:timestamp_submit  m:timestamp_gather
    0     7     linear  8.831019    15.831019       1            0.064874            1.430992
    1     4     linear  9.788889    13.788889       0            0.064862            1.453012
    2     0      cubic  2.144989     9.869049       2            1.452692            1.468436
    3     9     linear -9.236860    -0.236860       3            1.468123            1.483654
    4     2      cubic -9.783865  -934.550818       4            1.483340            1.588162
    ..  ...        ...       ...          ...     ...                 ...                 ...
    95    6      cubic  9.862098   965.197192      95           13.538506           13.671872
    96   10      cubic  9.997512  1009.253866      96           13.671596           13.884530
    97    6      cubic  9.965615   995.719961      97           13.884188           14.020144
    98    5      cubic  9.998324  1004.497422      98           14.019737           14.154467
    99    9      cubic  9.995800  1007.740379      99           14.154169           14.289366

The code defines a function ``run`` that takes a RunningJob ``job`` as input and returns the maximized objective ``y``. The ``if`` block at the end of the code defines a black-box optimization process using the ``CBO`` (Centralized Bayesian Optimization) algorithm from the ``deephyper`` library.

The optimization process is defined as follows:

1. A hyperparameter optimization problem is created using the ``HpProblem`` class from ``deephyper``. In this case, the problem has a three variables. The ``x`` hyperparameter is a real variable in a range from -10.0 to 10.0. The ``b`` hyperparameter is a discrete variable in a range from 0 to 10. The ``function`` hyperparameter is a categorical variable with two possible values.

2. An evaluator is created using the ``Evaluator.create`` method. The evaluator will be used to evaluate the function ``run`` with different configurations of suggested hyperparameters in the optimization problem. The evaluator uses the ``process`` method to distribute the evaluations across multiple worker processes, in this case 2 worker processes.
   
3. A search object is created using the ``CBO`` class, the problem and evaluator defined earlier. The ``CBO`` algorithm is a derivative-free optimization method that uses a Bayesian optimization approach to explore the hyperparameter space.
   
4. The optimization process is executed by calling the ``search.search`` method, which performs the evaluations of the ``run`` function with different configurations of the hyperparameters until a maximum number of evaluations (100 in this case) is reached.
   
5. The results of the optimization process, including the optimal configuration of the hyperparameters and the corresponding objective value, are printed to the console.

.. warning:: All search algorithms are MAXIMIZING the objective function. If you want to MINIMIZE the objective function, you have to return the negative of you objective.

Table of Contents
=================

.. toctree::
    :maxdepth: 2
    :titlesonly:
    :caption: Get Started

    Install <install/index>
    tutorials/index
    examples/index
    F.A.Q. <faq>
    Blog (Events & Workshops) <https://deephyper.github.io>
    Publications <https://deephyper.github.io/papers>
    Authors <https://deephyper.github.io/aboutus>


.. toctree::
    :caption: API Reference
    :maxdepth: 1
    :titlesonly:

    Analysis <_autosummary/deephyper.analysis>
    Core <_autosummary/deephyper.core>
    Ensemble <_autosummary/deephyper.ensemble>
    Evaluator <_autosummary/deephyper.evaluator>
    HPO <_autosummary/deephyper.hpo>
    Predictor <_autosummary/deephyper.predictor>
    Skopt <_autosummary/deephyper.skopt>
    Stopper <_autosummary/deephyper.stopper>

.. toctree::
    :maxdepth: 2
    :caption: Developer's Guide

    developer_guides/contributing
    developer_guides/software_architecture
    


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
