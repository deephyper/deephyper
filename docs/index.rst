**************************************************************************************************
DeepHyper: A Python Package for Massively Parallel Hyperparameter Optimization in Machine Learning
**************************************************************************************************

DeepHyper is first and foremost a hyperparameter optimization (HPO) library.
By leveraging this core HPO functionnality, DeepHyper also provides neural architecture search, multi-fidelity and ensemble capabilities. 
With DeepHyper, users can easily perform these tasks on a single machine or distributed across multiple machines, making it ideal for use in a variety of environments. 
Whether you're a beginner looking to optimize your machine learning models or an experienced data scientist looking to streamline your workflow, DeepHyper has something to offer. So why wait? Start using DeepHyper today and take your machine learning skills to the next level!

The package is organized around the following modules:

* :mod:`deephyper.analysis`: To analyse your results.
* :mod:`deephyper.ensemble`: To build ensembles of predictive models possibly with disentangled uncertainty quantification.
* :mod:`deephyper.evaluator`: To distribute the evaluation of tasks (e.g., training or inference).
* :mod:`deephyper.hpo`: To perform hyperparameter optimization (HPO) and neural architecture search (NAS).
* :mod:`deephyper.predictor`: To wrap predictive models from different libraries.
* :mod:`deephyper.stopper` : To apply multi-fidelity or early discarding strategies for hyperparameter optimization (HPO) and neural architecture search (NAS).

Quick Start
=========== 

The :ref:`pip installation <install-pip>` is recommended. It requires **Python >= 3.10**.

.. code-block:: python

    pip install deephyper


If you would like to install DeepHyper with its core set of machine learning features (Pytorch and Learning Curve Extrapolation) use the following command:

.. code-block:: python

    pip install "deephyper[core]"

More details about installation can be found on our :ref:`Installation <installation>` page.

We then present a simple example of how to use DeepHyper to optimize a black-box function with three hyperparameters: a real-valued parameter, a discrete parameter, and a categorical parameter.

To try this example, you can copy/paste the script and run it.

.. code-block:: python

    from deephyper.hpo import HpProblem, CBO
    from deephyper.evaluator import Evaluator


    def run(job):
        x = job.parameters["x"]
        b = job.parameters["b"]
        function = job.parameters["function"]

        if function == "linear":
            y = x + b
        elif function == "cubic":
            y = x**3 + b

        return y


    def optimize():
        problem = HpProblem()
        problem.add_hyperparameter((-10.0, 10.0), "x")
        problem.add_hyperparameter((0, 10), "b")
        problem.add_hyperparameter(["linear", "cubic"], "function")

        evaluator = Evaluator.create(run, method="process",
            method_kwargs={
                "num_workers": 2,
            },
        )

        search = CBO(problem, evaluator, random_state=42)
        results = search.search(max_evals=100)

        return results

    if __name__ == "__main__":
        results = optimize()
        print(results)

        row = results.iloc[-1]
        print("\nOptimum values")
        print("  function:", row["sol.p:function"])
        print("  x:", row["sol.p:x"])
        print("  b:", row["sol.p:b"])
        print("  y:", row["sol.objective"])

Running the example will output the results shown below. The best parameters are for the cubic function with ``x = 9.99`` and ``b = 10`` which produces ``y = 1009.87``.

.. code-block:: console

        p:b p:function       p:x    objective  job_id job_status  m:timestamp_submit  m:timestamp_gather  sol.p:b sol.p:function   sol.p:x  sol.objective
    0      7      cubic -1.103350     5.656803       0       DONE            0.011795            0.905777        3          cubic  8.374450     590.312101
    1      3      cubic  8.374450   590.312101       1       DONE            0.011875            0.906027        3          cubic  8.374450     590.312101
    2      6      cubic  4.680560   108.540056       2       DONE            0.917542            0.918856        3          cubic  8.374450     590.312101
    3      9     linear  8.787395    17.787395       3       DONE            0.917645            0.929052        3          cubic  8.374450     590.312101
    4      6      cubic  9.109560   761.948419       4       DONE            0.928757            0.938856        6          cubic  9.109560     761.948419
    ..   ...        ...       ...          ...     ...        ...                 ...                 ...      ...            ...       ...            ...
    96     9      cubic  9.998937  1008.681250      96       DONE           33.905465           34.311504       10          cubic  9.999978    1009.993395
    97    10      cubic  9.999485  1009.845416      97       DONE           34.311124           34.777270       10          cubic  9.999978    1009.993395
    98    10      cubic  9.996385  1008.915774      98       DONE           34.776732           35.236710       10          cubic  9.999978    1009.993395
    99    10      cubic  9.997400  1009.220073      99       DONE           35.236190           35.687774       10          cubic  9.999978    1009.993395
    100   10      cubic  9.999833  1009.949983     100       DONE           35.687380           36.111318       10          cubic  9.999978    1009.993395

    [101 rows x 12 columns]

    Optimum values
      function: cubic
      x: 9.99958232225758
      b: 10
      y: 1009.8747019108424

Let us now provide step-by-step details about this example.

The black-box function named ``run`` (it could be named anything but by convention we call it the ``run``-function) is defined by taking an input ``job`` that contains the different hyperparameters to optimize under ``job.parameters``. In our case, the function takes three hyperparameters: ``x``, ``b``, and ``function``. The ``run``-function returns a value ``y`` that is computed based on the values of the hyperparameters. The value of ``y`` is the objective value that we want to **maximize** (by convention we do maximization, to do minimization simply return the negative of your objective). The ``run``-function can be any computationally expensive function that you want to optimize. For example, it can be a simple Python execution, opening subprocesses, submitting a SLURM job, perfoming an HTTP request, etc. The search algorithms will learn to optimize the function just based on observed input hyperparameters and output values.

.. code-block:: python

    def run(job):
        x = job.parameters["x"]
        b = job.parameters["b"]
        function = job.parameters["function"]

        if function == "linear":
            y = x + b
        elif function == "cubic":
            y = x**3 + b

        return y

Then, we have the ``def optimize()`` function that defines the creation and execution of the search.

.. code-block:: python

    problem = HpProblem()
    problem.add_hyperparameter((-10.0, 10.0), "x") 
    problem.add_hyperparameter((0, 10), "b") 
    problem.add_hyperparameter(["linear", "cubic"], "function")

We start by defining the hyperparameter names, types and allowed ranges.
For this we create a :class:`deephyper.hpo.HpProblem` object.
We add to this problem three hyperparameters: ``"x"``, ``"b"`` and ``"function"``.
The ``"x"`` hyperparameter is defined by a continuous range between [-10, 10]. 
The float type of the bounds is important to infer the continuous type of the hyperparameter.
The ``"b"`` hyperparameter is defined by an integer range between [0, 10].
Similarly, the int type of the bounds is important to infer the discrete type of the hyperparameter.
Finally, the ``"function"`` hyperparameter is defined by a list of string values and it is therefore a categorical nominal hyperparameter (i.e., without order relation between its values).
The problem can be interactively printed ``print(problem)`` to review its definition:

.. code-block::

    Configuration space object:
        Hyperparameters:
            b, Type: UniformInteger, Range: [0, 10], Default: 5
            function, Type: Categorical, Choices: {linear, cubic}, Default: linear
            x, Type: UniformFloat, Range: [-10.0, 10.0], Default: 0.0


After the problem, we create a :class:`deephyper.evaluator.Evaluator` object.
The ``Evaluator`` is in charge of asynchronously distributing the computation of multiple calls to the ``run``-function.
It provides a simple interface ``Evaluator.submit(tasks)`` and ``tasks_done = Evaluator.gather()`` to perform asynchronous
calls to the ``run``-function.

.. code-block:: python

    evaluator = Evaluator.create(
        run,
        method="process",
        method_kwargs={
            "num_workers": 2,
        },
    )

The ``method="process"`` let us choose among available parallel backends. 
In this example, we picked the ``"process"`` method.
The ``method_kwargs`` let us configure the ``Evaluator`` a bit more. 
The keys of ``method_kwargs`` directly match the possible argument of the corresponding ``Evaluator`` subclass. 
Some of these arguments are common to all ``Evaluator`` methods and some are specific.
The :mod:`deephyper.evaluator` API reference can be used to review the available arguments of each subclass.
In our case, ``method="process"`` corresponds to the :class:`deephyper.evaluator.ProcessPoolEvaluator`.
The only argument set is ``"num_workers": 2`` to define two process-based workers for our ``Evaluator`` allowing 2 parallel calls to ``run``-function on a CPU with at least two hardware threads.

Finally, comes the last piece of the puzzle.
We create a :class:`deephyper.search.CBO` object for a Centralized Bayesian optimization with the ``problem`` and ``evaluator`` created previously.
All search methods are sublcasses of :class:`deephyper.hpo.Search`.
For reproducibility of this example we also set the ``random_state=42``.
Then, we execute the search by using the ``max_evals`` termination criterion to stop the search when ``max_evals`` results have been gathered.

.. code-block:: python

    search = CBO(problem, evaluator, random_state=42)
    results = search.search(max_evals=100)

The returned ``results`` is a Pandas DataFrame object that is also checkpointed locally in the current directory under ``results.csv`` (default value of the ``log_dir="."`` argument of ``Search`` subclasses).
This DataFrame contains 1 row per ``run``-function evaluation:

* the columns that start with ``p:`` are the hyperparameters.
* the ``objective`` is the returned values of the ``run``-function.
* the ``job_id`` is the ``Evaluator`` job id of the evaluation (an integer incremented by order of job creation). 
* the ``job_status`` is the ``Evaluator`` job status of the evaluation.
* the columns that start with ``m:`` are metadata of each evaluations. Some are added by DeepHyper but they can also be returned by the user as part of the ``run``-function returned value.
* the columns that start with ``sol.`` are the estimated solution according to the current solution selection method.

.. code-block:: console

        p:b p:function       p:x    objective  job_id job_status  m:timestamp_submit  m:timestamp_gather  sol.p:b sol.p:function   sol.p:x  sol.objective
    0      7      cubic -1.103350     5.656803       0       DONE            0.011795            0.905777        3          cubic  8.374450     590.312101
    1      3      cubic  8.374450   590.312101       1       DONE            0.011875            0.906027        3          cubic  8.374450     590.312101
    2      6      cubic  4.680560   108.540056       2       DONE            0.917542            0.918856        3          cubic  8.374450     590.312101
    3      9     linear  8.787395    17.787395       3       DONE            0.917645            0.929052        3          cubic  8.374450     590.312101
    4      6      cubic  9.109560   761.948419       4       DONE            0.928757            0.938856        6          cubic  9.109560     761.948419
    ..   ...        ...       ...          ...     ...        ...                 ...                 ...      ...            ...       ...            ...
    96     9      cubic  9.998937  1008.681250      96       DONE           33.905465           34.311504       10          cubic  9.999978    1009.993395
    97    10      cubic  9.999485  1009.845416      97       DONE           34.311124           34.777270       10          cubic  9.999978    1009.993395
    98    10      cubic  9.996385  1008.915774      98       DONE           34.776732           35.236710       10          cubic  9.999978    1009.993395
    99    10      cubic  9.997400  1009.220073      99       DONE           35.236190           35.687774       10          cubic  9.999978    1009.993395
    100   10      cubic  9.999833  1009.949983     100       DONE           35.687380           36.111318       10          cubic  9.999978    1009.993395


.. warning:: By convention in DeepHyper, all search algorithms are MAXIMIZING the objective function. If you want to MINIMIZE the objective function, you can simply return the negative of your objective value.

The next steps to learn more about DeepHyper is to follow our :ref:`Examples <examples>`.

Table of Contents
=================

.. toctree::
    :maxdepth: 2
    :titlesonly:
    :caption: Get Started

    Installation <install/index>
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
    CLI <_autosummary/deephyper.cli>
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
