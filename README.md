<p align="center">
<img src="docs/_static/logo/medium.png">
</p>

[![DOI](https://zenodo.org/badge/156403341.svg)](https://zenodo.org/badge/latestdoi/156403341)
![GitHub tag (latest by date)](https://img.shields.io/github/tag-date/deephyper/deephyper.svg?label=version)
[![Documentation Status](https://readthedocs.org/projects/deephyper/badge/?version=latest)](https://deephyper.readthedocs.io/en/latest/?badge=latest)
![PyPI - License](https://img.shields.io/pypi/l/deephyper.svg)
![PyPI - Downloads](https://img.shields.io/pypi/dm/deephyper.svg?label=Pypi%20downloads)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deephyper/tutorials/blob/main/tutorials/colab/DeepHyper_101.ipynb)
<!-- [![Build Status](https://travis-ci.com/deephyper/deephyper.svg?branch=develop)](https://travis-ci.com/deephyper/deephyper) -->

## What is DeepHyper?

DeepHyper is a powerful Python package for automating machine learning tasks, particularly focused on optimizing hyperparameters, searching for optimal neural architectures, and quantifying uncertainty through the use of deep ensembles. With DeepHyper, users can easily perform these tasks on a single machine or distributed across multiple machines, making it ideal for use in a variety of environments. Whether you're a beginner looking to optimize your machine learning models or an experienced data scientist looking to streamline your workflow, DeepHyper has something to offer. So why wait? Start using DeepHyper today and take your machine learning skills to the next level!

## Install instructions

Installation with `pip`:

```console
# For the most basic set of features (hyperparameter search)
pip install deephyper

# For the default set of features including:
# - hyperparameter search with transfer-learning
# - neural architecture search
# - deep ensembles
# - Ray-based distributed computing
# - Learning-curve extrapolation for multi-fidelity hyperparameter search
pip install "deephyper[default]"
```

More details about the installation process can be found at [DeepHyper Installations](https://deephyper.readthedocs.io/en/latest/install/index.html).

## Quickstart

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deephyper/tutorials/blob/main/tutorials/colab/DeepHyper_101.ipynb)

The black-box function named `run` is defined by taking an input dictionnary named `config` which contains the different variables to optimize. Then the run-function is binded to an `Evaluator` in charge of distributing the computation of multiple evaluations. Finally, a Bayesian search named `CBO` is created and executed to find the values of config which **MAXIMIZE** the return value of `run(config)`.

```python
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
    from deephyper.problem import HpProblem
    from deephyper.search.hps import CBO
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
```

Which outputs the following results where the best parameters are with `function == "cubic"`, 
`x == 9.99` and `b == 10`.

```verbatim
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
```

The code defines a function `run` that takes a RunningJob `job` as input and returns the maximized objective `y`. The `if` block at the end of the code defines a black-box optimization process using the `CBO` (Centralized Bayesian Optimization) algorithm from the `deephyper` library.

The optimization process is defined as follows:

1. A hyperparameter optimization problem is created using the `HpProblem` class from `deephyper`. In this case, the problem has a three variables. The `x` hyperparameter is a real variable in a range from -10.0 to 10.0. The `b` hyperparameter is a discrete variable in a range from 0 to 10. The `function` hyperparameter is a categorical variable with two possible values.
2. An evaluator is created using the `Evaluator.create` method. The evaluator will be used to evaluate the function `run` with different configurations of suggested hyperparameters in the optimization problem. The evaluator uses the `process` method to distribute the evaluations across multiple worker processes, in this case 2 worker processes.
3. A search object is created using the `CBO` class, the problem and evaluator defined earlier. The `CBO` algorithm is a derivative-free optimization method that uses a Bayesian optimization approach to explore the hyperparameter space.
4. The optimization process is executed by calling the `search.search` method, which performs the evaluations of the `run` function with different configurations of the hyperparameters until a maximum number of evaluations (100 in this case) is reached.
5. The results of the optimization process, including the optimal configuration of the hyperparameters and the corresponding objective value, are printed to the console.

## How do I learn more?

* Documentation: <https://deephyper.readthedocs.io>

* GitHub repository: <https://github.com/deephyper/deephyper>

* Blog: <https://deephyper.github.io>

## Contributions

Find all the list of contributors on the [DeepHyper Authors](https://deephyper.github.io/aboutus) page of the Documentation.

## Citing DeepHyper

Find all our publications on the [Research & Publication](https://deephyper.github.io/papers) page of the Documentation.

## How can I participate?

Questions, comments, feature requests, bug reports, etc. can be directed to:

* Issues on GitHub

Patches through pull requests are much appreciated on the software itself as well as documentation.
Optionally, please include in your first patch a credit for yourself in the list above.

The DeepHyper Team uses git-flow to organize the development: [Git-Flow cheatsheet](https://danielkummer.github.io/git-flow-cheatsheet/). For tests we are using: [Pytest](https://docs.pytest.org/en/latest/).

## Acknowledgements

* Scalable Data-Efficient Learning for Scientific Domains, U.S. Department of Energy 2018 Early Career Award funded by the Advanced Scientific Computing Research program within the DOE Office of Science (2018--Present)
* Argonne Leadership Computing Facility: This research used resources of the Argonne Leadership Computing Facility, which is a DOE Office of Science User Facility supported under Contract DE-AC02-06CH11357.
* SLIK-D: Scalable Machine Learning Infrastructures for Knowledge Discovery, Argonne Computing, Environment and Life Sciences (CELS) Laboratory Directed Research and Development (LDRD) Program (2016--2018)

## Copyright and license

Copyright © 2019, UChicago Argonne, LLC

DeepHyper is distributed under the terms of BSD License. See [LICENSE](https://github.com/deephyper/deephyper/blob/master/LICENSE.md)

Argonne Patent & Intellectual Property File Number: SF-19-007
