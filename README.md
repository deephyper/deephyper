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

DeepHyper is a powerful Python package for automating machine learning tasks, particularly focused on optimizing hyperparameters, searching for optimal neural architectures, and quantifying uncertainty through the use of deep ensembles. With DeepHyper, users can easily perform these tasks on a single machine or distributed across multiple machines, making it ideal for use in a variety of environments. Whether you're a beginner looking to optimize your machine learning models or an experienced data scientist looking to streamline your workflow, DeepHyper has something to offer. So why wait? Start using DeepHyper today and take your machine-learning skills to the next level!

## Install Instructions

Installation with `pip`:

```console
pip install deephyper
```

More details about the installation process can be found at [DeepHyper Installations](https://deephyper.readthedocs.io/en/latest/install/index.html).

## Quickstart

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deephyper/tutorials/blob/main/tutorials/colab/DeepHyper_101.ipynb)

The black-box function named `run` is defined by taking an input job named `job` which contains the different variables to optimize `job.parameters`. Then the run-function is bound to an `Evaluator` in charge of distributing the computation of multiple evaluations. Finally, a Bayesian search named `CBO` is created and executed to find the values of config which **MAXIMIZE** the return value of `run(job)`.

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


# Necessary IF statement otherwise it will enter in a infinite recursion
# when loading the 'run' function from a child processes
if __name__ == "__main__":
    from deephyper.hpo import CBO, HpProblem
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
     p:b p:function       p:x    objective  job_id job_status  m:timestamp_submit  m:timestamp_gather
0      7      cubic -1.103350     5.656803       0       DONE            0.018402            1.548068
1      3      cubic  8.374450   590.312101       1       DONE            0.018485            1.548254
2      9     linear  8.787395    17.787395       3       DONE            1.564276            1.565336
3      6      cubic  4.680560   108.540056       2       DONE            1.564209            1.565440
4      2      cubic  4.012429    66.598442       5       DONE            1.575218            1.576076
..   ...        ...       ...          ...     ...        ...                 ...                 ...
96    10      cubic  9.986875  1006.067558      96       DONE            9.895560            9.995656
97     9      cubic  9.999787  1008.936159      97       DONE            9.995220           10.095534
98     9      cubic  9.997146  1008.143990      98       DONE           10.095102           10.195398
99     7      cubic  9.999389  1006.816600      99       DONE           10.194956           10.297452
100    9      cubic  9.997912  1008.373594     100       DONE           10.296981           10.412184
```

The code defines a function `run` that takes a RunningJob `job` as input and returns the maximized objective `y`. The `if` block at the end of the code defines a black-box optimization process using the `CBO` (Centralized Bayesian Optimization) algorithm from the `deephyper` library.

The optimization process is defined as follows:

1. A hyperparameter optimization problem is created using the `HpProblem` class from `deephyper`. In this case, the problem has three variables. The `x` hyperparameter is a real variable in a range from -10.0 to 10.0. The `b` hyperparameter is a discrete variable in a range from 0 to 10. The `function` hyperparameter is a categorical variable with two possible values.
2. An evaluator is created using the `Evaluator.create` method. The evaluator will be used to evaluate the function `run` with different configurations of suggested hyperparameters in the optimization problem. The evaluator uses the `process` method to distribute the evaluations across multiple worker processes, in this case, 2 worker processes.
3. A search object is created using the `CBO` class, the problem and evaluator defined earlier. The `CBO` algorithm is a derivative-free optimization method that uses a Bayesian optimization approach to explore the hyperparameter space.
4. The optimization process is executed by calling the `search.search` method, which performs the evaluations of the `run` function with different configurations of the hyperparameters until a maximum number of evaluations (100 in this case) is reached.
5. The results of the optimization process, including the optimal configuration of the hyperparameters and the corresponding objective value, are printed to the console.

## How do I learn more?

* Documentation: <https://deephyper.readthedocs.io>

* GitHub repository: <https://github.com/deephyper/deephyper>

* Blog: <https://deephyper.github.io>

## Contributions

Find the list of contributors on the [DeepHyper Authors](https://deephyper.github.io/aboutus) page of the Documentation.

## Citing DeepHyper

If you wish to cite the Software, please use the following:

```
@misc{deephyper_software,
    title = {"DeepHyper: A Python Package for Scalable Neural Architecture and Hyperparameter Search"},
    author = {Balaprakash, Prasanna and Egele, Romain and Salim, Misha and Maulik, Romit and Vishwanath, Venkat and Wild, Stefan and others},
    organization = {DeepHyper Team},
    year = 2018,
    url = {https://github.com/deephyper/deephyper}
} 
```

Find all our publications on the [Research & Publication](https://deephyper.github.io/papers) page of the Documentation.

## How can I participate?

Questions, comments, feature requests, bug reports, etc. can be directed to:

* Issues on GitHub

Patches through pull requests are much appreciated on the software itself as well as documentation.
Optionally, please include in your first patch a credit for yourself in the list above.

The DeepHyper Team uses git-flow to organize the development: [Git-Flow cheatsheet](https://danielkummer.github.io/git-flow-cheatsheet/). For tests we are using: [Pytest](https://docs.pytest.org/en/latest/).

## Acknowledgments

* Scalable Data-Efficient Learning for Scientific Domains, U.S. Department of Energy 2018 Early Career Award funded by the Advanced Scientific Computing Research program within the DOE Office of Science (2018--Present)
* Argonne Leadership Computing Facility: This research used resources of the Argonne Leadership Computing Facility, which is a DOE Office of Science User Facility supported under Contract DE-AC02-06CH11357.
* SLIK-D: Scalable Machine Learning Infrastructures for Knowledge Discovery, Argonne Computing, Environment and Life Sciences (CELS) Laboratory Directed Research and Development (LDRD) Program (2016--2018)

## Copyright and license

Copyright © 2019, UChicago Argonne, LLC

DeepHyper is distributed under the terms of BSD License. See [LICENSE](https://github.com/deephyper/deephyper/blob/master/LICENSE.md)

Argonne Patent & Intellectual Property File Number: SF-19-007
