<p align="center">
<img src="docs/_static/logo/medium.png">
</p>

[![DOI](https://joss.theoj.org/papers/10.21105/joss.07975/status.svg)](https://doi.org/10.21105/joss.07975)
![GitHub tag (latest by date)](https://img.shields.io/github/tag-date/deephyper/deephyper.svg?label=version)
[![Documentation Status](https://readthedocs.org/projects/deephyper/badge/?version=latest)](https://deephyper.readthedocs.io/en/latest/?badge=latest)
![License](https://img.shields.io/github/license/deephyper/deephyper)
![PyPI - Downloads](https://img.shields.io/pypi/dm/deephyper.svg?label=Pypi%20downloads)

## What is DeepHyper?

DeepHyper is a powerful Python package for automating machine learning tasks, particularly focused on optimizing hyperparameters, searching for optimal neural architectures, and quantifying uncertainty through the use of deep ensembles. With DeepHyper, users can easily perform these tasks on a single machine or distributed across multiple machines, making it ideal for use in a variety of environments. Whether you're a beginner looking to optimize your machine learning models or an experienced data scientist looking to streamline your workflow, DeepHyper has something to offer. So why wait? Start using DeepHyper today and take your machine-learning skills to the next level!

## Installation

Installation with `pip`:

```console
pip install deephyper
```

More details about the installation process can be found in our [Installation](https://deephyper.readthedocs.io/en/stable/install/) documentation.

## Quickstart

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

Check out our online documentation with API reference and examples: <https://deephyper.readthedocs.io>

## Citing DeepHyper

To cite this repository:

```
@article{Egele2025,
    doi = {10.21105/joss.07975},
    url = {https://doi.org/10.21105/joss.07975},
    year = {2025},
    publisher = {The Open Journal},
    volume = {10},
    number = {109},
    pages = {7975},
    author = {Romain Egele and Prasanna Balaprakash and Gavin M. Wiggins and Brett Eiffert},
    title = {DeepHyper: A Python Package for Massively Parallel Hyperparameter Optimization in Machine Learning},
    journal = {Journal of Open Source Software}
}
```

## How can I participate?

Questions, comments, feature requests, bug reports, etc. can be directed to Github Issues.

Patches through pull requests are much appreciated on the software itself as well as documentation.

More documentation about how to contribute is available on [deephyper.readthedocs.io/en/latest/developer_guides/contributing.html](https://deephyper.readthedocs.io/en/latest/developer_guides/contributing.html).

## Acknowledgments

* Scalable Data-Efficient Learning for Scientific Domains, U.S. Department of Energy 2018 Early Career Award funded by the Advanced Scientific Computing Research program within the DOE Office of Science (2018--Present)
* Argonne Leadership Computing Facility: This research used resources of the Argonne Leadership Computing Facility, which is a DOE Office of Science User Facility supported under Contract DE-AC02-06CH11357.
* SLIK-D: Scalable Machine Learning Infrastructures for Knowledge Discovery, Argonne Computing, Environment and Life Sciences (CELS) Laboratory Directed Research and Development (LDRD) Program (2016--2018)

## Copyright and license

Copyright © 2019, UChicago Argonne, LLC

DeepHyper is distributed under the terms of BSD License. See [LICENSE](https://github.com/deephyper/deephyper/blob/master/LICENSE)

Argonne Patent & Intellectual Property File Number: SF-19-007
