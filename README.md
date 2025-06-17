<p align="center">
<img src="docs/_static/logo/medium.png">
</p>

[![DOI](https://joss.theoj.org/papers/10.21105/joss.07975/status.svg)](https://doi.org/10.21105/joss.07975)
![GitHub tag (latest by date)](https://img.shields.io/github/tag-date/deephyper/deephyper.svg?label=version)
[![Documentation Status](https://readthedocs.org/projects/deephyper/badge/?version=latest)](https://deephyper.readthedocs.io/en/latest/?badge=latest)
![License](https://img.shields.io/github/license/deephyper/deephyper)
![PyPI - Downloads](https://img.shields.io/pypi/dm/deephyper.svg?label=Pypi%20downloads)

# DeepHyper: A Python Package for Massively Parallel Hyperparameter Optimization in Machine Learning

DeepHyper is first and foremost a hyperparameter optimization (HPO) library. By leveraging this core HPO functionnality, DeepHyper also provides neural architecture search, multi-fidelity and ensemble capabilities. With DeepHyper, users can easily perform these tasks on a single machine or distributed across multiple machines, making it ideal for use in a variety of environments. Whether you’re a beginner looking to optimize your machine learning models or an experienced data scientist looking to streamline your workflow, DeepHyper has something to offer. So why wait? Start using DeepHyper today and take your machine learning skills to the next level!

## Installation

Installation with `pip`:

```console
pip install deephyper
```

More details about the installation process can be found in our [Installation](https://deephyper.readthedocs.io/en/stable/install/) documentation.

## Quickstart

The black-box function named `run` is defined by taking an input job named `job` which contains the different variables to optimize `job.parameters`. Then the run-function is bound to an `Evaluator` in charge of distributing the computation of multiple evaluations. Finally, a Bayesian search named `CBO` is created and executed to find the values of config which **MAXIMIZE** the return value of `run(job)`.

```python
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
    print("function:", row["sol.p:function"])
    print("x:", row["sol.p:x"])
    print("b:", row["sol.p:b"])
    print("y:", row["sol.objective"])
```

Which outputs the following results where the best parameters are with `function == "cubic"`, 
`x == 9.99` and `b == 10`.

```verbatim
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
```

More details about this example can be found in our [Quick Start](https://deephyper.readthedocs.io/en/stable/#quick-start) documentation.

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
