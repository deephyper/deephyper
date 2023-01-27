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

From PyPI:

```bash
pip install deephyper
```

From Github:

```bash
git clone https://github.com/deephyper/deephyper.git
pip install -e deephyper/
```

If you want to install deephyper with test and documentation packages:

From PyPI:

```bash
pip install 'deephyper[dev]'
```

From Github:

```bash
git clone https://github.com/deephyper/deephyper.git
pip install -e 'deephyper/[dev]'
```

## Quickstart

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deephyper/tutorials/blob/main/tutorials/colab/DeepHyper_101.ipynb)

The black-box function named `run` is defined by taking an input dictionnary named `config` which contains the different variables to optimize. Then the run-function is binded to an `Evaluator` in charge of distributing the computation of multiple evaluations. Finally, a Bayesian search named `CBO` is created and executed to find the values of config which maximize the return value of `run(config)`.

```python
def run(config: dict):
    return -config["x"]**2


# Necessary IF statement otherwise it will enter in a infinite loop
# when loading the 'run' function from a new process
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
        method="process",
        method_kwargs={
            "num_workers": 2,
        },
    )

    # define your search and execute it
    search = CBO(problem, evaluator)

    results = search.search(max_evals=100)
    print(results)
```

Which outputs the following where the best ``x`` found is clearly around ``0``.

```verbatim
         p:x  job_id     objective  timestamp_submit  timestamp_gather
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
```

We also asked [ChatGPT](https://chat.openai.com) to explain this example and here is the reply:

"The code defines a function `run` that takes a dictionary `config` as input and returns the negative of `x` squared. The `if` block at the end of the code defines a black-box optimization process using the `CBO` (Centralized Bayesian Optimization) algorithm from the `deephyper` library.

The optimization process is defined as follows:

1. A hyperparameter optimization problem is created using the `HpProblem` class from `deephyper`. In this case, the problem has a single continuous hyperparameter `x` that ranges from -10.0 to 10.0.
2. An evaluator is created using the `Evaluator.create` method. The evaluator will be used to evaluate the function `run` with different configurations of the hyperparameters in the optimization problem. The evaluator uses the `process` method to distribute the evaluations across multiple worker processes, in this case 2 worker processes.
3. A search object is created using the `CBO` class and the problem and evaluator defined earlier. The `CBO` algorithm is a derivative-free optimization method that uses a Bayesian optimization approach to explore the hyperparameter space.
4. The optimization process is executed by calling the `search.search` method, which performs the evaluations of the `run` function with different configurations of the hyperparameters until a maximum number of evaluations (100 in this case) is reached.
5. The results of the optimization process, including the optimal configuration of the hyperparameters and the corresponding objective value, are printed to the console."

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
