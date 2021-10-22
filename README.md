<p align="center">
<img src="docs/_static/logo/medium.png">
</p>

[![DOI](https://zenodo.org/badge/156403341.svg)](https://zenodo.org/badge/latestdoi/156403341)
![GitHub tag (latest by date)](https://img.shields.io/github/tag-date/deephyper/deephyper.svg?label=version)
<!-- [![Build Status](https://travis-ci.com/deephyper/deephyper.svg?branch=develop)](https://travis-ci.com/deephyper/deephyper) -->
[![Documentation Status](https://readthedocs.org/projects/deephyper/badge/?version=latest)](https://deephyper.readthedocs.io/en/latest/?badge=latest)
![PyPI - License](https://img.shields.io/pypi/l/deephyper.svg)
![PyPI - Downloads](https://img.shields.io/pypi/dm/deephyper.svg?label=Pypi%20downloads)

# What is DeepHyper?

DeepHyper is an automated machine learning ([AutoML](https://en.wikipedia.org/wiki/Automated_machine_learning)) package for deep neural networks. It comprises two components: 1) Neural architecture search is an approach for automatically searching for high-performing the deep neural network
search_space. 2) Hyperparameter search is an approach for automatically searching for high-performing hyperparameters for a given deep neural network. DeepHyper provides an infrastructure that targets experimental research in neural architecture
and hyperparameter search methods, scalability, and portability across HPC systems. It comprises three modules:
benchmarks, a collection of extensible and diverse benchmark problems;
search, a set of search algorithms for neural architecture search and hyperparameter search;
and evaluators, a common interface for evaluating hyperparameter configurations
on HPC platforms.

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

The black-box function named `run` is defined by taking an input dictionnary named `config` which contains the different variables to optimize. Then the run-function is binded to an `Evaluator` in charge of distributing the computation of multiple evaluations. Finally, a Bayesian search named `AMBS` is created and executed to find the values of config which maximize the return value of `run(config)`.

```python
def run(config: dict):
    return -config["x"]**2


# Necessary IF statement otherwise it will enter in a infinite loop
# when loading the 'run' function from a subprocess
if __name__ == "__main__":
    from deephyper.problem import HpProblem
    from deephyper.search.hps import AMBS
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
    search = AMBS(problem, evaluator)

    results = search.search(max_evals=100)
    print(results)
```

Which outputs the following where the best ``x`` found is clearly around ``0``.

```verbatim
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
```

## How do I learn more?

* Documentation: <https://deephyper.readthedocs.io>

* GitHub repository: <https://github.com/deephyper/deephyper>

## Who is responsible?

Currently, the core DeepHyper team is at Argonne National Laboratory:

* Prasanna Balaprakash <pbalapra@anl.gov>, Lead and founder
* Romain Egele <romainegele@gmail.com>, Co-Lead
* Misha Salim <msalim@anl.gov>
* Romit Maulik <rmaulik@anl.gov>
* Venkat Vishwanath <venkat@anl.gov>
* Stefan Wild <wild@anl.gov>

Modules, patches (code, documentation, etc.) contributed by:

* Elise Jennings <ejennings@anl.gov>
* Dipendra Kumar Jha <dipendrajha2018@u.northwestern.edu>
* Felix Perez <felix.perez@utdallas.edu>

## Citing DeepHyper

Find all our publications on the [Research & Publication](https://deephyper.readthedocs.io/en/latest/research.html) page of the Documentation.

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
