<p align="center">
<img src="docs/_static/logo/medium.png">
</p>

![GitHub tag (latest by date)](https://img.shields.io/github/tag-date/deephyper/deephyper.svg?label=version)
[![Build Status](https://travis-ci.com/deephyper/deephyper.svg?branch=develop)](https://travis-ci.com/deephyper/deephyper)
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

# Documentation

Deephyper documentation is on [ReadTheDocs](https://deephyper.readthedocs.io)

# Install instructions

From pip:
```
pip install deephyper
```

From github:
```
git clone https://github.com/deephyper/deephyper.git
cd deephyper/
pip install -e .
```

if you want to install deephyper with test and documentation packages:
```
# From Pypi
pip install 'deephyper[tests,docs]'

# From github
git clone https://github.com/deephyper/deephyper.git
cd deephyper/
pip install -e '.[tests,docs]'
```

# Directory search_space

```
benchmark/
    a set of problems for hyperparameter or neural architecture search which the user can use to compare our different search algorithms or as examples to build their own problems.
evaluator/
    a set of objects which help to run search on different systems and for different cases such as quick and light experiments or long and heavy runs.
search/
    a set of algorithms for hyperparameter and neural architecture search. You will also find a modular way to define new search algorithms and specific sub modules for hyperparameter or neural architecture search.
hps/
        hyperparameter search applications
nas/
        neural architecture search applications
```


# How do I learn more?

* Documentation: https://deephyper.readthedocs.io

* GitHub repository: https://github.com/deephyper/deephyper

# Quickstart

## Hyperparameter Search (HPS)

```
deephyper hps ambs --evaluator ray --problem deephyper.benchmark.hps.polynome2.Problem --run deephyper.benchmark.hps.polynome2.run --n-jobs 1
```

## Neural Architecture Search (NAS)

```
deephyper nas ambs --evaluator ray --problem deephyper.benchmark.nas.polynome2Reg.Problem --n-jobs 1
```

# Who is responsible?

Currently, the core DeepHyper team is at Argonne National Laboratory:

* Prasanna Balaprakash <pbalapra@anl.gov>, Lead and founder
* Romain Egele <regele@anl.gov>
* Misha Salim <msalim@anl.gov>
* Romit Maulik <rmaulik@anl.gov>
* Venkat Vishwanath <venkat@anl.gov>
* Stefan Wild <wild@anl.gov>

Modules, patches (code, documentation, etc.) contributed by:

* Elise Jennings <ejennings@anl.gov>
* Dipendra Kumar Jha <dipendrajha2018@u.northwestern.edu>


# Citing DeepHyper

If you are referencing DeepHyper in a publication, please cite the following papers:

 * P. Balaprakash, M. Salim, T. Uram, V. Vishwanath, and S. M. Wild. **DeepHyper: Asynchronous Hyperparameter Search for Deep Neural Networks**.
    In 25th IEEE International Conference on High Performance Computing, Data, and Analytics. IEEE, 2018.
 
 * P. Balaprakash, R. Egele, M. Salim, S. Wild, V. Vishwanath, F. Xia, T. Brettin, and R. Stevens. **Scalable reinforcement-learning-based neural architecture search for cancer deep learning research**.  In SC ’19:  IEEE/ACM International Conference on High Performance Computing, Network-ing, Storage and Analysis, 2019.

 <!-- * R. Egele, D. Jha, P. Balaprakash, M. Salim, V. Vishwanath, and S. M. Wild. **Scalable Reinforcement-Learning-Based Neural Architecture Search for Scientific and Engineering Applications**. In 34th International Conference on High Performance Computing, 2019. -->

# How can I participate?

Questions, comments, feature requests, bug reports, etc. can be directed to:

* Our mailing list: *deephyper@groups.io* or https://groups.io/g/deephyper

* Issues on GitHub

Patches are much appreciated on the software itself as well as documentation.
Optionally, please include in your first patch a credit for yourself in the
list above.

The DeepHyper Team uses git-flow to organize the development: [Git-Flow cheatsheet](https://danielkummer.github.io/git-flow-cheatsheet/). For tests we are using: [Pytest](https://docs.pytest.org/en/latest/).

# Acknowledgements

* Scalable Data-Efficient Learning for Scientific Domains, U.S. Department of Energy 2018 Early Career Award funded by the Advanced Scientific Computing Research program within the DOE Office of Science (2018--Present)
* Argonne Leadership Computing Facility: This research used resources of the Argonne Leadership Computing Facility, which is a DOE Office of Science User Facility supported under Contract DE-AC02-06CH11357.
* SLIK-D: Scalable Machine Learning Infrastructures for Knowledge Discovery, Argonne Computing, Environment and Life Sciences (CELS) Laboratory Directed Research and Development (LDRD) Program (2016--2018)

# Copyright and license

Copyright © 2019, UChicago Argonne, LLC

DeepHyper is distributed under the terms of BSD License. See [LICENSE](https://github.com/deephyper/deephyper/blob/master/LICENSE.md)

Argonne Patent & Intellectual Property File Number: SF-19-007

