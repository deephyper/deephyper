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
DeepHyper is a software package that uses learning, optimization, and parallel computing to automate the design and development of machine learning (ML) models for scientific and engineering applications. DeepHyper reduces the barrier to entry for using AI/ML model development by reducing manually intensive trial-and-error efforts for developing predictive models. The package performs four key functions:

1. pipeline optimization for ML (DeepHyper/POPT)
2. neural architecture search (DeepHyper/NAS)
3. hyperparameter search (DeepHyper/HPS)
4. ensemble uncertainty quantification (DeepHyper/AutoDEUQ)

### Pipeline optimization for ML (DeepHyper/POPT)

Predictive modeling with classical ML methods typically requires a pipeline of methods such as data preprocessing, data balancing, data splitting, variable importance analysis, variable selection, classification/regression algorithm selection, and cross-validation methods. Because of the myriad choices available for each method, employing an effective ML pipeline is beyond most scientists and engineers; therefore, they tend to resort to rules of thumb, often resulting in non-robust models. DeepHyper provides an interface to model the search space of the pipeline. It uses an intelligent search algorithm that samples a small number of pipeline configurations and progressively fits a surrogate model over the configuration-performance space until it exhausts the user-defined maximum number of evaluations. The asynchronous aspect allows the search to avoid waiting for all the evaluation results before proceeding to the next iteration. When an evaluation is finished, the data are used to retrain the surrogate model, which is then used to bias the search toward the promising configurations. The framework uses a master/worker computational paradigm, where one master node fits the surrogate model and generates promising pipeline configurations and worker nodes perform the computationally expensive evaluations and return the outputs to the master node.

### Hyperparameter search (DeepHyper/HPS)

ML methods used for predictive modeling typically require user-specified values for hyperparameters, which include the number of hidden layers and units per layer, sparsity/overfitting regularization parameters, batch size, learning rate, type of initialization, optimizer, and activation function specification. Traditionally, to find performance-optimizing hyperparameter settings, researchers have used a trial-and-error process or a brute-force grid/random search. Such approaches lead to far-from-optimal performance, however, or are impractical for addressing large numbers of hyperparameters. DeepHyper provides a set of scalable hyperparameter search methods for automatically searching for high-performing hyperparameters for a given DNN architecture. DeepHyper uses an asynchronous model-based search that relies on fitting a dynamically updated surrogate model that tries to learn the relationship between the hyperparameter configurations (input) and their validation errors (output). The surrogate model is cheap to evaluate and can be used to prune the search space and identify promising regions, where the model then is iteratively refined by obtaining new outputs at inputs that are predicted by the model to be high-performing.

### Neural architecture search (DeepHyper/NAS)

Scientific data sets are diverse and often require data-set-specific DNN models. Nevertheless, designing high-performing DNN architecture for a given data set is an expert-driven, time-consuming, trial-and-error manual task. To that end, DeepHyper provides a NAS for automatically identifying high-performing DNN architectures for a given set of training data. DeepHyper adopts an evolutionary algorithm that generates a population of DNN architectures, trains them concurrently by using multiple nodes, and improves the population by performing mutations on the existing architectures within a population. To reduce the training time of each architecture evaluation, DeepHyper adopts a distributed data-parallel training technique, splitting the training data and distributing the shards to multiple processing units. Multiple models with the same architecture are trained on different data shards, and the gradients from each model are averaged and used to update the weights of all the models. To maintain accuracy and reduce training time, DeepHyper combines aging evolution and an asynchronous Bayesian optimization method for tuning the hyperparameters of the data-parallel training simultaneously.

### Ensemble uncertainty quantification (DeepHyper/AutoDEUQ)

Uncertainty quantification in DNN predictions is of paramount importance for confident scientific utilization of DL. Prediction with uncertainty quantification is vital when DNN learning deployments are performed on unseen datasets that may not be from the distribution of the training data. In such cases, confidence estimates are essential for deciding when to discard predictions from neural networks, because of their proclivity to extrapolation. More importantly, DeepHyper/AutoDEUQ sheds light on learning systems that are frequently dismissed as black-box within the scientific community, and it paves the way for greater model trustworthiness. To that end, DeepHyper/AutoDEUQ employs a scalable deep-ensemble approach for uncertainty quantification. The approach involves constructing several models with varying architectures, independently, on the training dataset. Parallel independent runs of DeepHyper/NAS are used, wherein each NAS run starts with different initialization and randomization of datasets. The best model candidates suggested by the parallel search are then utilized in an ensemble setting for quantifying the model uncertainty. Furthermore, each generated model can be configured to use various data likelihood options. The quantiles of these function values can be used to compute calibrated prediction intervals to capture data uncertainty.

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
```

Which outputs the following where the best ``x`` found is clearly around ``0``.

```verbatim
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
```

## How do I learn more?

* Documentation: <https://deephyper.readthedocs.io>

* GitHub repository: <https://github.com/deephyper/deephyper>

## Contributions

Find all the list of contributors on the [DeepHyper Authors](https://deephyper.readthedocs.io/en/latest/research.html) page of the Documentation.

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
