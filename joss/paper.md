---
title: 'DeepHyper: A Python Package for Massively Parallel Hyperparameter Optimization in Machine Learning'
tags:
  - Python
  - machine learning
  - hyperparameter optimization
  - multi-fidelity
  - neural architecture search
  - ensemble
  - high-performance computing
authors:
  - name: Romain Egele
    orcid: 0000-0002-8992-8192
    equal-contrib: true
    affiliation: "1"
    corresponding: true
  - name: Prasanna Balaprakash
    orcid: 0000-0002-0292-5715
    equal-contrib: true
    affiliation: "1"
  - name: Gavin M. Wiggins
    affiliation: 1
  - name: Brett Eiffert
    affiliation: 1
affiliations:
  - name: Oak Ridge National Laboratory, Oak Ridge, TN, United States
    index: 1
    ror: 01qz5mb56
date: 3 February 2025
bibliography: paper.bib
---

# Summary

Machine learning models are increasingly applied across scientific disciplines, yet their effectiveness often hinges on heuristic decisions—such as data transformations, training strategies, and model architectures—that are not learned by the models themselves. Automating the selection of these heuristics and analyzing their sensitivity is crucial for building robust and efficient learning workflows. `DeepHyper` addresses this challenge by democratizing hyperparameter optimization, providing accessible tools to streamline and enhance machine learning workflows from a laptop to the largest supercomputer in the world. Building on top of hyperparameter optimization, it unlocks new capabilities around ensembles of models for improved accuracy and uncertainty quantification. All of these organized around efficient parallel computing.

# Statement of need

`DeepHyper` is a Python package for parallel hyperparameter optimization or neural architecture search. The project started in 2018 [@balaprakash2018deephyper] with a focus on making Bayesian optimization more efficient on high-performance computing clusters. It provides access to a variety of asynchronous parallel black-box optimization algorithms via `deephyper.hpo`. The software offers a variety of parallel programming backends such as Asyncio, threading, processes, Ray, and MPI via `deehyper.evaluator`. The hyperparameter optimization can be single or multi-objective, composed of mixed variables, using explicit or hidden constraints, and benefit from early-discarding strategies via `deephyper.stopper`. Leveraging the results of hyperparameter optimization or neural architecture search it provides parallel ensemble algorithms via `deephyper.ensemble` that can help improve accuracy or quantify disentangled predictive uncertainty. A diagram of our software architecture is shown in Figure 1.

![DeepHyper Software Architecture](figures/deephyper-architecture.png){width=50%}

`DeepHyper` was designed to help research in the field of automated machine learning and also to be used out-of-the box in scientific projects where learning workflows are being developed.

# Related Work

Numerous software packages now exist for hyperparameter optimization (HPO), including:

- BoTorch (and Ax platform) [@botorch2020], ([doc.](https://botorch.org))
- HyperMapper [@hypermapper2019], ([doc.](https://github.com/luinardi/hypermapper))
- Hyperopt [@hyperopt2013;@hyperopt2015], ([doc.](http://hyperopt.github.io/hyperopt/))
- OpenBox [@openbox2024], ([doc.](https://open-box.readthedocs.io))
- Optuna [@optuna2019], ([doc.](https://optuna.readthedocs.io))
- SMAC3 [@smac32022], ([doc.](https://automl.github.io/SMAC3))

These tools differ in scope and design: some are research-oriented (e.g., SMAC), while others prioritize production-readiness and usability (e.g., Optuna). In this section, we focus our comparison on SMAC and Optuna, which are representative of these two directions.

**DeepHyper** is designed to maximize HPO efficiency across a wide range of parallelization scales, from sequential (single-core) runs to massively parallel evaluations on high-performance computing (HPC) systems with thousands of cores.

## Feature Comparison

While the feature matrix below provides a high-level overview, it necessarily simplifies some nuanced implementation differences. We use the $\checkmark$ symbol for available features, the $\approx$ symbol for incomplete features, and no symbol for missing features.

### Hyperparameter Optimization Capabilities

| Feature                | DeepHyper    | Optuna       | SMAC3        |
|-----------------------:|:------------:|:------------:|:------------:|
| Single-objective       | $\checkmark$ | $\checkmark$ | $\checkmark$ |
| Multi-objective        | $\checkmark$ | $\checkmark$ | $\checkmark$ |
| Early stopping         | $\checkmark$ | $\checkmark$ | $\checkmark$ |
| Fault tolerance        | $\checkmark$ | $\checkmark$ |              |
| Transfer learning      | $\checkmark$ | $\checkmark$ | $\approx$    |
| Ensemble construction  | $\checkmark$ |              |              |
| Visualization          | $\approx$    | $\checkmark$ |              |

Table: Overview of optimization features available in different packages.

**Single-objective Optimization**  
DeepHyper employs a surrogate model based on random forests to estimate $P(\text{Objective} \mid \text{Hyperparameters})$, similar to SMAC. However, DeepHyper's implementation is typically faster per query, especially when the number of evaluations exceeds 200. In contrast, Optuna uses the Tree-structured Parzen Estimator (TPE), which models $P(\text{Hyperparameters} \mid \text{Objective})$. TPE offers faster query times but can struggle with complex optimization landscapes and tends to be less effective in refining continuous hyperparameters.

**Multi-objective Optimization**  
DeepHyper uses scalarization-based approaches inspired by ParEGO (also used in SMAC), with randomized weights and a variety of scalarization functions. Optuna defaults to NSGA-II, a genetic algorithm that evolves solutions along estimated Pareto fronts. NSGA-II typically converges more slowly but catches up when the evaluation budget is sufficient.

**Early Discarding**  
DeepHyper supports several early stopping techniques, including constant thresholds, median stopping rules, successive halving, and learning curve extrapolation. Optuna also offers several early discarding methods among which some are different from DeepHyper's.

**Fault Tolerance**  
DeepHyper handles failed evaluations by assigning them the worst observed objective value, thereby preserving optimizer stability and avoiding cascading failures.

**Transfer Learning**  
DeepHyper supports warm-starting optimizations using data from prior runs. This is effective when either the objective function changes while the search space remains fixed (e.g., different datasets), or when the search space expands (e.g., broader neural architecture configurations).

**Ensemble Construction**  
DeepHyper enables model ensembling from the pool of evaluated configurations, helping to reduce variance and capture epistemic uncertainty in predictions.

**Visualization**  
Basic visualization tools are provided via `deephyper.analytics`. For more interactive exploration, we recommend [SandDance](https://microsoft.github.io/SandDance/), a Visual Studio Code plugin. Figure 2 illustrates a 3D visualization of a Random Forest optimization, with `min_samples_split`, `min_weight_fraction_leaf`, and test accuracy as the x, y, and z axes (and color), respectively. The two plots compare configurations using `splitter="best"` (left) and `splitter="random"` (right). Such visualizations help identify the sensitivity of the objective to different hyperparameters.

![Visualization of hyperparameter optimization results with SandDance](figures/sanddance_viz.png){width=80%}

### Parallelization Capabilities

|                            | DeepHyper    |  Optuna      | SMAC3     |
|---------------------------:|:------------:|:------------:|:---------:|
| Asynchronous optimization  | $\checkmark$ | $\checkmark$ | ?         |
| Centralized  optimization  | $\checkmark$ | $\checkmark$ |           |
| Decentralized optimization | $\checkmark$ | $\checkmark$ |           |
| Parallelization backends   | $\checkmark$ |              | $\approx$ |
| Memory backends            | $\checkmark$ | $\checkmark$ | $\approx$ |

Table: Overview of parallelization features available in different packages.

The main difference between DeepHyper, Optuna and SMAC related to parallelization is that DeepHyper provides out-of-the-box parallelization software while Optuna leaves it to the user and SMAC limits itself to centralized parallelism with Dask.

**Asynchronous optimization**: DeepHyper's allows to submit and gather hyperparameter configuration by batch and asynchronously (in a centralized or decentralized setting).

**Centralized optimization**: DeepHyper's allows to run centralized optimization, including $1$ master running the optimization and $N$ workers evaluating hyperparameter configurations.

**Decentralized optimization**: DeepHyper's allows to run decentralized optimization, including $N$ workers, each running centralized optimization.

**Parallelization backends**: DeepHyper's provides compatibility with several parallelization backends: AsyncIO functions, thread-based, process-based, Ray, and MPI. This allows to easily adapt to the context of the execution.

**Memory backends**: DeepHyper's provides compatibility with several shared-memory backends: local memory, server managed memory, MPI-based remote memory access, Ray.

## Black-box Optimization Benchmarks

The benchmark is run on 9 different continuous black-box benchmark functions: [Ackley](https://www.sfu.ca/~ssurjano/ackley.html) (5 dim.), [Branin](https://www.sfu.ca/~ssurjano/branin.html) (2 dim.), [Griewank](https://www.sfu.ca/~ssurjano/griewank.html) (5 dim.), [Hartmann](https://www.sfu.ca/~ssurjano/hart6.html) (6 dim.), [Levy](https://www.sfu.ca/~ssurjano/levy.html) (5 dim.), [Michalewicz](https://www.sfu.ca/~ssurjano/michal.html) (5 dim.), [Rosen](https://www.sfu.ca/~ssurjano/rosen.html) (5 dim.), [Schwefel](https://www.sfu.ca/~ssurjano/schwef.html) (5 dim.) and [Shekel](https://www.sfu.ca/~ssurjano/shekel.html) (5 dim.). 
Figures 3-7 present the benchmark results with the average regret and the standard error over 10 random repetitions for each method. The average regret is given by $y^* - \hat{y}^*$ where $y^*$ is the true optimal objective value and $\hat{y}^*$ is the current estimated best optimal objective value. All methods are run sequentially (no parallelization features enabled for consistent evaluation) with default arguments. For DeepHyper, the `CBO` search is used. For Optuna, the `TPE` sampler with default arguments is used. For SMAC, the default parameters using the OptunaHub SMAC sampler is used.

![Ackley 5D (left) and Branin 2D (right)](figures/benchmarks/ackley_5d_and_branin_2d.png)

![Griewank 5D (left) and Hartmann 6D (right)](figures/benchmarks/griewank_5d_and_hartmann_6d.png)

![Levy 5D (left) and Michalewicz 5D (right)](figures/benchmarks/levy_5d_and_michal_5d.png)

![Rosen 5D (left) and Schwefel 5D (right)](figures/benchmarks/rosen_5d_and_schwefel_5d.png)

![Shekel 5D](figures/benchmarks/shekel_5d.png){width=45%}


# Black-box Optimization Methods

The algorithms used for hyperparameter optimization are black-box optimization algorithms for mixed search spaces. A mixed search space, can be composed of real, discrete, or categorical (nominal or ordinal) values. The search space can also include constraints (e.g., explicit such as $x_0 < x_1$, or implicit such as "unexpected out-of-memory error"). The objective function $Y = f(x)$ can be stochastic where $Y$ is a random variable, $x$ is a vector of input hyperparameters, and $f$ is the objective function. The `DeepHyper`, main hyperparameter optimization algorithm, is based on Bayesian optimization.

The Bayesian optimization of `DeepHyper` relies on Extremely Randomized Forest as surrogate model to estimate $E_Y[Y|X=x]$ by default. Extremely Randomized Forests [@geurts2006extremely] are a type of Randomized Forest algorithms in which the split decision involves a random process for each newly created node of a tree. It provides smoother epistemic uncertainty estimates with an increasing number of trees (left side in Figure 8) compared to usual Random Forests that use a deterministic "best" split decision (right side in Figure 8).

![Uncertainty of Randomized Forests, on the left-side with random split, and on the right-side with best split.](figures/random_forest.png)

Then, a custom acquisition function $\text{UCBd}(x) = \mu(x) + \kappa \cdot \sigma_\text{ep}(x)$, combines the mean predicition $\mu(x)$ with the epistemic uncertainty $\sigma_\text{ep}(x)$ of this surrogate (purple area in Figure 8) for improved efficiency. 
We also apply a periodic exponential decay (Figure 9), impacting the exploration-exploitation parameter $\kappa$ to escape local solutions [@egele2023asynchronous].

![Periodic Exponential Decay for Bayesian Optimization](figures/example-exp-decay.jpg){width=49%}

Batch parallel genetic algorithms are provided to resolve efficiently the sub-problem of optimizing the acquisition function and it is also more accurate than Monte-Carlo approaches to find the optimum of the acquisition function. A cheap and efficient multi-point acquisition strategy `qUCBd` is provided for better parallel scalability [@egele2023asynchronous].

The multi-objective optimization is enabled by scalarization functions and objective rescaling [@egele2023dmobo].

The early-discarding strategies include asynchronous successive halving and a robust learning curve extrapolation [@egele2024unreasonable].

The ensemble strategies is modular to allow: exploring models tested during hyperparameter optimization, classification and regression problems to be treated, disentangled uncertainty quantification [@egele2022autodeuq]. It can leverage the same parallelization features as the optimization.

Finally, a transfer-learning strategy for hyperparameter optimization [@dorier2022transferlearning] is available. This strategy used to be based on variational auto-encoders for tabular data. It is now based on the Gaussian mixture-model for tabular data.

# Software Development

The `DeepHyper` package adheres to best practices in the Python community by following the [PEP 8 style guide](https://pep8.org) for Python naming and formatting conventions. The layout and configuration of the package follows suggestions made by the [Python Packaging User Guide](https://packaging.python.org) which is maintained by the Python Packaging Authority. A `pyproject.toml` file provides all configuration settings and meta data for developing and publishing the `DeepHyper` package. The [ruff tool](https://astral.sh/ruff) is used to enforce style and format conventions during code development and during continuous integration checks via GitHub Actions. Unit tests are also conducted in the CI workflow with the [pytest](https://docs.pytest.org) framework. Conda environments and standard Python virtual environments are supported by the project to ensure consistent development environments. Documentation for the package is generated with the [Sphinx application](https://www.sphinx-doc.org/en/master/). See the Developer's Guide section in the DeepHyper [documentation](https://deephyper.readthedocs.io) for contributing guidelines and more software development information.

# Acknowledgements

We acknowledge contributions from Misha Salim, Romit Maulik, Venkat Vishwanath, Stefan Wild, Joceran Gouneau, Dipendra Jha, Kyle Felker, Matthieu Dorier, Felix Perez, Bethany Lush, Gavin M. Wiggins, Tyler H. Chang, Yixuan Sun, Shengli Jiang, rmjcs2020, Albert Lam, Taylor Childers, Z223I, Zachariah Carmichael, Hongyuan Liu, Sam Foreman, Akalanka Galappaththi, Brett Eiffert, Sun Haozhe, Sandeep Madireddy, Adrian Perez Dieguez, and Nesar Ramachandra. We also would like to thank Prof. Isabelle Guyon for her guidance on the machine learning methodologies developed in this software.

# References