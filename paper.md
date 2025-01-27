---
title: 'DeepHyper: A Python Package for Parallel Hyperparameter Optimization'
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
    affiliation: "1" # (Multiple affiliations must be quoted)
  - name: Prasanna Balaprakash
    orcid: 0000-0002-0292-5715
    equal-contrib: true
    affiliation: "1"
  - name: Author Without ORCID
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 2
  - name: Author with no affiliation
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 3
  - given-names: Ludwig
    dropping-particle: van
    surname: Beethoven
    affiliation: 3
affiliations:
 - name: Oak Ridge National Laboratory, Tenesse, United States
   index: 1
   ror: 01qz5mb56
 - name: Institution Name, Country
   index: 2
 - name: Independent Researcher, Country
   index: 3
date: 14 January 2025
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

Machine learning models are increasingly applied across scientific disciplines, yet their effectiveness often hinges on heuristic decisions—such as data transformations, training strategies, and model architectures—that are not learned by the models themselves. Automating the selection of these heuristics and analyzing their sensitivity is crucial for building robust and efficient learning workflows. `DeepHyper` addresses this challenge by democratizing hyperparameter optimization, providing accessible tools to streamline and enhance machine learning workflows from a laptop to the largest supercomputer in the world. Building on top of hyperparameter optimization it unlock new capabilities around ensembles of models for improved accuray and uncertainty quantification. All of these organized around efficient parallel computing.

![DeepHyper Logo](figures/logo-deephyper-nobg.png)

# Statement of need

`DeepHyper` is a Python package for parallel hyperparameter optimization or neural architecture search. It provides
access to a variety of asynchronous parallel black-box optimization algorithms `deephyper.search`. The 
software offers a variety of parallel programming backends such as Asyncio, threading, processes, Ray, and MPI `deehyper.evaluator`. The hyperparameter optimization can be single or multi-objective, composed of mixed variables,
using explicit or hidden constraints, and benefit from early-discarding strategies `deephyper.stopper`. Leveraging the results of hyperparameter optimization or neural architecture search it provides parallel ensemble algorithms `deephyper.ensemble` that can help improve accuracy or quantify disentangled predictive uncertainty.

![DeepHyper Software Architecture](figures/deephyper-architecture.png)

`DeepHyper` was designed to help research in the field of automated machine learning and also to be used out-of-the box
in scientific projects where learning workflows are being developed.

# Mathematics

`DeepHyper` main hyperparameter optimization algorithm is based on Bayesian optimization and it can manage stochastic objective functions.

The Bayesian optimization of the `DeepHyer` relies on Extremely Randomized Forest as default surrogate model. 

![Uncertainty for Random Forest (Best Split)](figures/random_forest_best_split.png)
![Uncertainty for Extremely Randomized Forest (Random Split)](figures/random_forest_random_split.png)

A custom acquisition function `UCBd`, focuses on the epistemic uncertainty of this surrogate for improved efficiency. A custom periodic exponential decay is available to escape local solutions (Egele et al., 2023).

![Periodic Exponential Decay for Bayesian Optimization](figures/example-exp-decay.jpg)

Batch parallel genetic algorithms are provided to optimize the acquisition efficiently and more accurately than Monte-Carlo approches. An efficient multi-point acquisition strategy `qUCBd` is provided for better parallel scalability (Egele et al., 2023).

The multi-objective optimization is enabled by scalarization functions and objective rescaling (Egele and Chang et al., 2023).

The early-discarding strategies include asynchronous successive halving and a robust learning curve extrapolation (Egele et al., 2024).

The ensemble strategies is modular to allow: exploring models tested during hyperparameter optimization, classification and regression problems to be treated, disentangled uncertainty quantification (Egele et al., 2022).

# Acknowledgements

We acknowledge contributions from Misha Salim, Romit Maulik, Venkat Vishwanath, Stefan Wild, Joceran Gouneau, Dipendra Jha, Kyle Felker, Matthieu Dorier, Felix Perez, Bethany Lush, Gavin M. Wiggins, Tyler H. Chang, Yixuan Sun, Shengli Jiang, @rmjcs2020, Albert Lam, Taylor Childers, @Z223I, Zachariah Carmichael, Hongyuan Liu, Sam Foreman, Akalanka Galappaththi,Brett Eiffert, Sun Haozhe, Sandeep Madireddy, Adrian Perez Dieguez, and Nesar Ramachandra. 
We also would like to thank Prof. Isabelle Guyon for her guidance on the machine learning methodologies developed in this software.

# References