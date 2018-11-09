.. deephyper documentation master file, created by
   sphinx-quickstart on Thu Sep 27 13:32:19 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Deephyper - high level deep learning framework
==============================================

Deephyper is a Python package which provides a common interface for the implementation and study of scalable hyperparameter and neural architecture search methods. In this package we provide to the user different modules:
    - ``benchmarks``: a set of problems for hyperparameter or neural architecture search which the user can use to compare our different search algorithms or as examples to build their own problems.
    - ``evaluators``: a set of objects which help to run searches on different systems and for different cases such as quick and light experiments or long and heavy runs.
    - ``search``: a set of algorithms for hyperparameter and neural architecture search. You will also find a modular way to define new search algorithms and specific sub modules for hyperparameter or neural architecture search.

.. toctree::
   :maxdepth: 2
   :caption: Quick Start

   quickstart/installation
   quickstart/local
   quickstart/theta

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   usage/workflow
   usage/benchmarks
   usage/evaluators
   usage/hyperparamsearch
   usage/nas


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
