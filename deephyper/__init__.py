"""
Deephyper is a Python package which provides a common interface for the implementation and study of scalable hyperparameter and neural architecture search methods. In this package we provide to the user different modules:
    - ``benchmark``: a set of problems for hyperparameter or neural architecture search which the user can use to compare our different search algorithms or as examples to build their own problems.
    - ``evaluator``: a set of objects which help to run search on different systems and for different cases such as quick and light experiments or long and heavy runs.
    - ``search``: a set of algorithms for hyperparameter and neural architecture search. You will also find a modular way to define new search algorithms and specific sub modules for hyperparameter or neural architecture search.
"""
from deephyper.__version__ import __version__
name = 'deephyper'
version = __version__