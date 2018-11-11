"""
Benchmarks are here for you to test the performance of different search algorithm and reproduce our results. They can also help you to test your installation of deephyper or
discover the many parameters of a search. In deephyper we have two different kind of benchmarks. The first type is `hyper parameters search` benchmarks and the second type is  `neural architecture search` benchmarks. To see a full explanation about the different kind of search please refer to the following pages: `Hyperparameter Searchs <searches/hps/index.html>`_  & `Neural Architecture Search <searches/hps/nas.html>`_.
"""

from deephyper.benchmarks.problem import Problem, HpProblem

__all__ = ['Problem', 'HpProblem']
