"""
The ``search`` module brings a modular way to implement new search algorithms and two sub modules. One is for hyperparameter search ``deephyper.search.hps`` and one is for neural architecture search ``deephyper.search.nas``.
The ``Search`` class is abstract and has different subclasses such as: ``deephyper.search.ambs`` and ``deephyper.search.ga``.
"""

from deephyper.search.search import Search
# from deephyper.search.hps.ambs import AMBS
# from deephyper.search.hps.ga import GA
# from deephyper.search.nas.ppo_a3c_sync import NasPPOSyncA3C
# from deephyper.search.nas.ppo_a3c_async import NasPPOAsyncA3C
# from deephyper.search.nas.random import NasRandom

__all__ = ['Search'] #, 'AMBS', 'GA', 'NasPPOSyncA3C', 'NasPPOAsyncA3C', 'NasRandom']
