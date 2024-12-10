"""
DeepHyper's software architecture is designed to be modular and extensible. It is built on top of the following main subpackages:

* :mod:`deephyper.analysis`: Tools to analyse results from DeepHyper.
* :mod:`deephyper.ensemble`: Tools to build ensembles of neural networks with uncertainty quantification.
* :mod:`deephyper.evaluator` : Tools to distribute the evaluation of tasks (e.g., neural network trainings).
* :mod:`deephyper.hpo`: Tools to define and run hyperparameter optimization (HPO) and neural architecture search (NAS).
* :mod:`deephyper.predictor`: Tools to wrap pure predictive models (i.e., that can only predict).
* :mod:`deephyper.stopper`: Tools to define multi-fidelity strategies for hyperparameter optimization (HPO) and neural architecture search (NAS).


DeepHyper installation requires **Python >= 3.10**.

"""

from deephyper.__version__ import __version__, __version_suffix__  # noqa: F401

name = "deephyper"
version = __version__
