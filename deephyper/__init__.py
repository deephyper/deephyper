"""
DeepHyper's software architecture is designed to be modular and extensible. It is built on top of the following main subpackages:

* :mod:`deephyper.ensemble`: Tools to build ensembles of neural networks with uncertainty quantification.
* :mod:`deephyper.nas`: Tools to define neural architecture search space and evaluation strategy.
* :mod:`deephyper.hpo`: Tools for defining neural architecture and hyper-parameter search problems.
* :mod:`deephyper.evaluator` : Tools to distribute the evaluation of tasks (e.g., neural network trainings).
* :mod:`deephyper.search`: Tools to define search strategies for neural architecture search and hyper-parameter optimization.
* :mod:`deephyper.hpo.stopper`: Tools to define multi-fidelity strategies for neural architecture and hyper-parameter optimization.


DeepHyper installation requires **Python >= 3.9**.

"""

import warnings
from deephyper.__version__ import __version__, __version_suffix__  # noqa: F401

name = "deephyper"
version = __version__

# Suppress warnings from deephyper.hpo.skopt using deprecated sklearn API
warnings.filterwarnings(
    "ignore", category=FutureWarning, message="sklearn.externals.joblib is deprecated"
)
warnings.filterwarnings(
    "ignore", category=FutureWarning, message="the sklearn.metrics.scorer module"
)
