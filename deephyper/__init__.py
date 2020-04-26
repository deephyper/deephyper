"""
DeepHyper is a scalable automated machine learning (`AutoML <https://en.wikipedia.org/wiki/Automated_machine_learning>`_) package for developing deep neural networks for scientific applications.
It comprises two components:

* :ref:`create-new-nas-problem`: fully-automated search for high-performing deep neural network architectures

* :ref:`create-new-hps-problem`: optimizing hyperparameters for a given reference model


DeepHyper provides an infrastructure that targets experimental research in NAS and HPS methods, scalability, and portability across diverse supercomputers.
It comprises three modules:

* :ref:`benchmarks`: Tools for defining NAS and HPS problems, as well as a curated set of sample benchmark problems for judging the efficacy of novel search algorithms.

* :ref:`evaluators`: A simple interface for NAS and HPS codes to dispatch model evaluation tasks. Implementations range from `subprocess` for laptop experiments to `ray` and `balsam` for large-scale runs on HPC systems.

* :ref:`SearchDH`: Search methods for NAS and HPS.  By extending the generic `Search` class, one can easily add new NAS or HPS methods to DeepHyper.


DeepHyper installation requires **Python 3.7**.

"""
import os
import warnings
from deephyper.__version__ import __version__, __version_suffix__

name = "deephyper"
version = __version__

# ! Check if a balsam db is connected or not
if os.environ.get("BALSAM_DB_PATH") is None:
    os.environ["BALSAM_SPHINX_DOC_BUILD_ONLY"] = "TRUE"

# Suppress warnings from skopt using deprecated sklearn API
warnings.filterwarnings(
    "ignore", category=FutureWarning, message="sklearn.externals.joblib is deprecated"
)
warnings.filterwarnings(
    "ignore", category=FutureWarning, message="the sklearn.metrics.scorer module"
)
