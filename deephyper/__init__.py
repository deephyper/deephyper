"""
DeepHyper is a scalable automated machine learning (`AutoML <https://en.wikipedia.org/wiki/Automated_machine_learning>`_) package for developing deep neural networks for scientific applications.
It comprises two components:

* ``Neural architecture search (NAS)``: It is designed for automatically searching for high-performing the deep neural network search_space.

* ``Hyperparameter search (HPS)``: It is designed for automatically searching for high-performing hyperparameters for a given deep neural network search_space.


DeepHyper provides an infrastructure that targets experimental research in NAS and HPS methods, scalability, and portability across diverse supercomputers.
It comprises three modules:

* ``benchmark``: Set of test problems for NAS and HPS that can be used for comparing different search methods. They can serve as examples to build new user-defined problems.

* ``evaluator``: Set of objects to run NAS and HPS on different target systems (from laptop to supercomputers) covering different use cases (quick/light experiments on laptop for testing and development to large production runs on supercomputers).

* ``search``: Set of search methods for NAS and HPS. It provides a modular way to define new search HPS and NAS search methods and submodules for implementing HPS and NAS.

DeepHyper installation requires **Python 3.6**.

"""
import os
from deephyper.__version__ import __version__, __version_suffix__
name = 'deephyper'
version = __version__

# ! Check if a balsam db is connected or not
if os.environ.get("BALSAM_DB_PATH") is None:
    os.environ["BALSAM_SPHINX_DOC_BUILD_ONLY"] = "TRUE"
