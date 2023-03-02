#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Note: To use the 'upload' functionality of this file, you must:
#   $ pip install twine

import os
import platform
import sys
from shutil import rmtree

from setuptools import Command, setup

# path of the directory where this file is located
here = os.path.abspath(os.path.dirname(__file__))

# query platform informations, e.g. 'macOS-12.0.1-arm64-arm-64bit'
platform_infos = platform.platform()


# What packages are required for this module to be executed?
REQUIRED = [
    "ConfigSpace>=0.4.20",
    "dm-tree",
    "Jinja2<3.1",
    "numpy",  # ==1.19.4",  # working with 1.20.1
    "pandas>=0.24.2",
    "packaging",
    "parse",
    "scikit-learn>=0.23.1",
    "scipy>=1.7",
    "tqdm>=4.64.0",
    "pyyaml",
]


# !Requirements for Neural Architecture Search (NAS)
REQUIRED_NAS = ["networkx", "pydot"]

REQUIRED_NAS_PLATFORM = {
    "default": ["tensorflow>=2.0.0", "tensorflow_probability"],
    "macOS-arm64": ["tensorflow_probability~=0.14"],
}
# if "macOS" in platform_infos and "arm64" in platform_infos:
#     REQUIRED_NAS = REQUIRED_NAS + REQUIRED_NAS_PLATFORM["macOS-arm64"]
# else:  # x86_64
REQUIRED_NAS = REQUIRED_NAS + REQUIRED_NAS_PLATFORM["default"]

# !Requirements for Pipeline Optimization for ML (popt)
REQUIRED_POPT = ["xgboost"]

# !Requirements for Automated Deep Ensemble with Uncertainty Quantification (AutoDEUQ)
REQUIRED_AUTODEUQ = REQUIRED_NAS + ["ray[default]>=1.3.0"]

# !Transfer Learning for Bayesian Optimization with SVD
REQUIRED_TL_SDV = ["sdv>=0.17.1"]


# What packages are optional?
EXTRAS = {
    "autodeuq": REQUIRED_AUTODEUQ,  # automated deep ensemble with uncertainty quantification
    "automl": ["xgboost"],  # for automl with scikit-learn
    "jax-cpu": ["jax[cpu]>=0.3.25", "numpyro[cpu]"],
    "jax-cuda": ["jax[cuda]>=0.3.25", "numpyro[cuda]"],
    "hps": [],  # hyperparameter search (already the base requirements)
    "nas": REQUIRED_NAS,  # neural architecture search
    "hps-tl": REQUIRED_TL_SDV,  # transfer learning for bayesian optimization,
    "mpi": ["mpi4py>=3.1.3"],
    "ray": ["ray[default]>=1.3.0"],
    "redis": ["redis[hiredis]"],
    "dev": [
        # Test
        "codecov",
        "pytest",
        "pytest-cov",
        # Packaging
        "twine",
        # Formatter and Linter
        "black==22.6.0",
        "flake8==5.0.4",
        "pre-commit",
        "rstcheck",
        # Documentation
        "GitPython",
        "ipython",
        "nbsphinx",
        "Sphinx~=3.5.4",
        "sphinx-book-theme==0.3.2",
        "sphinx-copybutton",
        "sphinx-gallery",
        "sphinx_lfs_content",
        "sphinx-togglebutton",
    ],
    "analytics": [
        "altair",
        "jupyter",
        "jupyter_contrib_nbextensions>=0.5.1",
        "nbconvert<6",
        "streamlit",
        "streamlit-aggrid",
        "tinydb",
    ],
    "hvd": ["horovod>=0.21.3", "mpi4py>=3.0.0"],
}

# Default dependencies for DeepHyper
DEFAULT_DEPENDENCIES = REQUIRED[:]
DEFAULT_DEPENDENCIES += EXTRAS["nas"]
DEFAULT_DEPENDENCIES += EXTRAS["autodeuq"]
DEFAULT_DEPENDENCIES += EXTRAS["hps-tl"]
DEFAULT_DEPENDENCIES += EXTRAS["jax-cpu"]
EXTRAS["default"] = DEFAULT_DEPENDENCIES

# Useful commands to build/upload the wheel to PyPI


class UploadCommand(Command):
    """Support setup.py upload."""

    description = "Build and publish the package."
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print("\033[1m{0}\033[0m".format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status("Removing previous builds…")
            rmtree(os.path.join(here, "dist"))
        except OSError:
            pass

        self.status("Building Source and Wheel (universal) distribution…")
        os.system("{0} setup.py sdist bdist_wheel --universal".format(sys.executable))

        self.status("Uploading the package to PyPI via Twine…")
        os.system("twine upload dist/*")

        sys.exit()


class TestUploadCommand(Command):
    """Support setup.py testupload."""

    description = "Build and publish the package to test.pypi."
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print("\033[1m{0}\033[0m".format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status("Removing previous builds…")
            rmtree(os.path.join(here, "dist"))
        except OSError:
            pass

        self.status("Building Source and Wheel (universal) distribution…")
        os.system("{0} setup.py sdist bdist_wheel --universal".format(sys.executable))

        self.status("Uploading the package to PyPI via Twine…")
        os.system("twine upload --repository-url https://test.pypi.org/legacy/ dist/*")

        sys.exit()


class TestInstallCommand(Command):
    """Support setup.py testinstall"""

    description = "Install deephyper from TestPyPI."
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print("\033[1m{0}\033[0m".format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        self.status("Downloading the package from Test PyPI and installing it")
        os.system("pip install --index-url https://test.pypi.org/simple/ deephyper")

        sys.exit()


# Where the magic happens:
setup(
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    cmdclass={
        "upload": UploadCommand,
        "testupload": TestUploadCommand,
        "testinstall": TestInstallCommand,
    },
)
