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
    "ConfigSpace>=1.1.1",
    "dm-tree",
    "Jinja2>=3.1.4",  # Related to security vulnerability: https://security.snyk.io/vuln/SNYK-PYTHON-JINJA2-6809379
    "matplotlib",
    "numpy>=1.26.0",
    "pandas>=0.24.2",
    "packaging",
    "parse",
    "scikit-learn>=0.23.1",
    "scipy>=1.10",
    "tqdm>=4.64.0",
    "psutil",
    "pymoo>=0.6.0",
    "pyyaml",
]


# !Requirements for tensorflow with keras 2
REQUIRED_TF_KERAS_2 = [
    "tensorflow~=2.17.0",
    "tensorflow_probability~=0.24.0",
    "tf-keras~=2.17.0",
]

# !Requirements for torch
REQUIRED_TORCH = ["torch>=2.0.0"]


# !Transfer Learning for Bayesian Optimization with SVD
REQUIRED_TL_SDV = ["sdv~=1.15.0"]


# What packages are optional?
EXTRAS = {
    "jax-cpu": ["jax[cpu]>=0.3.25", "numpyro[cpu]"],
    "jax-cuda": ["jax[cuda]>=0.3.25", "numpyro[cuda]"],
    "tf-keras2": REQUIRED_TF_KERAS_2,
    "torch": REQUIRED_TORCH,
    "hpo-tl": REQUIRED_TL_SDV,  # Transfer Learning for bayesian optimization,
    "mpi": ["mpi4py>=3.1.3"],
    "ray": ["ray[default]>=1.3.0"],
    "redis": ["redis"],
    "redis-hiredis": ["redis[hiredis]"],
    "dev": [
        # Test
        "pytest",
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
        "sphinx>=5",
        "sphinx-book-theme==1.1.3",
        "pydata-sphinx-theme==0.15.4",
        "sphinx-copybutton",
        "sphinx-gallery",
        # "sphinx_lfs_content", # Try to not use lfs anymore
        "sphinx-togglebutton",
    ],
}

# Default dependencies for DeepHyper
DEFAULT_DEPENDENCIES = REQUIRED[:]
DEFAULT_DEPENDENCIES += EXTRAS["tf_keras2"]
DEFAULT_DEPENDENCIES += EXTRAS["torch"]
DEFAULT_DEPENDENCIES += EXTRAS["hpo-tl"]
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
