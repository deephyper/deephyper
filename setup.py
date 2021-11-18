#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Note: To use the 'upload' functionality of this file, you must:
#   $ pip install twine

import os
import sys
from shutil import rmtree

from setuptools import Command, setup

# on_rtd = os.environ.get("READTHEDOCS") == "True"

# What packages are required for this module to be executed?
REQUIRED = [
    # "tensorflow>=2.0.0",
    # "tensorflow_probability",
    "numpy",  # ==1.19.4",  # working with 1.20.1
    "dh-scikit-optimize==0.9.4",
    "scikit-learn>=0.23.1",
    "tqdm",
    # nas
    "networkx",
    "joblib>=0.10.3",
    "pydot",
    "ray[default]>=1.3.0",
    "pandas>=0.24.2",
    "Jinja2",
    "ConfigSpace>=0.4.20",
    "xgboost",
    "openml==0.10.2",
    "matplotlib>=3.0.3",
]

# What packages are optional?
EXTRAS = {
    "dev": [
        # Test
        "pytest",
        "codecov",
        "pytest-cov",
        # Packaging
        "twine",
        # Formatter and Linter
        "black",
        "rstcheck",
        # Documentation
        "Sphinx~=3.5.4",
        "sphinx-book-theme",
        "nbsphinx",
        "sphinx-copybutton",
        "sphinx-togglebutton",
        "GitPython",
        "ipython",
        # Other
        "deepspace>=0.0.5",
    ],
    "analytics": [
        "jupyter",
        "jupyter_contrib_nbextensions>=0.5.1",
        "nbconvert<6",
        "seaborn>=0.9.1",
    ],
    "hvd": ["horovod>=0.21.3", "mpi4py>=3.0.0"],
    "deepspace": ["deepspace>=0.0.5"],
}

# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))


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
    # $ setup.py publish support.
    cmdclass={
        "upload": UploadCommand,
        "testupload": TestUploadCommand,
        "testinstall": TestInstallCommand,
    }
)
