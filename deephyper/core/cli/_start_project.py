"""
Start a DeepHyper Project
-------------------------

Command line to create a new DeepHyper project package. The package is automatically installed to the current virtual Python environment.

It can be used with:

.. code-block:: console

    $ deephyper start-project project_name
"""
import os
import pathlib
import subprocess


def add_subparser(subparsers):
    """
    :meta private:
    """
    subparser_name = "start-project"
    function_to_call = main

    subparser = subparsers.add_parser(
        subparser_name, help="Set up a new DeepHyper project as a Python package."
    )

    subparser.add_argument("path", type=str, help="Path to the new project directory.")
    subparser.set_defaults(func=function_to_call)


def main(path, *args, **kwargs):
    """
    :meta private:
    """
    path = os.path.abspath(path)
    project_name = os.path.basename(path)
    path_pkg = os.path.join(path, project_name)
    pathlib.Path(path_pkg).mkdir(parents=True, exist_ok=False)
    with open(os.path.join(path, "setup.py"), "w") as fp:
        fp.write(
            f"from setuptools import setup, find_packages\n\nsetup(\n    name='{project_name}',\n    packages=find_packages(),\n    install_requires=[]\n)"
        )
    with open(os.path.join(path_pkg, "__init__.py"), "w") as fp:
        pass
    with open(os.path.join(path, ".deephyper"), "w") as fp:
        pass
    result = subprocess.run(
        ["pip", "install", "-e", f"{path}"],
        stdout=subprocess.PIPE,
    )
    if result.returncode != 0:
        print(
            f"The package could not be installed automatically! Export the following in your environment to access the package from anywhere:\n"
            f"export PYTHONPATH={path}:$PYTHONPATH"
        )
    else:
        print(result.stdout.decode("utf-8"), end="")
