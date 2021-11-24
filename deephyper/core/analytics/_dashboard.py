"""
Dashboard
---------
A tool to open an interactive dashboard in the browser to help analyse DeepHyper results.

It can be use such as:

.. code-block:: console

    $ deephyper-analytics dashboard
"""
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))


def add_subparser(subparsers):
    """
    :meta private:
    """
    subparser_name = "dashboard"
    function_to_call = main

    parser = subparsers.add_parser(
        subparser_name, help="Open a dashboard in the browser."
    )

    return subparser_name, function_to_call


def main(*args, **kwargs):
    """
    :meta private:
    """

    path_st_app = os.path.join(HERE, "dashboard", "_views.py")

    result = subprocess.run(
        ["streamlit", "run", path_st_app],
    )
