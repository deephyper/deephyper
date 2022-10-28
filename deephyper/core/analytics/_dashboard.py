"""
Dashboard
---------
A tool to open an interactive dashboard in the browser to help analyse DeepHyper results.

It can be used such as:

.. code-block:: console

    $ deephyper-analytics dashboard --database db.json

Then an interactive dashboard will appear in your browser.
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
    parser.add_argument(
        "-d",
        "--database",
        default="~/.deephyper/db.json",
        help="Path to the default database used for the dashboard.",
    )

    return subparser_name, function_to_call


def main(database, *args, **kwargs):
    """
    :meta private:
    """

    path_st_app = os.path.join(HERE, "dashboard", "_views.py")
    database = os.path.abspath(database)

    # the "--" is a posix standard to separate streamlit arguments from other arguments
    # which are forwarded to the launched script
    subprocess.run(
        ["streamlit", "run", path_st_app, "--", database],
    )
