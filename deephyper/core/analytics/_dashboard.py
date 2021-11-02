"""
Dashboard
---------
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

    # parser.add_argument("path", type=str, help="Path to the input CSV file.")

    return subparser_name, function_to_call


def main(*args, **kwargs):
    """
    :meta private:
    """

    path_st_app = os.path.join(HERE, "dashboard", "_streamlit_app.py")

    result = subprocess.run(
        ["streamlit", "run", path_st_app],
    )
