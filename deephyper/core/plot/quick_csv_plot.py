"""
usage: $ deephyper-analytics  plot csv -p results.csv --xy elapsed_sec objective
"""
import sys

import matplotlib.pyplot as plt
import pandas as pd


def add_subparser(subparsers):
    subparser_name = "plot"
    function_to_call = main

    parser = subparsers.add_parser(
        subparser_name, help="Tool to generate a quick 2D plot from file."
    )
    subparsers = parser.add_subparsers(help="Kind of analytics.")

    # best search_spaces
    subparser = subparsers.add_parser("csv", help="Plot for CSV files.")
    subparser.add_argument(
        "--path", "-p", type=str, default="results.csv", help="Path to CSV file."
    )
    subparser.add_argument(
        "--xy",
        metavar="xy",
        type=str,
        nargs=2,
        default=["elapsed_sec", "objective"],
        help="name of x y variables in the CSV file.",
    )

    return subparser_name, function_to_call


def main(path, xy, *args, **kwargs):

    if sys.argv[2] == "csv":
        df = pd.read_csv(path)

        plt.figure()

        plt.scatter(df[xy[0]], df[xy[1]])

        plt.xlabel(xy[0])
        plt.ylabel(xy[1])

        plt.show()

