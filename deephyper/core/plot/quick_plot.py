"""
usage:

::

    $ deephyper-analytics quickplot nas_big_data/combo/exp_sc21/combo_1gpu_8_age/infos/results.csv
"""

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from deephyper.core.exceptions import DeephyperRuntimeError

width = 8
height = width / 1.618
fontsize = 18
matplotlib.rcParams.update(
    {
        "font.size": fontsize,
        "figure.figsize": (width, height),
        "figure.facecolor": "white",
        "savefig.dpi": 72,
        "figure.subplot.bottom": 0.125,
        "figure.edgecolor": "white",
        "xtick.labelsize": fontsize,
        "ytick.labelsize": fontsize,
    }
)


def add_subparser(subparsers):
    subparser_name = "quickplot"
    function_to_call = main

    parser = subparsers.add_parser(
        subparser_name, help="Tool to generate a quick 2D plot from file."
    )

    # best search_spaces
    parser.add_argument("path", nargs="+", type=str)
    parser.add_argument(
        "--xy",
        metavar="xy",
        type=str,
        nargs=2,
        default=["elapsed_sec", "objective"],
        help="name of x y variables in the CSV file.",
    )

    return subparser_name, function_to_call

def plot_for_csv(path: list, xy: list):
    df = pd.read_csv(path)

    plt.figure()

    plt.scatter(df[xy[0]], df[xy[1]], s=1.5, alpha=0.8)

    plt.xlabel(xy[0])
    plt.ylabel(xy[1])
    plt.grid()
    plt.tight_layout()
    plt.show()

def main(path: list, xy: list, *args, **kwargs):

    if len(path) == 1:
        input_extension = path[0].split(".")[-1]
        if input_extension == "csv":
            plot_for_csv(path[0], xy)
        else:
            raise DeephyperRuntimeError(f"Extension of input file '{input_extension}' is not yet supported.")
    else:
        raise DeephyperRuntimeError("Multiple input files not yet supported for quickplot.")


