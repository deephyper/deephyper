"""
Quick Plot
----------

A tool to have quick and simple visualization from your data.

It can be use such as:

.. code-block:: console

    $ deephyper-analytics quickplot nas_big_data/combo/exp_sc21/combo_1gpu_8_age/infos/results.csv
    $ deephyper-analytics quickplot save/history/*.json --xy time val_r2
    $ deephyper-analytics quickplot save/history/*.json --xy epochs val_r2
"""

import json
from datetime import datetime

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
        default=[],
        help="name of x y variables in the CSV file.",
    )

    return subparser_name, function_to_call


def plot_for_single_csv(path: str, xy: list):
    """Generate a plot from a single CSV file.

    :meta private:

    Args:
        path (str): Path to the CSV file.
        xy (list): If empty ``list`` then it will use ``"elapsed_sec"`` for x-axis and ``"objective"`` for the y-axis.

    Raises:
        DeephyperRuntimeError: if only 1 or more than 2 arguments are provided.
    """

    if len(xy) == 0:
        xy = ["elapsed_sec", "objective"]
    elif len(xy) != 2:
        raise DeephyperRuntimeError(
            "--xy must take two arguments such as '--xy elapsed_sec objective'"
        )

    df = pd.read_csv(path)

    plt.figure()

    plt.scatter(df[xy[0]], df[xy[1]], s=5, alpha=1.0)

    plt.xlabel(xy[0])
    plt.ylabel(xy[1])
    plt.grid()
    plt.tight_layout()
    plt.show()


def plot_for_single_json(path: str, xy: list):
    """[summary]

    :meta private:

    Args:
        path (str): [description]
        xy (list): [description]

    Raises:
        DeephyperRuntimeError: [description]
    """

    if len(xy) == 0:
        xy = ["epochs", "val_loss"]
    elif len(xy) != 2:
        raise DeephyperRuntimeError(
            "--xy must take two arguments such as '--xy epochs val_loss'"
        )

    xlabel, ylabel = xy

    with open(path, "r") as f:
        history = json.load(f)

    x = list(range(len(history[ylabel]))) if xlabel == "epochs" else history[xlabel]
    y = history[ylabel]

    plt.figure()

    plt.plot(x, y)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.tight_layout()
    plt.show()


def plot_multiple_training(path: list, ylabel: str):
    """[summary]

    :meta private:

    Args:
        path (list): [description]
        ylabel (str): [description]
    """
    for p in path:
        with open(p, "r") as f:
            history = json.load(f)

        x = list(range(len(history[ylabel])))
        y = history[ylabel]

        plt.plot(x, y)

    plt.xlabel("Epochs")


def plot_multiple_objective_wrp_time(path: list, ylabel: str):
    """[summary]

    :meta private:

    Args:
        path (list): [description]
        ylabel (str): [description]
    """

    times = []
    objectives = []

    for p in path:
        with open(p, "r") as f:
            history = json.load(f)

        time = "_".join(p[:-5].split("_")[-2:])
        time = datetime.strptime(time, "%d-%b-%Y_%H-%M-%S").timestamp()
        times.append(time)

        objective = max(history[ylabel])
        objectives.append(objective)

    plt.scatter(times, objectives)

    plt.xlabel("Time")


def plot_for_multiple_json(path: list, xy: list):
    """
    :meta private:
    """
    if len(xy) == 0:
        xy = ["epochs", "val_loss"]
    elif len(xy) != 2:
        raise DeephyperRuntimeError(
            "--xy must take two arguments such as '--xy epochs val_loss'"
        )

    xlabel, ylabel = xy

    plt.figure()

    if xlabel == "epochs":
        plot_multiple_training(path, ylabel)
    elif xlabel == "time":
        plot_multiple_objective_wrp_time(path, ylabel)

    plt.ylabel(ylabel)
    plt.grid()
    plt.tight_layout()
    plt.show()


def main(path: list, xy: list, *args, **kwargs):
    """
    :meta private:
    """

    def extension(path):
        return path.split(".")[-1]

    if len(path) == 1:
        if extension(path[0]) == "csv":
            plot_for_single_csv(path[0], xy)
        elif extension(path[0]) == "json":
            plot_for_single_json(path[0], xy)
        else:
            raise DeephyperRuntimeError(
                f"Extension of input file '{extension(path[0])}' is not yet supported."
            )
    else:

        # Comparing multiple results.csv files (different search experiments)
        if all([extension(p) == "csv" for p in path]):
            raise DeephyperRuntimeError(
                "Comparison of multiple experiments is not yet supported."
            )
        # Comparing multiple history.json files (different neural networks)
        elif all([extension(p) == "json" for p in path]):
            plot_for_multiple_json(path, xy)
        else:
            raise DeephyperRuntimeError(
                "Multiple input files should all have the same extension '.csv' or '.json'"
            )
