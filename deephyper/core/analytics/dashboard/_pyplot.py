import json
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import streamlit as st
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


def to_max(l):
    r = [l[0]]
    for e in l[1:]:
        r.append(max(r[-1], e))
    return r


def plot_single_line(df, x_label, y_label):

    fig = plt.figure()

    plt.scatter(df[x_label], df[y_label])

    plt.xlabel(x_label.title())
    plt.ylabel(y_label.title())
    plt.grid()
    plt.tight_layout()

    st.pyplot(fig)
    plt.close()


def plot_single_line_improvement(df, x_label, y_label):

    fig = plt.figure()

    plt.plot(df[x_label], to_max(df[y_label].tolist()))

    plt.xlabel(x_label.title())
    plt.ylabel(y_label.title())
    plt.grid()
    plt.tight_layout()

    st.pyplot(fig)
    plt.close()


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
