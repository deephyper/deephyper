"""Visualization tools for Hyperparameter Optimization.
"""
import matplotlib.pyplot as plt
import pandas as pd


def filter_failed_objectives(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out lines from the DataFrame with failed objectives.

    Args:
        df (pd.DataFrame): the results of a Hyperparameter Search.
    Returns:
        pd.DataFrame: the results of a Hyperparameter Search without failed objectives.
    """
    if pd.api.types.is_string_dtype(df.objective):
        df = df[~df.objective.str.startswith("F")]
        df.objective = df.objective.astype(float)
    return df


def parameters_at_max(df: pd.DataFrame) -> dict:
    """Return the parameters at the maximum of the objective function.

    Args:
        df (pd.DataFrame): the results of a Hyperparameter Search.

    Returns:
        dict: the parameters at the maximum of the objective function.
    """
    df = filter_failed_objectives(df)
    config = df.iloc[df.objective.argmax()].to_dict()
    config = {k[2:]: v for k, v in config.items() if k.startswith("p:")}
    return config


def plot_search_trajectory_single_objective_hpo(results, ax=None, **kwargs):
    """Plot the search trajectory of a Single-Objective Hyperparameter Search.

    Args:
        results (pd.DataFrame): the results of a Hyperparameter Search.
        ax (matplotlib.pyplot.axes): the axes to use for the plot.

    Returns:
        (matplotlib.pyplot.figure, matplotlib.pyplot.axes): the figure and axes of the plot.
    """

    scatter_kwargs = dict(marker="o", s=10, c="skyblue")
    scatter_kwargs.update(kwargs)

    fig = plt.gcf()
    if fig is None:
        fig = plt.figure()

    if ax is None:
        ax = fig.gca()

    ax.plot(results.objective.cummax())
    ax.scatter(
        list(range(len(results))),
        results.objective,
        **scatter_kwargs,
    )
    ax.set_xlabel("Evaluations")
    ax.set_ylabel("Objective")

    return fig, ax
