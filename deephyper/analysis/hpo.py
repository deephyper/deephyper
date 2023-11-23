"""Visualization tools for Hyperparameter Optimization.
"""
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def filter_failed_objectives(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Filter out lines from the DataFrame with failed objectives.

    Args:
        df (pd.DataFrame): the results of a Hyperparameter Search.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: ``df_without_failures, df_with_failures`` the first are results of a Hyperparameter Search without failed objectives and the second are results of Hyperparameter search with failed objectives.
    """
    # Single-Objective
    if "objective" in df.columns:
        if pd.api.types.is_string_dtype(df.objective):
            mask = df.objective.str.startswith("F")

            df_with_failures = df[mask]

            df_without_failures = df[~mask]
            df_without_failures.loc[
                :, "objective"
            ] = df_without_failures.objective.astype(float)
        else:
            df_without_failures = df
            df_with_failures = df[np.zeros(len(df), dtype=bool)]

    # Multi-Objective
    elif "objective_0" in df.columns:
        objcol = list(df.filter(regex=r"^objective_\d+$").columns)

        mask = np.zeros(len(df), dtype=bool)
        for col in objcol:
            if pd.api.types.is_string_dtype(df[col]):
                mask = mask | df[col].str.startswith("F")

        df_with_failures = df[mask]
        df_without_failures = df[~mask]
        df_without_failures.loc[:, objcol] = df_without_failures[objcol].astype(float)
    else:
        raise ValueError(
            "The DataFrame does not contain neither a column named 'objective' nor columns named 'objective_<int>'."
        )

    return df_without_failures, df_with_failures


def parameters_at_max(
    df: pd.DataFrame, column: str = "objective"
) -> Tuple[dict, float]:
    """Return the parameters at the maximum of the objective function.

    Args:
        df (pd.DataFrame): the results of a Hyperparameter Search.
        column (str, optional): the column to use for the maximization. Defaults to ``"objective"``.

    Returns:
        Tuple[dict, float]: the parameters at the maximum of the ``column`` and its corresponding value.
    """
    df, _ = filter_failed_objectives(df)
    idx = df[column].argmax()
    value = df.iloc[idx][column]
    config = df.iloc[idx].to_dict()
    config = {k[2:]: v for k, v in config.items() if k.startswith("p:")}
    return config, value


def plot_search_trajectory_single_objective_hpo(
    results, show_failures: bool = True, ax=None, **kwargs
):
    """Plot the search trajectory of a Single-Objective Hyperparameter Search.

    Args:
        results (pd.DataFrame): the results of a Hyperparameter Search.
        show_failures (bool, optional): whether to show the failed objectives. Defaults to ``True``.
        ax (matplotlib.pyplot.axes): the axes to use for the plot.

    Returns:
        (matplotlib.pyplot.figure, matplotlib.pyplot.axes): the figure and axes of the plot.
    """

    if results.objective.dtype != np.float64:
        x = np.arange(len(results))
        mask_failed = np.where(results.objective.str.startswith("F"))[0]
        mask_success = np.where(~results.objective.str.startswith("F"))[0]
        x_success, x_failed = x[mask_success], x[mask_failed]
        y_success = results.objective[mask_success].astype(float)

    y_min, y_max = y_success.min(), y_success.max()
    y_min = y_min - 0.05 * (y_max - y_min)
    y_max = y_max - 0.05 * (y_max - y_min)

    scatter_kwargs = dict(marker="o", s=10, c="skyblue")
    scatter_kwargs.update(kwargs)

    fig = plt.gcf()
    if fig is None:
        fig = plt.figure()

    if ax is None:
        ax = fig.gca()

    ax.plot(x_success, y_success.cummax())
    ax.scatter(x_success, y_success, **scatter_kwargs, label="Successes")

    if show_failures:
        ax.scatter(
            x_failed,
            np.full_like(x_failed, y_min),
            marker="v",
            color="red",
            label="Failures",
        )

    ax.set_xlabel("Evaluations")
    ax.set_ylabel("Objective")
    ax.legend()
    ax.grid(True)
    # ax.set_ylim(y_min, y_max)
    ax.set_xlim(x.min(), x.max())

    return fig, ax
