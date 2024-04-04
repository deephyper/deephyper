"""Visualization tools for Hyperparameter Optimization.
"""

from typing import Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from deephyper.analysis import rank


def read_results_from_csv(file_path: str) -> pd.DataFrame:
    """Read the results of a Hyperparameter Search from a CSV file.

    Args:
        file_path (str): the path to the CSV file.

    Returns:
        pd.DataFrame: the results of a Hyperparameter Search.
    """
    return pd.read_csv(file_path, index_col=None)


def filter_failed_objectives(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Filter out lines from the DataFrame with failed objectives.

    Args:
        df (pd.DataFrame): the results of a Hyperparameter Search.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: ``df_without_failures, df_with_failures`` the first are results of a Hyperparameter Search without failed objectives and the second are results of Hyperparameter search with failed objectives.
    """
    # Single-Objective
    if "objective" in df.columns:
        if pd.api.types.is_string_dtype(df["objective"]):
            mask = df["objective"].str.startswith("F")

            df_with_failures = df[mask]

            df_without_failures = df[~mask]
            df_without_failures.loc[:, "objective"] = df_without_failures[
                "objective"
            ].astype(float)
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


def plot_search_trajectory_single_objective_hps(
    results, show_failures: bool = True, column="objective", ax=None, **kwargs
):
    """Plot the search trajectory of a Single-Objective Hyperparameter Search.

    Args:
        results (pd.DataFrame): the results of a Hyperparameter Search.
        show_failures (bool, optional): whether to show the failed objectives. Defaults to ``True``.
        column (str, optional): the column to use for the y-axis of the plot. Defaults to ``"objective"``.
        ax (matplotlib.pyplot.axes): the axes to use for the plot.

    Returns:
        (matplotlib.pyplot.figure, matplotlib.pyplot.axes): the figure and axes of the plot.
    """

    if results[column].dtype != np.float64:
        x = np.arange(len(results))
        mask_failed = np.where(results[column].str.startswith("F"))[0]
        mask_success = np.where(~results[column].str.startswith("F"))[0]
        x_success, x_failed = x[mask_success], x[mask_failed]
        y_success = results[column][mask_success].astype(float)
    else:
        x = np.arange(len(results))
        x_success = x
        x_failed = np.array([])
        y_success = results[column]

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

    if show_failures and len(x_failed) > 0:
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
    ax.set_xlim(x.min(), x.max())

    return fig, ax


def compile_worker_activity(results, profile_type="submit/gather"):
    """Compute the number of active workers.

    Args:
        results (pd.DataFrame): the results of a Hyperparameter Search.
        profile_type (str, optional): the type of profile to build. It can be `"submit/gather"` or `"start/end"`. Defaults to "submit/gather".

    Returns:
        timestamps, n_jobs_active: a list of timestamps and a list of the number of active jobs at each timestamp.
    """
    if profile_type == "submit/gather":
        key_start, key_end = "m:timestamp_submit", "m:timestamp_gather"
    elif profile_type == "start/end":
        key_start, key_end = "m:timestamp_start", "m:timestamp_end"
    else:
        raise ValueError(
            f"Unknown profile_type='{profile_type}' it should be one of ['submit/gather', 'start/end']."
        )

    if key_start not in results.columns or key_end not in results.columns:
        raise ValueError(
            f"Columns '{key_start}' and '{key_end}' are not present in the DataFrame."
        )

    results = results.sort_values(by=[key_start], ascending=True)

    history = []

    for _, row in results.iterrows():
        history.append((row[key_start], 1))
        history.append((row[key_end], -1))

    history = sorted(history, key=lambda v: v[0])
    nb_workers = 0
    timestamp = np.zeros((len(history) + 1,))
    n_jobs_running = np.zeros((len(history) + 1,))
    for i, (time, incr) in enumerate(history):
        nb_workers += incr
        timestamp[i + 1] = time
        n_jobs_running[i + 1] = nb_workers

    return timestamp, n_jobs_running


def plot_worker_utilization(
    results,
    num_workers: int = None,
    profile_type: str = "submit/gather",
    ax=None,
    **kwargs,
):
    """Plot the worker utilization of a search.

    Args:
        results (pd.DataFrame): the results of a Hyperparameter Search.
        num_workers (int, optional): the number of workers. If passed the normalized utilization will be shown (/num_workers). Otherwise, the raw number of active workers is shown. Defaults to ``None``.
        profile_type (str, optional): the type of profile to build. It can be `"submit/gather"` or `"start/end"`. Defaults to "submit/gather".
        ax (matplotlib.pyplot.axes): the axes to use for the plot.

    Returns:
        (matplotlib.pyplot.figure, matplotlib.pyplot.axes): the figure and axes of the plot.
    """

    x, y = compile_worker_activity(results, profile_type=profile_type)

    if num_workers:
        y = y / num_workers

    plot_kwargs = dict()
    plot_kwargs.update(kwargs)

    fig = plt.gcf()
    if fig is None:
        fig = plt.figure()

    if ax is None:
        ax = fig.gca()

    ax.plot(x, y, **plot_kwargs)

    ax.set_xlabel("Time (sec.)")
    if num_workers:
        ax.set_ylabel("Utilization")
    else:
        ax.set_ylabel("Active Workers")
    ax.grid(True)
    ax.set_xlim(x.min(), x.max())

    return fig, ax


class RealNormalizer:
    def fit(self, x):
        self.x_min = np.min(x)
        self.x_max = np.max(x)

    def transform(self, x):
        # min-max
        return x - self.x_min / (self.x_max - self.x_min)

    def inverse_transform(self, x):
        return x * (self.x_max - self.x_min) + self.x_min


class ColumnNormalizer:
    def fit(self, x):
        if np.all(np.issubdtype(x, np.number)):  # float or int
            if np.all(np.issubdtype(x, np.integer)):  # int
                pass
            else:  # mixed of floats and ints
                pass
        else:  # categories
            pass


def plot_parallel_coordinate(
    results,
    parameters_columns=None,
    objective_column="objective",
    rank_mode="min",
    highlight=True,
    constant_predictor=0.035726056,
):
    """Plot a parallel coordinate plot of the hyperparameters.
    Args:
        results (pd.DataFrame): the results of a Hyperparameter Search.
        parameters_columns (list, optional): list of columns to include in the plot.
        objective_column (str): name of the objective column
        rank_mode (str, optional): mode of ranking. Defaults to "min".
        highlight (bool, optional): whether to highlight the best solutions. Defaults to True.
        constant_predictor (float, optional): value to compare the objective to. Defaults to 0.035726056.
    Returns:
        fig: figure of the parallel coordinate plot
    """

    if parameters_columns is None:
        cols = [c for c in results.columns if c.startswith("p:")]
    else:
        cols = parameters_columns

    cols += [objective_column]

    results = results.copy()

    results, _ = filter_failed_objectives(results)

    for col in results.columns:
        if results[col].dtype == bool:
            results[col] = results[col].astype(str)

    column_values = results[cols].values
    objective_values = results[objective_column].values

    cmap = mpl.colormaps["plasma"]

    fig, ax = plt.subplots(1, len(cols) - 1, sharey=False)

    x = np.arange(0, len(cols)).tolist()

    for i in range(len(cols) - 1):
        for j in range(len(results)):
            y = column_values[j].tolist()
            ax[i].plot(x, y, color=cmap(objective_values[j]))

        # Set x-axis
        if i == len(cols) - 2:
            # Objective axis
            ax[i].set_xticks(ticks=x, labels=["" for _ in x], rotation=45)
        else:
            ax[i].set_xticks(ticks=x, labels=cols, rotation=45)
            ax[i].set_yticks([])

        ax[i].set_xlim([x[i], x[i] + 1])
        ax[i].set_yticks([])

    norm = mpl.cm.ScalarMappable(norm=None, cmap=cmap)
    norm.set_clim(objective_values.min(), objective_values.max())
    plt.colorbar(mappable=norm, ax=ax[-1], label=objective_column)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0)

    return fig, ax
