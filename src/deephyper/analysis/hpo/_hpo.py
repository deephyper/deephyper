"""Visualization tools for Hyperparameter Optimization."""

from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

from deephyper.analysis import rank
from deephyper.analysis.hpo._paxplot import pax_parallel


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
        Tuple[pd.DataFrame, pd.DataFrame]: ``df_without_failures, df_with_failures`` the first are
        results of a Hyperparameter Search without failed objectives and the second are results of
        Hyperparameter search with failed objectives.
    """
    # Single-Objective
    if "objective" in df.columns:
        if pd.api.types.is_string_dtype(df["objective"]):
            mask = df["objective"].str.startswith("F")

            df_with_failures = df[mask]

            df_without_failures = df[~mask]
            df_without_failures = df_without_failures.astype({"objective": float})
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
        df_without_failures = df_without_failures.astype({objcol_i: float for objcol_i in objcol})
    else:
        raise ValueError(
            "The DataFrame does not contain neither a column named 'objective' nor columns named "
            "'objective_<int>'."
        )

    return df_without_failures, df_with_failures


def parameters_from_row(row: pd.Series) -> dict:
    """Extract the parameters from a row of a DataFrame.

    Args:
        row (pd.Series): a row of a DataFrame.

    Returns:
        dict: the parameters of the row.
    """
    return {k[2:]: v for k, v in row.to_dict().items() if k.startswith("p:")}


def parameters_at_max(df: pd.DataFrame, column: str = "objective") -> Tuple[dict, float]:
    """Return the parameters at the maximum of the given ``column`` function.

    Args:
        df (pd.DataFrame): the results of a Hyperparameter Search.
        column (str, optional): the column to use for the maximization. Defaults to ``"objective"``.

    Returns:
        Tuple[dict, float]: the parameters at the maximum of the ``column`` and its corresponding
            value.
    """
    df, _ = filter_failed_objectives(df)
    idx = df[column].argmax()
    value = df.iloc[idx][column]
    config = parameters_from_row(df.iloc[idx])
    return config, value


def parameters_at_topk(
    df: pd.DataFrame, column: str = "objective", k: int = 1
) -> List[Tuple[dict, float]]:
    """Return the parameters at the top-k of the given ``column``.

    Args:
        df (pd.DataFrame): the results of a Hyperparameter Search.
        column (str, optional): the column to use for the maximization. Defaults to ``"objective"``.
        k (int, optional): the number of top-k to return. Defaults to ``1``.

    Returns:
        List[Tuple[dict, float]]: the parameters at the maximum of the ``column`` and its
            corresponding value.
    """
    df, _ = filter_failed_objectives(df)
    df = df.nlargest(k, columns=column)
    return [(parameters_from_row(row), row[column]) for _, row in df.iterrows()]


def plot_search_trajectory_single_objective_hpo(
    results,
    show_failures: bool = True,
    column="objective",
    mode="max",
    x_units="evaluations",
    label=None,
    ax=None,
    plot_kwargs=None,
    scatter_success_kwargs=None,
    scatter_failure_kwargs=None,
):
    """Plot the search trajectory of a Single-Objective Hyperparameter Optimization.

    Args:
        results (pd.DataFrame): the results of Hyperparameter Optimization.
        show_failures (bool, optional): whether to show the failed objectives. Defaults to ``True``.
        column (str, optional): the column to use for the y-axis of the plot. Defaults to
            ``"objective"``.
        mode (str, optional): if the plot should be made for minimization ``"min"`` or maximization
            ``"max"``. Defaults to ``"max"``.
        x_units (str, optional): if the plot should be made with respect to evaluations
            ``"evaluations"`` or time ``"seconds"``. Defaults to ``"evaluations"``.
        label (str, optional): the label of the plot. Defaults to ``None``.
        ax (matplotlib.pyplot.axes): the axes to use for the plot.
        plot_kwargs (dict, optional): keywords arguments passed to ``ax.plot(...)``.
        scatter_success_kwargs (dict, optional): keywords arguments passed to ``ax.scatter(...)``
            for the successful evaluations.
        scatter_failure_kwargs (dict, optional): keywords arguments passed to ``ax.scatter(...)``
            for the failed evaluations.

    Returns:
        (matplotlib.pyplot.figure, matplotlib.pyplot.axes): the figure and axes of the plot.
    """
    assert mode in ["min", "max"]
    assert x_units in ["evaluations", "seconds"]

    # Manage default values
    if plot_kwargs is None:
        plot_kwargs = dict()
    _plot_kwargs = dict(
        color="skyblue",
        label="Trajectory" if label is None else label,
    )
    _plot_kwargs.update(plot_kwargs)

    if scatter_success_kwargs is None:
        scatter_success_kwargs = dict()
    _scatter_success_kwargs = dict(
        marker="o",
        s=10,
        c="skyblue" if show_failures else None,
        label="Successes" if label is None else None,
    )
    _scatter_success_kwargs.update(scatter_success_kwargs)

    if scatter_failure_kwargs is None:
        scatter_failure_kwargs = dict()
    _scatter_failure_kwargs = dict(
        marker="v",
        color="red",
        label="Failures" if label is None else None,
    )
    _scatter_failure_kwargs.update(scatter_failure_kwargs)

    if x_units == "evaluations":
        x = np.arange(len(results))
        x_label = "Evaluations"
        results = results.sort_values(by=["m:timestamp_gather"], ascending=True)
    else:
        if "m:timestamp_end" in results.columns:
            results = results.sort_values(by=["m:timestamp_end"], ascending=True)
            x = results["m:timestamp_end"].to_numpy()
        else:
            results = results.sort_values(by=["m:timestamp_gather"], ascending=True)
            x = results["m:timestamp_gather"].to_numpy()
        x_label = "Time (sec.)"

    if results[column].dtype != np.float64:
        mask_failed = np.where(results[column].str.startswith("F"))[0]
        mask_success = np.where(~results[column].str.startswith("F"))[0]
        x_success, x_failed = x[mask_success], x[mask_failed]
        y_success = results[column][mask_success].astype(float)
    else:
        x_success = x
        x_failed = np.array([])
        y_success = results[column]

    if mode == "min":
        y_success = -y_success
        y_plot = y_success.cummin()
    else:
        y_plot = y_success.cummax()

    y_min, y_max = y_success.min(), y_success.max()
    y_min = y_min - 0.05 * (y_max - y_min)
    y_max = y_max - 0.05 * (y_max - y_min)

    fig = plt.gcf()
    if fig is None:
        fig = plt.figure()

    if ax is None:
        ax = fig.gca()

    ax.plot(x_success, y_plot, **_plot_kwargs)

    ax.scatter(
        x_success,
        y_success,
        **_scatter_success_kwargs,
    )

    if show_failures and len(x_failed) > 0:
        ax.scatter(
            x_failed,
            np.full_like(x_failed, y_min),
            **_scatter_failure_kwargs,
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel("Objective")
    ax.legend()
    ax.grid(True)
    ax.set_xlim(x.min(), x.max())

    return fig, ax


def compile_worker_activity(results, profile_type="submit/gather"):
    """Compute the number of active workers.

    Args:
        results (pd.DataFrame): the results of a Hyperparameter Search.
        profile_type (str, optional): the type of profile to build. It can be `"submit/gather"` or
            `"start/end"`. Defaults to "submit/gather".

    Returns:
        timestamps, n_jobs_active: a list of timestamps and a list of the number of active jobs at
            each timestamp.
    """
    if profile_type == "submit/gather":
        key_start, key_end = "m:timestamp_submit", "m:timestamp_gather"
    elif profile_type == "start/end":
        key_start, key_end = "m:timestamp_start", "m:timestamp_end"
    else:
        raise ValueError(
            f"Unknown profile_type='{profile_type}' it should be one of "
            "['submit/gather', 'start/end']"
        )

    if key_start not in results.columns or key_end not in results.columns:
        raise ValueError(f"Columns '{key_start}' and '{key_end}' are not present in the DataFrame.")

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
        num_workers (int, optional): the number of workers. If passed the normalized utilization
            will be shown (/num_workers). Otherwise, the raw number of active workers is shown.
            Defaults to ``None``.
        profile_type (str, optional): the type of profile to build. It can be `"submit/gather"`
            or `"start/end"`. Defaults to "submit/gather".
        ax (matplotlib.pyplot.axes): the axes to use for the plot.
        kwargs (dict): other keywords arguments passed to the ``ax.step(...)``.

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

    ax.step(x, y, where="post", **plot_kwargs)

    ax.set_xlabel("Time (sec.)")
    if num_workers:
        ax.set_ylabel("Utilization")
    else:
        ax.set_ylabel("Active Workers")
    ax.grid(True)
    ax.set_xlim(x.min(), x.max())

    return fig, ax


def add_colorbar_px(paxfig, data, cmap="viridis", colorbar_kwargs={}):
    # Attribute
    paxfig._pax_colorbar = True

    # Local vars
    n_lines = len(paxfig.axes[0].lines)
    n_axes = len(paxfig.axes)

    vmin = data.min()
    vmax = data.max()
    # Change line colors
    for i in range(n_lines):
        # Get value
        # Get color
        # color = paxfig._get_color_gradient(scale_val, 0, 1, cmap)
        color = (data[i] - vmin) / (vmax - vmin)
        # Assign color to line
        for j in paxfig.axes[:-1]:
            j.lines[i].set_color(cmap(color))

    # Create blank axis for colorbar
    width_ratios = paxfig.axes[0].get_gridspec().get_width_ratios()
    new_n_axes = n_axes + 1
    new_width_ratios = width_ratios + [0.5]
    gs = paxfig.add_gridspec(1, new_n_axes, width_ratios=new_width_ratios)
    ax_colorbar = paxfig.add_subplot(gs[0, n_axes])

    # Create colorbar
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=data.min(), vmax=data.max()), cmap=cmap)
    cbar = paxfig.colorbar(sm, orientation="vertical", ax=ax_colorbar, **colorbar_kwargs)
    cbar.locator = MaxNLocator(integer=True)
    cbar.update_ticks()

    main_ax_pos = paxfig.axes[-1].get_position()
    cbar_ax = cbar.ax
    cbar_ax.set_position(
        [
            main_ax_pos.x1 + 0.03,  # X position (left)
            main_ax_pos.y0,  # Y position (bottom)
            0.02,  # Width of colorbar
            main_ax_pos.height,  # Height of colorbar
        ]
    )

    # Figure formatting
    for i in range(n_axes):
        paxfig.axes[i].set_subplotspec(gs[0:1, i : i + 1])
    ax_colorbar.set_axis_off()
    return paxfig


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
        rank_mode (str, optional): mode of ranking. Defaults to ``"min"``.
        highlight (bool, optional): whether to highlight the best solutions. Defaults to ``True``.
        constant_predictor (float, optional): value to compare the objective to. Defaults to
            ``0.035726056``.

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

    results = results[cols]

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["xtick.labelsize"] = 12  # Set the X-axis tick label font size
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["font.size"] = 16
    plt.rcParams["axes.linewidth"] = 1
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.titleweight"] = "bold"

    cmap = LinearSegmentedColormap.from_list(
        "my_gradient",
        (
            # Edit this gradient at https://eltos.github.io/gradient/#0:00D0FF-33.3:0000FF-66.7:FF0000-100:FFD800
            (0.000, (0.000, 0.816, 1.000)),
            (0.333, (0.000, 0.000, 1.000)),
            (0.667, (1.000, 0.000, 0.000)),
            (1.000, (1.000, 0.847, 0.000)),
        ),
    )

    # Add labels
    def convertCols(cols):
        names = []
        for c in cols:
            c = c.replace("p:", "")
            c = c.replace("_", " ")
            c = c.split()
            c = [word.capitalize() for word in c]
            names.append(" ".join(c))
        return names

    cols = convertCols(cols)

    if highlight:
        # Split data
        paxfig = pax_parallel(n_axes=len(cols))
        if rank_mode == "min":
            df_highlight = results[results[objective_column] < constant_predictor]
            df_highlight["rank"] = rank(df_highlight[objective_column])
            df_grey = results[results[objective_column] >= constant_predictor]
        elif rank_mode == "max":
            df_highlight = results[results[objective_column] > constant_predictor]
            df_highlight["rank"] = rank(-df_highlight[objective_column])
            df_grey = results[results[objective_column] <= constant_predictor]

        paxfig.plot(df_highlight.to_numpy()[:, :-1], line_kwargs={"alpha": 0.5})

        paxfig.set_labels(cols)

        paxfig = add_colorbar_px(
            paxfig=paxfig,
            data=df_highlight["rank"].to_numpy(),
            # data=df_highlight['rank'].to_numpy(),
            cmap=cmap,
            colorbar_kwargs={"label": "Rank"},
        )

        try:
            paxfig.plot(
                df_grey.to_numpy(),
                line_kwargs={"alpha": 0.1, "color": "grey", "zorder": 0},
            )
        except:  # noqa
            pass

    else:
        paxfig = pax_parallel(n_axes=len(cols))
        paxfig.plot(results[cols].to_numpy(), line_kwargs={"alpha": 0.5})

        # Add colorbar
        color_col = len(cols) - 1
        paxfig.add_colorbar(ax_idx=color_col, cmap=cmap, colorbar_kwargs={"label": "Rank"})

    fig = plt.gcf()
    ax = fig.gca()
    # h = 4
    # fig.set_size_inches((len(paxfig.axes) / 2.0) * h, h)
    # plt.show()

    return fig, ax
