import matplotlib.pyplot as plt


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
