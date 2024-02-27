import matplotlib.pyplot as plt
import paxplot  # reuqires paxplot. pip install paxplot
from deephyper.analysis import rank
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator


def parallelCoordPlot(
    df,
    cols,
    objective_name,
    rank_mode="min",
    highlight=True,
    constant_predictor=0.035726056,
):
    """function to create a parallel coordinate plot of the data
    Args:
        df (pd.DataFrame): dataframe containing the data
        cols (list): list of columns to include in the plot, it has to inlcude the objective column.
        objective_name (str): name of the objective column
        rank_mode (str, optional): mode of ranking. Defaults to "min".
        highlight (bool, optional): whether to highlight the best solutions. Defaults to True.
        constant_predictor (float, optional): value to compare the objective to. Defaults to 0.035726056.
    Returns:
        fig: figure of the parallel coordinate plot
    """

    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(str)

    df = df[cols].copy()

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
        paxfig = paxplot.pax_parallel(n_axes=len(cols))
        if rank_mode == "min":
            df_highlight = df[df[objective_name] < constant_predictor]
            df_highlight["rank"] = rank(df_highlight[objective_name])
            df_grey = df[df[objective_name] >= constant_predictor]
        elif rank_mode == "max":
            df_highlight = df[df[objective_name] > constant_predictor]
            df_highlight["rank"] = rank(-df_highlight[objective_name])
            df_grey = df[df[objective_name] <= constant_predictor]

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
        except:
            pass

    else:
        paxfig = paxplot.pax_parallel(n_axes=len(cols))
        paxfig.plot(df[cols].to_numpy(), line_kwargs={"alpha": 0.5})

        # Add colorbar
        color_col = len(cols) - 1
        paxfig.add_colorbar(
            ax_idx=color_col, cmap=cmap, colorbar_kwargs={"label": "Rank"}
        )
    fig = plt.gcf()
    h = 4
    fig.set_size_inches((len(paxfig.axes) / 2.0) * h, h)
    plt.show()

    return fig


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
    sm = plt.cm.ScalarMappable(
        norm=plt.Normalize(vmin=data.min(), vmax=data.max()), cmap=cmap
    )
    cbar = paxfig.colorbar(
        sm, orientation="vertical", ax=ax_colorbar, **colorbar_kwargs
    )
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
