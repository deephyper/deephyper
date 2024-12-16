import matplotlib as mpl


def figure_size(width, fraction=1):
    """Set figure dimensions to avoid scaling in LaTeX.

    Args:
        width (float): Document textwidth or columnwidth in pts.
        fraction (float, optional): Fraction of the width which you wish the figure to occupy.

    Returns:
        tuple: Dimensions of figure in inches.
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


def update_matplotlib_rc(width=252 * 1.8, fraction=1.0, fontsize=10):
    mpl.rcParams.update(
        {
            "figure.figsize": figure_size(width=width, fraction=fraction),
            "figure.facecolor": "white",
            "figure.edgecolor": "white",
            "savefig.dpi": 360,
            "figure.subplot.bottom": 0.5,
            # Use LaTeX to write all text
            "text.usetex": True,
            # "font.family": "serif",
            # Use 10pt font in plots, to match 10pt font in document
            "axes.labelsize": fontsize,
            "font.size": fontsize,
            # Make the legend/label fonts a little smaller
            "legend.fontsize": fontsize - 3,
            "xtick.labelsize": fontsize - 1,
            "ytick.labelsize": fontsize - 1,
            # tight layout,
            "figure.autolayout": True,
        }
    )
