"""Core paxplot functions from: https://github.com/kravitsjacob/paxplot"""

import copy
import functools
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.figure import Figure


class PaxFigure(Figure):
    _safe_inherited_functions = ["savefig", "set_size_inches", "draw", "show"]

    def __init__(self, *args, data=[], **kwargs):
        """Paxplot extension of Matplot Figure"""
        # Setup
        super().__init__(*args, **kwargs)
        self._show_unsafe_warning = True

        # Paxplot attributes
        self._pax_data = []
        self._pax_data_scale = []
        self._pax_lims = []
        self._pax_ticks = []
        self._pax_ticks_scale = []
        self._pax_ticks_labels = []
        self._pax_custom_lims = []
        self._pax_custom_ticks = []
        self._pax_colorbar = False

    def _scale_vals(self, data, lower=None, upper=None):
        """Scale `data` between lower and upper

        Args:
        data : array-like
            Data to be scalled
        lower : numeric, optional
            Lower value for scaling, by default None
        upper : numeric, optional
            Upper value for scaling, by default None

        Returns:
        array-like
            Scaled data
        """
        # Convert to numpy
        data = np.array(data)

        if lower is None and upper is None:
            lower = data.min()
            upper = data.max()

        # Scale data
        data_scale = (data - lower) / (upper - lower)

        return data_scale

    def _get_color_gradient(self, val, lower, upper, colormap):
        """Get color gradient values for the `val`

        Args:
        val : float
            value to get color for scaling
        lower : float
            Lower value
        upper : float
            Upper value for scaling
        colormap : str
            Matplotlib colormap to use for coloring

        Returns:
        color: str
            string color code
        """
        color = mpl.colors.rgb2hex(
            cm.get_cmap(colormap)(self._scale_vals(val, lower, upper))
        )
        return color

    def _update_plot_lines(self, ax_idx):
        """Update plotted lines based on scaled data (_pax_data_scale)

        Args:
        ax_idx : int
            Axis index to update line data
        """
        if ax_idx == 0:  # First axis
            for i, line in enumerate(self.axes[ax_idx].lines):
                # Replace left y value
                y_l_scaled = self._pax_data_scale[i, ax_idx]
                line.set_ydata([y_l_scaled, line.get_ydata()[1]])
        elif ax_idx == self._pax_data.shape[1] - 1:  # Last axis
            for i, line in enumerate(self.axes[ax_idx - 1].lines):
                # Replace right y value
                y_r_scaled = self._pax_data_scale[i, ax_idx]
                line.set_ydata([line.get_ydata()[0], y_r_scaled])
        else:  # Middle axes
            for i, line in enumerate(self.axes[ax_idx].lines):
                # Replace left y value
                y_l_scaled = self._pax_data_scale[i, ax_idx]
                line.set_ydata([y_l_scaled, line.get_ydata()[1]])
            for i, line in enumerate(self.axes[ax_idx - 1].lines):
                # Replace right y value
                y_r_scaled = self._pax_data_scale[i, ax_idx]
                line.set_ydata([line.get_ydata()[0], y_r_scaled])

    def _update_plot_ticks(self, ax_idx):
        """Update ticks based on tick labels (_pax_ticks_labels) and scaled tick
        location data (_pax_ticks_scale)

        Args:
        ax_idx : _type_
            _description_

        Raises:
        ------
        ValueError
            _description_
        """
        self.axes[ax_idx].set_yticks(ticks=self._pax_ticks_scale[ax_idx])
        try:
            self.axes[ax_idx].set_yticklabels(labels=self._pax_ticks_labels[ax_idx])
        except ValueError:
            raise ValueError("Length of `labels` must be same as length of `ticks`")

        # Set bounds on axis (always between 0 and 1)
        self.axes[ax_idx].set_ylim([0.0, 1.0])

    def _default_format(self):
        """Set the default format of a Paxplot Figure"""
        # Set attributes
        def_vals = [[0, 1]] * len(self.axes)
        def_bools = [False] * len(self.axes)
        self._pax_lims = copy.deepcopy(def_vals)
        self._pax_ticks = copy.deepcopy(def_vals)
        self._pax_ticks_scale = copy.deepcopy(def_vals)
        self._pax_ticks_labels = copy.deepcopy(def_vals)
        self._pax_custom_lims = copy.deepcopy(def_bools)
        self._pax_custom_ticks = copy.deepcopy(def_bools)
        self._pax_invert = copy.deepcopy(def_bools)

        # Remove space between plots
        subplots_adjust_args = {"wspace": 0.0, "hspace": 0.0}
        self.subplots_adjust(**subplots_adjust_args)

        for ax in self.axes:
            # Remove axes frame
            ax.spines["top"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # Set limits
            ax.set_ylim([0, 1])
            ax.set_xlim([0, 1])

            # Set x ticks
            ax.set_xticks([0], [" "])
            ax.tick_params(axis="x", length=0.0, pad=10)

            # Set y ticks
            ax.set_yticks([0, 1])

        # Adjust ticks on last axis
        self.axes[-1].yaxis.tick_right()

    def _default_lim(self, ax_idx):
        """Set the default limits for an axis. Default limits are between the
        minimum and maximum values.

        Args:
        ax_idx : int
            Index of matplotlib axes
        """
        # Set attibutes
        self._pax_custom_lims[ax_idx] = False

        # Column statistics
        col = self._pax_data[:, ax_idx]
        minimum = min(col)
        maximum = max(col)

        # Set limits
        self._set_lim(ax_idx=ax_idx, bottom=minimum, top=maximum)

    def _default_ticks(self, ax_idx):
        """Set the default ticks for an axis. Default ticks are six labels
        between the current limits of the axis.

        Args:
        ax_idx : int
            Index of matplotlib axes
        """
        # Set attibutes
        self._pax_custom_ticks[ax_idx] = False

        # Set limits
        n_ticks = 6
        precision = 2
        bottom = self._pax_lims[ax_idx][0]
        top = self._pax_lims[ax_idx][1]
        ticks = np.linspace(bottom, top, num=n_ticks + 1)
        labels = ticks.round(precision)
        self._set_ticks(ax_idx=ax_idx, ticks=ticks, labels=labels)

    def _convert_string_data(self, data: list):
        """Convert string input data to numerical data

        Args:
        data : list
            Data to be plotted from `plot`

        Returns:
        data : list
            Converted `data`
        """
        for col_i in range(len(data[0])):
            # Extract column
            column = [row[col_i] for row in data]

            if type(column[0]) is str:
                # Unique values
                strings = list(dict.fromkeys(column))  # Preserves order
                numbers = list(range(len(strings)))
                numbers = self._scale_vals(
                    numbers,
                )
                strings.reverse()

                # Translation of strings to numbers for tick position
                translate_dict = dict(zip(strings, numbers))
                column_translated = [translate_dict.get(item, item) for item in column]
                for row_idx, row in enumerate(data):
                    row[col_i] = column_translated[row_idx]

                # Set ticks
                self.set_ticks(ax_idx=col_i, ticks=numbers, labels=strings)

        return np.array(data)

    def plot(self, data: list, line_kwargs={}):
        """Plot the supplied data

        Args:
        data : array-like
            Data to be plotted

        line_kwargs: dict
            Keyword arguments for lines corresponding to data
        """
        # Initial Checking
        if len(data[0]) < len(self.axes) and not self._pax_colorbar:
            warnings.warn(
                "Supplied data has fewer columns than figure. Figure created "
                "with empty column(s)",
                Warning,
            )
        elif len(data[0]) > len(self.axes):
            raise ValueError(
                "Supplied data has more columns than figure. Please recreate "
                "paxfigure with appropriate n_axes"
            )

        # Convert input data to numpy
        data_input = np.array(data)

        # Check if conversion needed
        if not np.issubdtype(data_input.dtype.type, np.number):
            data_input = self._convert_string_data(data)

        # Update data attributes
        if len(self._pax_data) == 0:
            self._pax_data = data_input
        else:
            self._pax_data = np.vstack([self._pax_data, data_input])

        # Scale input data based on current limits
        data_input_scale = data_input.copy().astype(np.single)
        for col_idx, col in enumerate(data_input.T):
            data_input_scale[:, col_idx] = self._scale_vals(
                data=col,
                lower=self._pax_lims[col_idx][0],
                upper=self._pax_lims[col_idx][1],
            )

        # Update scaled data attributes
        if len(self._pax_data_scale) == 0:
            self._pax_data_scale = data_input_scale
        else:
            self._pax_data_scale = np.vstack([self._pax_data_scale, data_input_scale])

        # Add scaled input data to plot
        for ax_idx, ax in enumerate(self.axes[:-1]):
            ax.plot(data_input_scale[:, ax_idx : ax_idx + 2].T, **line_kwargs)

        # Limits
        for ax_idx in range(self._pax_data.shape[1]):
            if self._pax_custom_lims[ax_idx]:  # Respect custom limits
                self._set_lim(
                    ax_idx=ax_idx,
                    bottom=self._pax_lims[ax_idx][0],
                    top=self._pax_lims[ax_idx][1],
                )
            else:  # Default limits of data
                self._default_lim(ax_idx=ax_idx)

        # Respect custom ticks
        for ax_idx in range(self._pax_data.shape[1]):
            if self._pax_custom_ticks[ax_idx]:  # Respect custom ticks
                self._set_ticks(
                    ax_idx=ax_idx,
                    ticks=self._pax_ticks[ax_idx],
                    labels=self._pax_ticks_labels[ax_idx],
                )

    def set_lim(self, ax_idx: int, bottom: float, top: float):
        """Set custom limits on axis

        Args:
        ax_idx : int
            Index of matplotlib axes
        bottom : numeric
            Lower limit
        top : numeric
            Upper limit
        """
        # Set attibutes
        self._pax_custom_lims[ax_idx] = True

        # Set ticks
        self._set_lim(ax_idx=ax_idx, bottom=bottom, top=top)

    def _set_lim(self, ax_idx: int, bottom: float, top: float):
        """Private function to set custom limits on axis

        Args:
        ax_idx : int
            Index of matplotlib axes
        bottom : numeric
            Lower limit
        top : numeric
            Upper limit
        """
        # Check bottom top values
        try:
            if bottom == top:
                bottom = bottom - 1.0
                top = top + 1.0
        except TypeError:
            raise TypeError(
                f"Both `bottom` and `top` must be numeric values. Currently "
                f"`bottom` is of type {type(bottom)} and `top` is of type"
                f"{type(top)}"
            )

        # Checking if data is plotted
        try:
            self._pax_data[:, ax_idx]
        except TypeError:
            raise AttributeError(
                "Paxplot does not support set_lim if no data has been" "plotted"
            )

        # Set attribute data
        self._pax_lims[ax_idx] = [bottom, top]

        # Scale data
        col = self._pax_data[:, ax_idx]
        self._pax_data_scale[:, ax_idx] = self._scale_vals(
            col, lower=bottom, upper=top
        ).astype(np.single)

        # Update plot of scaled data
        self._update_plot_lines(ax_idx)

        # Ticks
        if self._pax_custom_ticks[ax_idx]:  # Preserve custom ticks
            self._set_ticks(
                ax_idx=ax_idx,
                ticks=self._pax_ticks[ax_idx],
                labels=self._pax_ticks_labels[ax_idx],
            )
        else:  # Default ticks
            self._default_ticks(ax_idx=ax_idx)

    def set_ticks(self, ax_idx: int, ticks: list, labels=None):
        """Set the axis tick locations and optionally labels.

        Args:
        ax_idx : int
            Index of matplotlib axes
        ticks : list of floats
            List of tick locations.
        labels : list of str, optional
            List of tick labels. If not set, the labels show the data value.
        """
        # Set attibutes
        self._pax_custom_ticks[ax_idx] = True

        # Set ticks
        self._set_ticks(ax_idx=ax_idx, ticks=ticks, labels=labels)

    def _set_ticks(self, ax_idx: int, ticks: list, labels=None):
        """Private function to set the axis tick locations and optionally labels.

        Args:
        ax_idx : int
            Index of matplotlib axes
        ticks : list of floats
            List of tick locations.
        labels : list of str, optional
            List of tick labels. If not set, the labels show the data value.
        """
        # Tick tests ('ask permission' mindset as nested try/except gets nasty)
        try:
            ticks + [1]
        except TypeError:
            raise TypeError(f"`ticks` must be array-like not type {type(ticks)}")
        try:
            min(ticks)
        except TypeError:
            raise TypeError(
                "All entries in `ticks` must be numeric. To set string ticks,"
                " use the `labels` argument"
            )

        # Retrieve matplotlib axes
        try:
            ax = self.axes[ax_idx]  # noqa
        except IndexError:
            raise IndexError(
                "You are trying to set the limits of axis with index "
                f"{ax_idx}. However, axis index only goes up to "
                f"{len(self.axes)-1}."
            )
        except TypeError:
            raise TypeError(f"Type of `ax_idx` must be integer not {type(ax_idx)}")

        # Set tick attibutes
        self._pax_ticks[ax_idx] = ticks

        # Scale tick based on current limits
        lim_bottom = self._pax_lims[ax_idx][0]
        lim_top = self._pax_lims[ax_idx][1]
        self._pax_ticks_scale[ax_idx] = self._scale_vals(
            ticks, lower=lim_bottom, upper=lim_top
        )

        # Tick labels
        if labels is None:
            labels = ticks.copy()
        self._pax_ticks_labels[ax_idx] = labels

        # Update ticks on plots
        self._update_plot_ticks(ax_idx)

        # Check if limits need updating
        lim_min = min(self._pax_lims[ax_idx])
        lim_max = max(self._pax_lims[ax_idx])
        if ticks[0] < lim_min or ticks[-1] > lim_max:
            bottom = min(np.append(ticks, lim_min))
            top = max(np.append(ticks, lim_max))
            self._set_lim(ax_idx=ax_idx, bottom=bottom, top=top)

    def set_even_ticks(
        self, ax_idx: int, n_ticks=6, minimum=None, maximum=None, precision=2
    ):
        """Set evenly spaced axis ticks between minimum and maximum value. If
        no minimum and maximum values are specified, the limits of the
        underlying plotted data are assumed.

        Args:
        ax_idx : int
            Index of matplotlib axes
        n_ticks : int
            Number of ticks
        minimum : numeric
            minimum value for ticks
        maximum : numeric
            maximum value for ticks
        precision : int
            number of decimal points for tick labels
        """
        # Set custom tick attributes
        self._pax_custom_ticks[ax_idx] = True

        # Set automatic min and maximum
        if minimum is None and maximum is None:
            minimum = self._pax_data[:, ax_idx].min()
            maximum = self._pax_data[:, ax_idx].max()

        # Minimum/maximum check
        if minimum > maximum:
            raise ValueError("Value for `minimum` cannot be greater than `maximum`")

        # Retrieve matplotlib axes
        try:
            self.axes[ax_idx]
        except IndexError:
            raise IndexError(
                f"You are trying to set the limits of axis with index "
                f"{ax_idx}. However, axis index only goes up to "
                f"{len(self.axes)-1}."
            )
        except TypeError:
            raise TypeError(f"Type of `ax_idx` must be integer not {type(ax_idx)}")

        # Generate ticks
        try:
            ticks = np.linspace(minimum, maximum, num=n_ticks + 1)
        except TypeError:
            raise TypeError(f"Type of `n_ticks` must be integer not {type(n_ticks)}")
        labels = ticks.round(precision)

        # Set ticks
        self._set_ticks(ax_idx=ax_idx, ticks=ticks, labels=labels)

    def set_label(self, ax_idx: int, label: str):
        """Set the label for the axis

        Args:
        ax_idx : int
            Index of matplotlib axes
        label : str
            The label text
        """
        try:
            ax = self.axes[ax_idx]
        except IndexError:
            raise IndexError(
                f"You are trying to set the limits of axis with index "
                f"{ax_idx}. However, axis index only goes up to "
                f"{len(self.axes)-1}."
            )
        except TypeError:
            raise TypeError(f"Type of `ax_idx` must be integer not {type(ax_idx)}")

        ax.set_xticks(ticks=[0.0])
        ax.set_xticklabels([label])

    def set_labels(self, labels: list):
        """Set labels for all axes. A wrapper for set_label

        Args:
        labels : list
            Labels for each axis. Must be same length as number of axes.
        """
        # Checking length
        if len(self._pax_data[0]) != len(labels):
            raise IndexError("Length of `labels` must equal number of axes")

        # Set labels
        for i, label in enumerate(labels):
            self.set_label(i, label)

    def invert_axis(self, ax_idx: int):
        """Invert axis

        Args:
        ax_idx : int
            Index of matplotlib axes
        """
        # Local vars
        try:
            ax = self.axes[ax_idx]  # noqa
        except IndexError:
            raise IndexError(
                f"You are trying to set the limits of axis with index "
                f"{ax_idx}. However, axis index only goes up to "
                f"{len(self.axes)-1}."
            )
        except TypeError:
            raise TypeError(f"Type of `ax_idx` must be integer not {type(ax_idx)}")

        # Checking if data is plotted
        try:
            self._pax_data[:, ax_idx]
        except TypeError:
            raise AttributeError(
                "Paxplot does not support invert_axis if no data has been" "plotted"
            )

        # Set attribute
        self._pax_invert[ax_idx] = True
        self._set_lim(
            ax_idx=ax_idx,
            bottom=self._pax_lims[ax_idx][1],
            top=self._pax_lims[ax_idx][0],
        )

    def add_legend(self, labels=[]):
        """Create a legend for a specified figure

        Args:
        labels : list
            List of data labels
        """
        # Check if too many labels supplied
        if len(labels) > len(self.axes[0].lines):
            warnings.warn(
                "More labels supplied than data. Some labels are unused.", Warning
            )

        if len(labels) > 0:
            try:
                for ax in self.axes:
                    for i, line in enumerate(ax.lines):
                        line.set_label(labels[i])
            except IndexError:
                raise IndexError(
                    f"Incorrect number of labels specified. You have supplied "
                    f"{len(labels)} labels, but {len(ax.lines)} were expected"
                )

        # Create blank axis for legend
        n_axes = len(self.axes)
        width_ratios = self.axes[0].get_gridspec().get_width_ratios()
        new_n_axes = n_axes + 1
        new_width_ratios = width_ratios + [1.0]
        gs = self.add_gridspec(1, new_n_axes, width_ratios=new_width_ratios)
        ax_legend = self.add_subplot(gs[0, n_axes])

        # Create legend
        lines = self.axes[0].lines
        labels = [i.get_label() for i in lines]
        ax_legend.legend(lines, labels, loc="center right")

        # Figure formatting
        for i in range(n_axes):
            self.axes[i].set_subplotspec(gs[0:1, i : i + 1])
        ax_legend.set_axis_off()

    def add_colorbar(self, ax_idx: int, cmap="viridis", colorbar_kwargs={}):
        """Add colorbar to paxfigure

        Args:
        ax : int
            axes index
        data : array-like
            Data to be plotted
        cmap : str
            Matplotlib colormap to use for coloring
        colorbar_kwargs : dict
            Matplotlib colorbar keyword arguments
        """
        # Attribute
        self._pax_colorbar = True

        # Local vars
        n_lines = len(self.axes[0].lines)
        n_axes = len(self.axes)

        # Testing
        try:
            self.axes[ax_idx]
        except IndexError:
            raise IndexError(
                f"You are trying to set the limits of axis with index "
                f"{ax_idx}. However, axis index only goes up to "
                f"{len(self.axes)-1}."
            )
        except TypeError:
            raise TypeError(f"Type of `ax_idx` must be integer not {type(ax_idx)}")

        # Change line colors
        for i in range(n_lines):
            # Get value
            if ax_idx < len(self.axes) - 1:
                scale_val = self.axes[ax_idx].lines[i].get_ydata()[0]
            else:
                scale_val = self.axes[ax_idx - 1].lines[i].get_ydata()[1]
            # Get color
            color = self._get_color_gradient(scale_val, 0, 1, cmap)
            # Assign color to line
            for j in self.axes[:-1]:
                j.lines[i].set_color(color)

        # Create blank axis for colorbar
        width_ratios = self.axes[0].get_gridspec().get_width_ratios()
        new_n_axes = n_axes + 1
        new_width_ratios = width_ratios + [0.5]
        gs = self.add_gridspec(1, new_n_axes, width_ratios=new_width_ratios)
        ax_colorbar = self.add_subplot(gs[0, n_axes])

        # Create colorbar
        sm = plt.cm.ScalarMappable(
            norm=plt.Normalize(
                vmin=self._pax_lims[ax_idx][0], vmax=self._pax_lims[ax_idx][1]
            ),
            cmap=cmap,
        )
        self.colorbar(sm, orientation="vertical", ax=ax_colorbar, **colorbar_kwargs)

        # Figure formatting
        for i in range(n_axes):
            self.axes[i].set_subplotspec(gs[0:1, i : i + 1])
        ax_colorbar.set_axis_off()


def add_unsafe_warning(func, fig):
    """Generate warning if not supported by Paxplot"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if fig._show_unsafe_warning:
            warnings.warn(
                f"The function you have called ({func.__name__}) is not "
                "officially supported by Paxplot, but it may still work. "
                "Report issues to "
                "https://github.com/kravitsjacob/paxplot/issues",
                Warning,
            )
        return func(*args, **kwargs)

    return wrapper


def disable_unsafe_warnings(func, fig):
    """Temporarily disables safety warnings for the duration of the function
    execution.

    This allows a known safe function needs to make safe calls to otherwise
    unsafe functions without throwing a warning.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        original_flag_value = fig._show_unsafe_warning
        fig._show_unsafe_warning = False
        result = func(*args, **kwargs)
        fig._show_unsafe_warning = original_flag_value
        return result

    return wrapper


def pax_parallel(n_axes: int):
    """Wrapper for paxplot analagous to the matplotlib.pyplot.subplots function

    Parameters
    ----------
    n_axes : int
        Number of axes to create

    Returns:
    -------
    fig : PaxFigure
        Paxplot figure class
    """
    # Check type of n_axes
    try:
        width_ratios = [1.0] * (n_axes - 1)
    except TypeError:
        raise TypeError(
            f"n_axes should by of type int. You have supplied a type" f"{type(n_axes)}"
        )

    # Create figure
    width_ratios.append(0.0)  # Last axis small
    fig, _ = plt.subplots(
        1,
        n_axes,
        sharey=False,
        gridspec_kw={"width_ratios": width_ratios},
        FigureClass=PaxFigure,
    )
    fig._default_format()

    pax_figure_functions = set(
        filter(
            lambda func_name: callable(getattr(PaxFigure, func_name)),
            vars(PaxFigure).keys(),
        )
    )

    unsafe_functions = set(
        filter(
            lambda func_name: (
                func_name not in PaxFigure._safe_inherited_functions
                and func_name not in pax_figure_functions
            ),
            dir(Figure),
        )
    )

    # Add unsafe function warnings
    for func_name in dir(PaxFigure):
        cond_1 = not func_name.startswith("__")
        cond_2 = not func_name.startswith("_")
        cond_3 = not func_name.startswith("get")
        cond_4 = callable(getattr(PaxFigure, func_name))
        if cond_1 and cond_2 and cond_3 and cond_4:
            func = getattr(fig, func_name)
            if func_name in unsafe_functions:
                func = add_unsafe_warning(func, fig)
            else:
                func = disable_unsafe_warnings(func, fig)
            setattr(fig, func_name, func)

    return fig
