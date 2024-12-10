"""
This analysis subpackage contains modules to analyze results returned by deephyper.
"""

from ._rank import rank
from ._matplotlib import figure_size, update_matplotlib_rc

__all__ = ["rank", "figure_size", "update_matplotlib_rc"]
