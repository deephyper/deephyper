"""Analysis subpackage.

This subpackage contains tools to analyze results obtained with DeepHyper.
"""

from ._rank import rank
from ._matplotlib import figure_size, update_matplotlib_rc

__all__ = ["rank", "figure_size", "update_matplotlib_rc"]
