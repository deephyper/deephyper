# for the documentation
from . import _topk, _quick_plot

commands = [_topk, _quick_plot]

__doc__ = "Provides command lines tools to visualize results from DeepHyper.\n\n"

for c in commands:
    __doc__ += c.__doc__