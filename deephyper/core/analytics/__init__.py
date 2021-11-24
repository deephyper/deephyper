# for the documentation
from . import _topk, _quick_plot, _dashboard

commands = [_topk, _quick_plot, _dashboard]

__doc__ = "Provides command lines tools to visualize results from DeepHyper.\n\n"

for c in commands:
    __doc__ += c.__doc__