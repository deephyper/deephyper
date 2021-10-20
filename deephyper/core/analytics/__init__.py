# for the documentation
from . import _topk, _quick_plot

commands = [_topk, _quick_plot]

__doc__ = ""

for c in commands:
    __doc__ += c.__doc__