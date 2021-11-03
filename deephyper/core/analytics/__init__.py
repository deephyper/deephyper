# for the documentation
from . import _topk, _quick_plot, _dashboard

commands = [_topk, _quick_plot, _dashboard]

__doc__ = ""

for c in commands:
    __doc__ += c.__doc__