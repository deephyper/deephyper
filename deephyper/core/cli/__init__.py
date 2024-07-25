# For the documentation
from . import _cli, _hps

commands = [_cli, _hps]

__doc__ = ""

for c in commands:
    __doc__ += c.__doc__
