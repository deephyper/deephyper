# for the documentation
from . import _cli, _hps, _nas, _new_problem, _start_project

commands = [_cli, _hps, _nas, _new_problem, _start_project]

__doc__ = ""

for c in commands:
    __doc__ += c.__doc__
