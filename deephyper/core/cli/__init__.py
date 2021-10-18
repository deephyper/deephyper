from deephyper.core.cli._cli import main

# for the documentation
from . import _cli, _hps, _nas, _new_problem, _ray_cluster, _ray_submit, _start_project

commands = [_cli, _hps, _nas, _new_problem, _ray_cluster, _ray_submit, _start_project]

__doc__ = ""

for c in commands:
    __doc__ += c.__doc__