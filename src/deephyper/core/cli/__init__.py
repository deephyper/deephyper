from . import _cli, _hpo  # noqa: D104

commands = [_cli, _hpo]

__doc__ = ""

for c in commands:
    __doc__ += c.__doc__
