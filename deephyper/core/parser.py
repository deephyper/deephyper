import inspect
from inspect import signature

def add_arguments_from_signature(parser, obj):
    """Add arguments to parser base on obj default keyword parameters.

    Args:
        parser (ArgumentParser)): the argument parser to which we want to add arguments.
        obj (type): the class from which we want to extract default parameters for the constructor.
    """
    sig = signature(obj)

    for p_name, p in sig.parameters.items():
        if p.kind == inspect._POSITIONAL_OR_KEYWORD:
            if p.default is not inspect._empty:
                parser.add_argument(
                    f"--{p_name.replace('_', '-')}",
                    default=p.default,
                    help=f"Defaults to '{str(p.default)}'.")
