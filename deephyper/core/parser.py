import argparse
import inspect
from inspect import signature


def add_arguments_from_signature(parser, obj, prefix="", exclude=[]):
    """Add arguments to parser base on obj default keyword parameters.

    :meta private:

    Args:
        parser (ArgumentParser)): the argument parser to which we want to add arguments.
        obj (type): the class from which we want to extract default parameters for the constructor.
    """
    sig = signature(obj)
    prefix = f"{prefix}-" if len(prefix) > 0 else ""
    added_arguments = []

    for p_name, p in sig.parameters.items():
        if not (p_name in exclude):

            if p.kind == inspect._POSITIONAL_OR_KEYWORD:

                arg_format = f"--{prefix}{p_name.replace('_', '-')}"
                arg_kwargs = {"help": ""}

                # check type int
                if not (p.annotation is inspect._empty):
                    arg_kwargs["type"] = p.annotation
                    arg_kwargs["help"] += f"Type[{p.annotation.__name__}]. "

                # check default value
                if p.default is not inspect._empty:
                    arg_kwargs["default"] = p.default
                    arg_kwargs["help"] += f"Defaults to '{str(p.default)}'. "
                else:
                    arg_kwargs["required"] = True

                parser.add_argument(
                    arg_format,
                    **arg_kwargs,
                )

            added_arguments.append(p_name)

    return added_arguments


def str2bool(v):
    """
    :meta private:
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
