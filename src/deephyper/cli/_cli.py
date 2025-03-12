"""Command line interface.
=======================

It can be used in the shell with:

.. code-block:: console

    $ deephyper --help

    usage: deephyper [-h] {hpo} ...

    DeepHyper command line.

    positional arguments:
    {hpo}
        hpo                 Command line to run hyperparameter optimization.

    optional arguments:
    -h, --help            show this help message and exit

"""  # noqa: D205

import argparse

from . import _hpo


def create_parser():
    """Creates the deephyper CL parser.

    :meta private:
    """
    parser = argparse.ArgumentParser(description="DeepHyper command line.")

    subparsers = parser.add_subparsers()

    # hyper-parameter search
    _hpo.add_subparser(subparsers)

    return parser


def main():
    """Entry point of the CL.

    :meta private:
    """
    parser = create_parser()

    args = parser.parse_args()

    if hasattr(args, "func"):
        func = args.func
        kwargs = vars(args)
        kwargs.pop("func")
        func(**kwargs)
    else:
        parser.print_help()
