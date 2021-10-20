"""DeepHyper command line interface.

It can be used in the shell with:

.. code-block:: console

    $ deephyper --help

    usage: deephyper [-h] {hps,nas,new-problem,ray-cluster,ray-submit,start-project} ...

    DeepHyper command line.

    positional arguments:
    {hps,nas,new-problem,ray-cluster,ray-submit,start-project}
        hps                 Command line to run hyperparameter search.
        nas                 Command line to run neural architecture search.
        new-problem         Tool to init an hyper-parameter search package or a neural architecture search problem folder.
        start-project       Set up a new project folder for DeepHyper benchmarks

    optional arguments:
    -h, --help            show this help message and exit
"""
import argparse

from deephyper.core.cli import _hps, _nas, _new_problem, _start_project


def create_parser():
    """
    :meta private:
    """
    parser = argparse.ArgumentParser(description="DeepHyper command line.")

    subparsers = parser.add_subparsers()

    # hyper-parameter search
    _hps.add_subparser(subparsers)

    # neural architecture search cli
    _nas.add_subparser(subparsers)

    # new-problem
    _new_problem.add_subparser(subparsers)

    # start-project
    _start_project.add_subparser(subparsers)

    return parser


def main():
    """
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
