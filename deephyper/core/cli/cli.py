"""DeepHyper command line interface.

It can be used in the shell with:

.. code-block:: console

    $ deephyper --help
    usage: deephyper [-h] {start-project,new-problem,nas,hps,ray-submit,ray-cluster} ...

    DeepHyper command line.

    positional arguments:
    {start-project,new-problem,nas,hps,ray-submit,ray-cluster}
        start-project       Set up a new project folder for DeepHyper benchmarks
        new-problem         Tool to init an hyper-parameter search package or a neural architecture search problem folder.
        nas                 Command line to run neural architecture search.
        hps                 Command line to run hyperparameter search.
        ray-submit          Create and submit an HPS or NAS job directly via Ray.
        ray-cluster         Manipulate a Ray cluster.

    optional arguments:
    -h, --help            show this help message and exit
"""
import argparse

from deephyper.core.cli import start_project, new_problem
from deephyper.core.cli import hps, nas, ray_submit, ray_cluster


def create_parser():
    """
    :meta private:
    """
    parser = argparse.ArgumentParser(
        description='DeepHyper command line.')

    subparsers = parser.add_subparsers()

    # start-project
    start_project.add_subparser(subparsers)

    # new-problem
    new_problem.add_subparser(subparsers)

    # neural architecture search cli
    nas.add_subparser(subparsers)

    # hyper-parameter search
    hps.add_subparser(subparsers)

    # balsam-submit
    # balsam_submit.add_subparser(subparsers)

    # ray-submit
    ray_submit.add_subparser(subparsers)

    # ray-cluster
    ray_cluster.add_subparser(subparsers)

    return parser


def main():
    """
    :meta private:
    """
    parser = create_parser()

    args = parser.parse_args()

    if hasattr(args, 'func'):
        func = args.func
        kwargs = vars(args)
        kwargs.pop("func")
        func(**kwargs)
    else:
        parser.print_help()
