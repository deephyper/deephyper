
import argparse
import os
import sys

from deephyper.core.cli import mpi4py_mock
sys.modules['mpi4py'] = mpi4py_mock
from deephyper.core.cli import start_project, new_problem
from deephyper.core.cli import hps, nas, balsam_submit


def create_parser():
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
    balsam_submit.add_subparser(subparsers)

    return parser


def main():
    parser = create_parser()

    args = parser.parse_args()

    if hasattr(args, 'func'):
        args.func(**vars(args))
    else:
        parser.print_help()
