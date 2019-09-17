
import argparse
import os
import sys

from deephyper.core.cli import mpi4py_mock
from deephyper.core.cli import tensorflow_mock
from deephyper.core.cli import keras_mock
sys.modules['mpi4py'] = mpi4py_mock
from deephyper.core.cli import hps_init, hps, nas_init, nas, balsam_submit


def create_parser():
    parser = argparse.ArgumentParser(
        description='DeepHyper command line.')

    subparsers = parser.add_subparsers()

    # nas-init
    nas_init.add_subparser(subparsers)

    # neural architecture search cli
    nas.add_subparser(subparsers)

    # hps-init
    hps_init.add_subparser(subparsers)

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
