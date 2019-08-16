
import argparse
import os
import sys

from deephyper.core.cli import hps_init, hps, nas_init, nas


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

    return parser


def main():
    parser = create_parser()

    args = parser.parse_args()

    # try:
    args.func(**vars(args))
    # except AttributeError:
    #      parser.print_help()
