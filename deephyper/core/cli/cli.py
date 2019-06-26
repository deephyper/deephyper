
import argparse
import os
import sys

from deephyper.core.cli import nas_init


def create_parser():
    parser = argparse.ArgumentParser(
        description='DeepHyper command line.')

    subparsers = parser.add_subparsers(help='Menus.')

    # nas-init
    nas_init.add_subparser(subparsers)

    return parser


def main():
    parser = create_parser()

    args = parser.parse_args()

    try:
        args.func(**vars(args))
    except AttributeError:
        parser.print_help()
