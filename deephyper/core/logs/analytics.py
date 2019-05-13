import argparse
import sys

from deephyper.core.logs import parsing, json
from deephyper.core.plot import single, multi


def create_parser():
    parser = argparse.ArgumentParser(
        description='Run some analytics for deephyper.')

    subparsers = parser.add_subparsers(help='Kind of analytics.')

    mapping = dict()

    # parsing
    name, func = parsing.add_subparser(subparsers)
    mapping[name] = func

    # json
    name, func = json.add_subparser(subparsers)
    mapping[name] = func

    # plots single
    name, func = single.add_subparser(subparsers)
    mapping[name] = func

    # plots multi
    name, func = multi.add_subparser(subparsers)
    mapping[name] = func

    return parser, mapping


def main():
    parser, mapping = create_parser()

    args = parser.parse_args()

    mapping[sys.argv[1]](**vars(args))
