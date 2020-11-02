import argparse
import os
import sys

from deephyper.core.logs import json, parsing
from deephyper.core.plot import hps, multi, post_train, single, quick_csv_plot


def create_parser():
    parser = argparse.ArgumentParser(description="Run some analytics for deephyper.")

    subparsers = parser.add_subparsers(help="Kind of analytics.")

    mapping = dict()

    modules = [
        parsing,  # parsing deephyper.log
        json,  # operation on json
        single,  # generate dh-analytics single notebook
        multi,  # generate dh-analytics multi notebook
        post_train,
        hps,  # generate notebook for hyperparamter optimization analytics
        quick_csv_plot,  # output quick plots
    ]

    for module in modules:
        name, func = module.add_subparser(subparsers)
        mapping[name] = func

    return parser, mapping


def main():
    parser, mapping = create_parser()

    args = parser.parse_args()

    mapping[sys.argv[1]](**vars(args))
