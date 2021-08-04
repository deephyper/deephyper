import argparse
import os
import sys
import signal

from deephyper.search.util import load_attr

HPS_SEARCHES = {
    "ambs": "deephyper.search.hps.AMBS",
}


def add_subparser(parsers):
    parser_name = "hps"

    parser = parsers.add_parser(
        parser_name, help="Command line to run hyperparameter search."
    )

    subparsers = parser.add_subparsers()

    for name, module_attr in HPS_SEARCHES.items():
        search_cls = load_attr(module_attr)

        search_parser = search_cls.get_parser()
        subparser = subparsers.add_parser(
            name=name, parents=[search_parser], conflict_handler="resolve"
        )

        subparser.set_defaults(func=main)


def main(**kwargs):
    print(kwargs)

    search_name = sys.argv[2]
    search_cls = load_attr(HPS_SEARCHES[search_name])

    # load problem
    ...

    # load run function
    ...

    # filter arguments from evaluator class signature

    # filter arguments from search class signature
    ...

    #TODO: How about checkpointing and transfer learning?

    # execute the search
    ...
