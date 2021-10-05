
import argparse

from deephyper.core.cli import start_project, new_problem
from deephyper.core.cli import hps, nas, ray_submit, ray_cluster


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
    # balsam_submit.add_subparser(subparsers)

    # ray-submit
    ray_submit.add_subparser(subparsers)

    # ray-cluster
    ray_cluster.add_subparser(subparsers)

    return parser


def main():
    parser = create_parser()

    args = parser.parse_args()

    if hasattr(args, 'func'):
        func = args.func
        kwargs = vars(args)
        kwargs.pop("func")
        func(**kwargs)
    else:
        parser.print_help()
