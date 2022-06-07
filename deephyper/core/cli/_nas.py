"""
Neural Architecture Search
--------------------------

Command line to execute neural architecture search or joint hyperparameter and neural architecture search.

.. code-block:: bash

    $ deephyper nas regevo --help

    usage: deephyper nas regevo [-h] --problem PROBLEM --evaluator EVALUATOR [--random-state RANDOM_STATE] [--log-dir LOG_DIR] [--verbose VERBOSE] [--population-size POPULATION_SIZE] [--sample-size SAMPLE_SIZE]
                                [--max-evals MAX_EVALS] [--timeout TIMEOUT] --run-function RUN_FUNCTION [--num-workers NUM_WORKERS] [--callbacks CALLBACKS] [--ray-address RAY_ADDRESS]
                                [--ray-password RAY_PASSWORD] [--ray-num-cpus RAY_NUM_CPUS] [--ray-num-gpus RAY_NUM_GPUS] [--ray-num-cpus-per-task RAY_NUM_CPUS_PER_TASK]
                                [--ray-num-gpus-per-task RAY_NUM_GPUS_PER_TASK] [--ray-ray-kwargs RAY_RAY_KWARGS]

    optional arguments:
    -h, --help            show this help message and exit
    --problem PROBLEM
    --evaluator EVALUATOR
    --random-state RANDOM_STATE
                            Type[int]. Defaults to 'None'.
    --log-dir LOG_DIR     Type[str]. Defaults to '.'.
    --verbose VERBOSE     Type[int]. Defaults to '0'.
    --population-size POPULATION_SIZE
                            Type[int]. Defaults to '100'.
    --sample-size SAMPLE_SIZE
                            Type[int]. Defaults to '10'.
    --max-evals MAX_EVALS
                            Type[int]. Defaults to '-1' when an number of evaluations is not imposed.
    --timeout TIMEOUT     Type[int]. Number of seconds before killing the search. Defaults to 'None' when a time budget is not imposed.
    --run-function RUN_FUNCTION
    --num-workers NUM_WORKERS
                            Type[int]. Defaults to '1'.
    --callbacks CALLBACKS
                            Defaults to 'None'.
    --ray-address RAY_ADDRESS
                            Type[str]. Defaults to 'None'.
    --ray-password RAY_PASSWORD
                            Type[str]. Defaults to 'None'.
    --ray-num-cpus RAY_NUM_CPUS
                            Type[int]. Defaults to 'None'.
    --ray-num-gpus RAY_NUM_GPUS
                            Type[int]. Defaults to 'None'.
    --ray-num-cpus-per-task RAY_NUM_CPUS_PER_TASK
                            Type[float]. Defaults to '1'.
    --ray-num-gpus-per-task RAY_NUM_GPUS_PER_TASK
                            Type[float]. Defaults to 'None'.
    --ray-ray-kwargs RAY_RAY_KWARGS
                            Type[dict]. Defaults to '{}'.
"""
import argparse
import sys
import logging

from deephyper.core.parser import add_arguments_from_signature
from deephyper.evaluator import EVALUATORS, Evaluator
from deephyper.core.utils import load_attr

NAS_SEARCHES = {
    "random": "deephyper.search.nas._random.Random",
    "regevo": "deephyper.search.nas._regevo.RegularizedEvolution",
    "agebo": "deephyper.search.nas._agebo.AgEBO",
    "ambsmixed": "deephyper.search.nas._ambsmixed.AMBSMixed",
    "regevomixed": "deephyper.search.nas._regevomixed.RegularizedEvolutionMixed",
}


def build_parser_from(cls):
    """
    :meta private:
    """
    parser = argparse.ArgumentParser(conflict_handler="resolve")

    # add the arguments of a specific search
    add_arguments_from_signature(parser, cls)

    # add argument of Search.search interface
    parser.add_argument(
        "--max-evals",
        default=-1,
        type=int,
        help="Type[int]. Defaults to '-1' when an number of evaluations is not imposed.",
    )
    parser.add_argument(
        "--timeout",
        default=None,
        type=int,
        help="Type[int]. Number of seconds before killing the search. Defaults to 'None' when a time budget is not imposed.",
    )

    # add arguments for evaluators
    evaluator_added_arguments = add_arguments_from_signature(parser, Evaluator)

    for eval_name, eval_cls in EVALUATORS.items():
        try:
            eval_cls = load_attr(f"deephyper.evaluator.{eval_cls}")
            add_arguments_from_signature(
                parser, eval_cls, prefix=eval_name, exclude=evaluator_added_arguments
            )
        except ModuleNotFoundError as e:  # some evaluators are optional
            pass

    return parser


def add_subparser(parsers):
    """
    :meta private:
    """
    parser_name = "nas"

    parser = parsers.add_parser(
        parser_name, help="Command line to run neural architecture search."
    )

    subparsers = parser.add_subparsers()

    for name, module_attr in NAS_SEARCHES.items():
        search_cls = load_attr(module_attr)

        search_parser = build_parser_from(search_cls)

        subparser = subparsers.add_parser(
            name=name, parents=[search_parser], conflict_handler="resolve"
        )

        subparser.set_defaults(func=main)


def main(**kwargs):
    """
    :meta private:
    """

    sys.path.insert(0, ".")

    if kwargs["verbose"]:
        logging.basicConfig(filename="deephyper.log", level=logging.INFO)

    search_name = sys.argv[2]

    # load search class
    logging.info(f"Loading the search '{search_name}'...")
    search_cls = load_attr(NAS_SEARCHES[search_name])

    # load problem
    logging.info("Loading the problem...")
    problem = load_attr(kwargs.pop("problem"))

    # load run function
    logging.info("Loading the run-function...")
    run_function = load_attr(kwargs.pop("run_function"))

    # filter arguments from evaluator class signature
    logging.info("Loading the evaluator...")
    evaluator_method = kwargs.pop("evaluator")
    base_arguments = ["num_workers", "callbacks"]
    evaluator_kwargs = {k: kwargs.pop(k) for k in base_arguments}

    for method in EVALUATORS.keys():
        evaluator_method_kwargs = {
            k[len(evaluator_method) + 1 :]: kwargs.pop(k)
            for k in kwargs.copy()
            if method in k
        }
        if method == evaluator_method:
            evaluator_kwargs = {**evaluator_kwargs, **evaluator_method_kwargs}

    # create evaluator
    logging.info(
        f"Evaluator(method={evaluator_method}, method_kwargs={evaluator_kwargs}"
    )
    evaluator = Evaluator.create(
        run_function, method=evaluator_method, method_kwargs=evaluator_kwargs
    )
    logging.info(f"Evaluator has {evaluator.num_workers} workers available.")

    # filter arguments from search class signature
    # remove keys in evaluator_kwargs
    kwargs = {k: v for k, v in kwargs.items() if k not in evaluator_kwargs}
    max_evals = kwargs.pop("max_evals")
    timeout = kwargs.pop("timeout")

    # TODO: How about checkpointing and transfer learning?

    # execute the search
    # remaining kwargs are for the search
    logging.info("Starting the search")
    search = search_cls(problem, evaluator, **kwargs)

    search.search(max_evals=max_evals, timeout=timeout)
