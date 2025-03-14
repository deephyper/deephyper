"""Hyperparameter optimization.
----------------------------

Use the command line help option to get more information.

.. code-block:: bash

   $ deephyper hpo cbo --help

    usage: deephyper hpo cbo [-h] --problem PROBLEM --evaluator EVALUATOR [--random-state RANDOM_STATE] [--log-dir LOG_DIR] [--verbose VERBOSE] [--stopper STOPPER] [--surrogate-model SURROGATE_MODEL]
                            [--surrogate-model-kwargs SURROGATE_MODEL_KWARGS] [--acq-func ACQ_FUNC] [--acq-optimizer ACQ_OPTIMIZER] [--acq-optimizer-freq ACQ_OPTIMIZER_FREQ] [--kappa KAPPA] [--xi XI]
                            [--n-points N_POINTS] [--filter-duplicated FILTER_DUPLICATED] [--update-prior UPDATE_PRIOR] [--update-prior-quantile UPDATE_PRIOR_QUANTILE]
                            [--multi-point-strategy MULTI_POINT_STRATEGY] [--n-jobs N_JOBS] [--n-initial-points N_INITIAL_POINTS] [--initial-point-generator INITIAL_POINT_GENERATOR]
                            [--initial-points INITIAL_POINTS] [--filter-failures FILTER_FAILURES] [--max-failures MAX_FAILURES] [--moo-lower-bounds MOO_LOWER_BOUNDS]
                            [--moo-scalarization-strategy MOO_SCALARIZATION_STRATEGY] [--moo-scalarization-weight MOO_SCALARIZATION_WEIGHT] [--scheduler SCHEDULER] [--objective-scaler OBJECTIVE_SCALER]
                            [--max-evals MAX_EVALS] [--timeout TIMEOUT] --run-function RUN_FUNCTION [--num-workers NUM_WORKERS] [--callbacks CALLBACKS] [--run-function-kwargs RUN_FUNCTION_KWARGS]
                            [--storage STORAGE] [--search-id SEARCH_ID] [--mpicomm-comm MPICOMM_COMM] [--mpicomm-root MPICOMM_ROOT] [--ray-address RAY_ADDRESS] [--ray-password RAY_PASSWORD]
                            [--ray-num-cpus RAY_NUM_CPUS] [--ray-num-gpus RAY_NUM_GPUS] [--ray-include-dashboard RAY_INCLUDE_DASHBOARD] [--ray-num-cpus-per-task RAY_NUM_CPUS_PER_TASK]
                            [--ray-num-gpus-per-task RAY_NUM_GPUS_PER_TASK] [--ray-ray-kwargs RAY_RAY_KWARGS]

    options:
    -h, --help            show this help message and exit
    --problem PROBLEM
    --evaluator EVALUATOR
    --random-state RANDOM_STATE
                            Type[int]. Defaults to 'None'.
    --log-dir LOG_DIR     Type[str]. Defaults to '.'.
    --verbose VERBOSE     Type[int]. Defaults to '0'.
    --stopper STOPPER     Defaults to 'None'.
    --surrogate-model SURROGATE_MODEL
                            Defaults to 'ET'.
    --surrogate-model-kwargs SURROGATE_MODEL_KWARGS
                            Type[dict]. Defaults to 'None'.
    --acq-func ACQ_FUNC   Type[str]. Defaults to 'UCBd'.
    --acq-optimizer ACQ_OPTIMIZER
                            Type[str]. Defaults to 'auto'.
    --acq-optimizer-freq ACQ_OPTIMIZER_FREQ
                            Type[int]. Defaults to '10'.
    --kappa KAPPA         Type[float]. Defaults to '1.96'.
    --xi XI               Type[float]. Defaults to '0.001'.
    --n-points N_POINTS   Type[int]. Defaults to '10000'.
    --filter-duplicated FILTER_DUPLICATED
                            Type[bool]. Defaults to 'True'.
    --update-prior UPDATE_PRIOR
                            Type[bool]. Defaults to 'False'.
    --update-prior-quantile UPDATE_PRIOR_QUANTILE
                            Type[float]. Defaults to '0.1'.
    --multi-point-strategy MULTI_POINT_STRATEGY
                            Type[str]. Defaults to 'cl_max'.
    --n-jobs N_JOBS       Type[int]. Defaults to '1'.
    --n-initial-points N_INITIAL_POINTS
                            Type[int]. Defaults to '10'.
    --initial-point-generator INITIAL_POINT_GENERATOR
                            Type[str]. Defaults to 'random'.
    --initial-points INITIAL_POINTS
                            Defaults to 'None'.
    --filter-failures FILTER_FAILURES
                            Type[str]. Defaults to 'min'.
    --max-failures MAX_FAILURES
                            Type[int]. Defaults to '100'.
    --moo-lower-bounds MOO_LOWER_BOUNDS
                            Defaults to 'None'.
    --moo-scalarization-strategy MOO_SCALARIZATION_STRATEGY
                            Type[str]. Defaults to 'Chebyshev'.
    --moo-scalarization-weight MOO_SCALARIZATION_WEIGHT
                            Defaults to 'None'.
    --scheduler SCHEDULER
                            Defaults to 'None'.
    --objective-scaler OBJECTIVE_SCALER
                            Defaults to 'auto'.
    --max-evals MAX_EVALS
                            Defaults to '-1' when number of evaluations is not imposed.
    --timeout TIMEOUT     Number of seconds before killing search, defaults to 'None'.
    --run-function RUN_FUNCTION
    --num-workers NUM_WORKERS
                            Type[int]. Defaults to '1'.
    --callbacks CALLBACKS
                            Type[list]. Defaults to 'None'.
    --run-function-kwargs RUN_FUNCTION_KWARGS
                            Type[dict]. Defaults to 'None'.
    --storage STORAGE     Type[Storage]. Defaults to 'None'.
    --search-id SEARCH_ID
                            Type[Hashable]. Defaults to 'None'.
    --mpicomm-comm MPICOMM_COMM
                            Defaults to 'None'.
    --mpicomm-root MPICOMM_ROOT
                            Defaults to '0'.
    --ray-address RAY_ADDRESS
                            Type[str]. Defaults to 'None'.
    --ray-password RAY_PASSWORD
                            Type[str]. Defaults to 'None'.
    --ray-num-cpus RAY_NUM_CPUS
                            Type[int]. Defaults to 'None'.
    --ray-num-gpus RAY_NUM_GPUS
                            Type[int]. Defaults to 'None'.
    --ray-include-dashboard RAY_INCLUDE_DASHBOARD
                            Type[bool]. Defaults to 'False'.
    --ray-num-cpus-per-task RAY_NUM_CPUS_PER_TASK
                            Type[float]. Defaults to '1'.
    --ray-num-gpus-per-task RAY_NUM_GPUS_PER_TASK
                            Type[float]. Defaults to 'None'.
    --ray-ray-kwargs RAY_RAY_KWARGS
                            Type[dict]. Defaults to 'None'.

"""  # noqa: D205, E501

import argparse
import logging
import sys

from deephyper.cli.utils import add_arguments_from_signature, load_attr
from deephyper.evaluator import EVALUATORS, Evaluator

HPS_SEARCHES = {
    "cbo": "deephyper.hpo.CBO",
    "random": "deephyper.hpo.RandomSearch",
    "regevo": "deephyper.hpo.RegularizedEvolution",
    "eds": "deephyper.hpo.ExperimentalDesignSearch",
}


def build_parser_from(cls):
    """Build the parser automatically from the classes.

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
        help="Defaults to '-1' when number of evaluations is not imposed.",
    )
    parser.add_argument(
        "--timeout",
        default=None,
        type=int,
        help="Number of seconds before killing search, defaults to 'None'.",
    )

    # add arguments for evaluators
    evaluator_added_arguments = add_arguments_from_signature(parser, Evaluator)

    for eval_name, eval_cls in EVALUATORS.items():
        try:
            eval_cls = load_attr(f"deephyper.evaluator.{eval_cls}")
            add_arguments_from_signature(
                parser, eval_cls, prefix=eval_name, exclude=evaluator_added_arguments
            )
        except ModuleNotFoundError:  # some evaluators are optional
            pass

    return parser


def add_subparser(parsers):
    """Definition of the parser of the command.

    :meta private:
    """
    parser_name = "hpo"

    parser = parsers.add_parser(
        parser_name, help="Command line to run hyperparameter optimization."
    )

    subparsers = parser.add_subparsers()

    for name, module_attr in HPS_SEARCHES.items():
        search_cls = load_attr(module_attr)

        search_parser = build_parser_from(search_cls)

        subparser = subparsers.add_parser(
            name=name, parents=[search_parser], conflict_handler="resolve"
        )

        subparser.set_defaults(func=main)


def main(**kwargs):
    """Entry point of the command.

    :meta private:
    """
    sys.path.insert(0, ".")

    if kwargs["verbose"]:
        logging.basicConfig(filename="deephyper.log", level=logging.INFO)

    search_name = sys.argv[2]

    # load search class
    logging.info(f"Loading the search '{search_name}'...")
    search_cls = load_attr(HPS_SEARCHES[search_name])

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

    # remove the arguments from unused evaluator
    for method in EVALUATORS.keys():
        evaluator_method_kwargs = {
            k[len(evaluator_method) + 1 :]: kwargs.pop(k) for k in kwargs.copy() if method in k
        }
        if method == evaluator_method:
            evaluator_kwargs = {**evaluator_kwargs, **evaluator_method_kwargs}

    # create evaluator
    logging.info(f"Evaluator(method={evaluator_method}, method_kwargs={evaluator_kwargs}")
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
    logging.info(f"Evaluator has {evaluator.num_workers} workers available.")
    search = search_cls(problem, evaluator, **kwargs)

    search.search(max_evals=max_evals, timeout=timeout)
