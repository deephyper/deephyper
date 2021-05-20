import argparse
import json
import time
import logging
import os
from pprint import pformat

import numpy as np
from deephyper.evaluator.encoder import Encoder
from deephyper.evaluator.evaluate import Evaluator
from deephyper.search import util

logger = logging.getLogger(__name__)


class Namespace:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self.__dict__[k] = v


class Search:
    """Abstract representation of a black box optimization search.

    A search comprises 3 main objects: a problem, a run function and an evaluator:
        The `problem` class defines the optimization problem, providing details like the search domain.  (You can find many kind of problems in `deephyper.benchmark`)
        The `run` function executes the black box function/model and returns the objective value which is to be optimized.
        The `evaluator` abstracts the run time environment (local, supercomputer...etc) in which run functions are executed.

    Args:
        problem (str): Module path to the Problem instance you want to use for the search (e.g. deephyper.benchmark.hps.polynome2.Problem).
        run (str): Module path to the run function you want to use for the search (e.g. deephyper.benchmark.hps.polynome2.run).
        evaluator (str): value in ['balsam', 'ray',  'subprocess', 'processPool', 'threadPool'].
        max_evals (int): the maximum number of evaluations to run. The exact behavior related to this parameter can vary in different search.
    """

    def __init__(
        self,
        problem: str,
        run: str,
        evaluator: str,
        max_evals: int = 1000000,
        seed: int = None,
        num_nodes_master: int = 1,
        num_workers: int = None,
        log_dir: int = None,
        **kwargs,
    ):
        kwargs["problem"] = problem
        kwargs["run"] = run
        kwargs["evaluator"] = evaluator
        kwargs["max_evals"] = max_evals  # * For retro compatibility
        kwargs["seed"] = seed

        # Loading problem instance and run function
        self.problem = util.generic_loader(problem, "Problem")
        if self.problem.seed == None:
            self.problem.seed = seed
        else:
            kwargs["seed"] = self.problem.seed
        self.run_func = util.generic_loader(run, "run")

        notice = f"Maximizing the return value of function: {run}"
        logger.info(notice)
        util.banner(notice)

        self.evaluator = Evaluator.create(
            self.run_func,
            method=evaluator,
            num_nodes_master=num_nodes_master,
            num_workers=num_workers,
            **kwargs,
        )
        self.num_workers = self.evaluator.num_workers
        self.max_evals = max_evals
        self.log_dir = os.getcwd() if log_dir is None else log_dir

        # set the random seed
        np.random.seed(self.problem.seed)

        logger.info(f"Options: " + pformat(kwargs, indent=4))
        logger.info(f"Created {evaluator} evaluator")
        logger.info(f"Evaluator: num_workers is {self.num_workers}")
        self.write_init_infos()

    def write_init_infos(self):
        infos = {}
        infos["start_timestamp"] = time.time()
        infos["num_workers"] = self.num_workers
        infos["max_evals"] = self.max_evals
        infos["problem"] = self.problem.space

        path = os.path.join(self.log_dir, "init_infos.json")
        with open(path, "w") as f:
            json.dump(infos, f, cls=Encoder, indent=2)


    def main(self):
        raise NotImplementedError

    @classmethod
    def get_parser(cls, parser=None) -> argparse.ArgumentParser:
        """Return the fully extended parser.

        Returns:
            ArgumentParser: the fully extended parser.
        """
        base_parser = cls._base_parser(parser)
        parser = cls._extend_parser(base_parser)
        return parser

    @classmethod
    def parse_args(cls, arg_str=None) -> None:
        parser = cls.get_parser()
        if arg_str is not None:
            return parser.parse_args(arg_str)
        else:
            return parser.parse_args()

    @staticmethod
    def _extend_parser(base_parser) -> argparse.ArgumentParser:
        raise NotImplementedError

    @staticmethod
    def _base_parser(parser=None) -> argparse.ArgumentParser:
        if parser is None:
            parser = argparse.ArgumentParser(conflict_handler="resolve")
        parser.add_argument(
            "--problem",
            help="Module path to the Problem instance you want to use for the search.",
        )
        parser.add_argument(
            "--run",
            help="Module path to the run function you want to use for the search.",
        )
        parser.add_argument(
            "--backend", default="tensorflow", help="Keras backend module name"
        )
        parser.add_argument(
            "--max-evals", type=int, default=1000000, help="maximum number of evaluations"
        )
        parser.add_argument(
            "--eval-timeout-minutes",
            type=int,
            default=4096,
            help="Kill evals that take longer than this",
        )
        parser.add_argument(
            "--evaluator",
            default="ray",
            choices=[
                "balsam",
                "ray",
                "rayhorovod",
                "subprocess",
                "processPool",
                "threadPool",
            ],
            help="The evaluator is an object used to evaluate models.",
        )
        parser.add_argument(
            "--ray-address",
            default="",
            help='This parameter is mandatory when using evaluator==ray. It reference the "IP:PORT" redis address for the RAY-driver to connect on the RAY-head.',
        )
        parser.add_argument(
            "--ray-password",
            default="5241590000000000",
            help="",
        )
        parser.add_argument(
            "--driver-num-cpus",
            type=int,
            default=None,
            help="Valid only if evaluator==ray",
        )
        parser.add_argument(
            "--driver-num-gpus",
            type=int,
            default=None,
            help="Valid only when evaluator==ray",
        )
        parser.add_argument(
            "--num-cpus-per-task",
            type=int,
            default=1,
            help="Valid only if evaluator==ray",
        )
        parser.add_argument(
            "--num-gpus-per-task",
            type=int,
            default=None,
            help="Valid only when evaluator==ray",
        )
        parser.add_argument("--seed", default=None, help="Random seed used.")
        parser.add_argument(
            "--cache-key",
            default="uuid",
            choices=["uuid", "to_dict"],
            help="Cache policy.",
        )
        parser.add_argument(
            "--num-ranks-per-node",
            default=1,
            type=int,
            help="Number of ranks per nodes for each evaluation. Only valid if evaluator==balsam and balsam job-mode is 'mpi'.",
        )
        parser.add_argument(
            "--num-evals-per-node",
            default=1,
            type=int,
            help="Number of evaluations performed on each node. Only valid if evaluator==balsam and balsam job-mode is 'serial'.",
        )
        parser.add_argument(
            "--num-nodes-per-eval",
            default=1,
            type=int,
            help="Number of nodes used for each evaluation. This Parameter is usefull when using data-parallelism or model-parallism with evaluator==balsam and balsam job-mode is 'mpi'.",
        )
        parser.add_argument(
            "--num-threads-per-rank",
            default=64,
            type=int,
            help="Number of threads per MPI rank. Only valid if evaluator==balsam and balsam job-mode is 'mpi'.",
        )
        parser.add_argument(
            "--num-threads-per-node",
            default=None,
            type=int,
            help="Number of threads per node. Only valid if evaluator==balsam and balsam job-mode is 'mpi'.",
        )
        parser.add_argument(
            "--num-workers",
            default=None,
            type=int,
            help="Number of parallel workers for the search. By default, it is being automatically computed depending on the chosen evaluator. If fixed then the default number of workers is override by this value.",
        )
        parser.add_argument(
            "--log-dir",
            default=None,
            type=str,
            help="Path of the directory where to store information about the run.",
        )
        return parser
