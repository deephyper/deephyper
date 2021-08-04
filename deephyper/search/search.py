import argparse
import logging
import signal

from deephyper.core.exceptions import SearchTerminationError
from deephyper.evaluator.evaluate import EVALUATORS


class Search:
    def __init__(
        self, problem, evaluator, random_state=None, log_dir=".", verbose=0, **kwargs
    ):
        self._problem = problem
        self._evaluator = evaluator
        self._random_state = random_state
        self._log_dir = log_dir
        self._verbose = verbose

    def terminate(self):
        """Terminate the search.

        Raises:
            SearchTerminationError: raised when the search is terminated with SIGALARM
        """
        logging.info("Search is being stopped!")
        raise SearchTerminationError

    def _set_timeout(self, timeout=None):
        def handler(signum, frame):
            self.terminate()

        signal.signal(signal.SIGALRM, handler)

        if type(timeout) is int:
            signal.alarm(timeout)

    def search(self, max_evals=-1, timeout=None):

        self._set_timeout(timeout)

        try:
            self._search(max_evals, timeout)
        except SearchTerminationError:
            self._evaluator.dump_evals()

    def _search(self, max_evals, timeout):
        raise NotImplementedError

    @classmethod
    def get_parser(cls):
        parser = argparse.ArgumentParser(conflict_handler="resolve")
        parser.add_argument(
            "--problem",
            required=True,
            help="Python package path to the problem instance you want to use for the search. For example, 'dh_project.polynome2.problem.Problem'. It can also be the PATH to a Python script containing an attribute named 'Problem'.",
        )
        parser.add_argument(
            "--run",
            required=True,
            help="Python package path to the run function you want to use for the evaluator. For example, 'dh_project.polynome2.model_run.run'. It can also be the PATH to a Python script containing an attribute named 'run'.",
        )
        parser.add_argument(
            "--evaluator",
            default="ray",
            choices=list(EVALUATORS.keys()),
            help="The evaluator used for the search.",
        )
        parser.add_argument(
            "--random-state", default=None, help="The random state of the search."
        )
        parser.add_argument(
            "--log-dir",
            default=".",
            help="PATH to the folder where the logs are written.",
        )
        parser.add_argument(
            "-v", "--verbose", type=bool, default=None, nargs="?", const=True
        )
        parser.add_argument(
            "--max-evals",
            type=int,
            default=-1,
            help="The maximum number of evaluations. If negative it will not stop based on the number of evaluations.",
        )
        parser.add_argument(
            "--timeout",
            type=int,
            default=None,
            help="The time budget (in seconds) of the search.",
        )

        #! Evaluator arguments
        # global
        parser.add_argument(
            "--num-workers",
            default=None,
            type=int,
            help="Number of parallel workers for the search. By default, it is being automatically computed depending on the chosen evaluator. If fixed then the default number of workers is override by this value.",
        )
        parser.add_argument(
            "--cache-key",
            default="uuid",
            choices=["uuid", "to_dict"],
            help="Cache policy.",
        )

        # ray evaluator
        parser.add_argument(
            "--ray-address",
            default="",
            help="This parameter is mandatory when using evaluator==ray. It reference the 'IP:PORT' redis address for the RAY-driver to connect on the RAY-head.",
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
            help="Valid only if 'evaluator == ray'.",
        )
        parser.add_argument(
            "--driver-num-gpus",
            type=int,
            default=None,
            help="Valid only when 'evaluator == ray'.",
        )
        parser.add_argument(
            "--num-cpus-per-task",
            type=int,
            default=1,
            help="Valid only if 'evaluator == ray'.",
        )
        parser.add_argument(
            "--num-gpus-per-task",
            type=int,
            default=None,
            help="Valid only when 'evaluator == ray'.",
        )

        # balsam evaluator
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

        return parser