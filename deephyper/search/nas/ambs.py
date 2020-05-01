import json
import os
import signal

from deephyper.core.logs.logging import JsonMessage as jm
from deephyper.evaluator.evaluate import Encoder
from deephyper.search import util
from deephyper.search.nas import NeuralArchitectureSearch
from deephyper.search.nas.optimizer import Optimizer

dhlogger = util.conf_logger("deephyper.search.nas.ambs")

SERVICE_PERIOD = 2  # Delay (seconds) between main loop iterations
CHECKPOINT_INTERVAL = 1  # How many jobs to complete between optimizer checkpoints
EXIT_FLAG = False


def on_exit(signum, stack):
    global EXIT_FLAG
    EXIT_FLAG = True


class AMBNeuralArchitectureSearch(NeuralArchitectureSearch):
    """Asynchronous Model-Based Search.

    Args:
        problem (str): python attribute import of the ``NaProblem`` instance (e.g. ``mypackage.mymodule.myproblem``).
        run (str): python attribute import of the run function (e.g. ``mypackage.mymodule.myrunfunction``).
        evaluator (str): the name of the evaluator to use.
        surrogate_model (str, optional): Choices are ["RF", "ET", "GBRT", "DUMMY", "GP"]. ``RF`` is Random Forest, ``ET`` is Extra Trees, ``GBRT`` is Gradient Boosting Regression Trees, ``DUMMY`` is random, ``GP`` is Gaussian process. Defaults to "RF".
        liar_strategy (str, optional): ["cl_max", "cl_min", "cl_mean"]. Defaults to "cl_max".
        acq_func (str, optional): Acquisition function, choices are ["gp_hedge", "LCB", "EI", "PI"]. Defaults to "gp_hedge".
        n_jobs (int, optional): Number of parallel jobs to distribute the surrogate model (learner). Defaults to -1, means as many as the number of logical cores.
    """

    def __init__(
        self,
        problem,
        run,
        evaluator,
        surrogate_model="RF",
        liar_strategy="cl_max",
        acq_func="gp_hedge",
        n_jobs=-1,
        **kwargs,
    ):

        super().__init__(problem=problem, run=run, evaluator=evaluator, **kwargs)

        self.free_workers = self.evaluator.num_workers

        dhlogger.info(
            jm(
                type="start_infos",
                alg="ambs-nas",
                nworkers=self.free_workers,
                encoded_space=json.dumps(self.problem.space, cls=Encoder),
            )
        )

        dhlogger.info("Initializing AMBS")
        self.optimizer = Optimizer(
            self.problem,
            self.num_workers,
            surrogate_model=surrogate_model,
            liar_strategy=liar_strategy,
            acq_func=acq_func,
            n_jobs=n_jobs,
            **kwargs,
        )

    @staticmethod
    def _extend_parser(parser):
        NeuralArchitectureSearch._extend_parser(parser)
        parser.add_argument(
            "--surrogate-model",
            default="RF",
            choices=["RF", "ET", "GBRT", "DUMMY", "GP"],
            help="type of surrogate model (learner)",
        )
        parser.add_argument(
            "--liar-strategy",
            default="cl_max",
            choices=["cl_min", "cl_mean", "cl_max"],
            help="Constant liar strategy",
        )
        parser.add_argument(
            "--acq-func",
            default="gp_hedge",
            choices=["LCB", "EI", "PI", "gp_hedge"],
            help="Acquisition function type",
        )
        parser.add_argument(
            "--acq-kappa",
            type=float,
            default=1.96,
            help='Controls how much of the variance in the predicted values should be taken into account. If set to be very high, then we are favouring exploration over exploitation and vice versa. Used when the acquisition is "LCB".',
        )
        parser.add_argument(
            "--n-jobs",
            default=-1,
            type=int,
            help="Number of processes to use for surrogate model (learner).",
        )
        return parser

    def main(self):
        timer = util.DelayTimer(max_minutes=None, period=SERVICE_PERIOD)
        chkpoint_counter = 0
        num_evals = 0

        dhlogger.info(f"Generating {self.num_workers} initial points...")
        XX = self.optimizer.ask_initial(n_points=self.num_workers)
        self.evaluator.add_eval_batch(XX)

        # MAIN LOOP
        for elapsed_str in timer:
            dhlogger.info(f"Elapsed time: {elapsed_str}")
            results = list(self.evaluator.get_finished_evals())
            num_evals += len(results)
            chkpoint_counter += len(results)
            if EXIT_FLAG or num_evals >= self.max_evals:
                break
            if results:
                dhlogger.info(f"Refitting model with batch of {len(results)} evals")
                self.optimizer.tell(results)
                dhlogger.info(
                    f"Drawing {len(results)} points with strategy {self.optimizer.strategy}"
                )
                # ! 'ask' is written as a generator because asking for a large batch is
                # ! slow. We get better performance when ask is batched. The RF is
                # ! constantly re-fitting during the call to ask. So it becomes slow
                # ! when there are a large number of workers.
                for batch in self.optimizer.ask(n_points=len(results)):
                    self.evaluator.add_eval_batch(batch)
            if chkpoint_counter >= CHECKPOINT_INTERVAL:
                self.evaluator.dump_evals(saved_key="arch_seq")
                chkpoint_counter = 0

        dhlogger.info("Hyperopt driver finishing")
        self.evaluator.dump_evals(saved_key="arch_seq")


if __name__ == "__main__":
    args = AMBNeuralArchitectureSearch.parse_args()
    search = AMBNeuralArchitectureSearch(**vars(args))
    signal.signal(signal.SIGINT, on_exit)
    signal.signal(signal.SIGTERM, on_exit)
    search.main()
