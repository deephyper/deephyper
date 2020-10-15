"""Asynchronous Model-Based Search.

Arguments of AMBS:

* ``surrogate-model``

    * ``RF`` : Random Forest (default)
    * ``ET`` : Extra Trees
    * ``GBRT`` : Gradient Boosting Regression Trees
    * ``DUMMY`` :
    * ``GP`` : Gaussian process

* ``liar-strategy``

    * ``cl_max`` : (default)
    * ``cl_min`` :
    * ``cl_mean`` :

* ``acq-func`` : Acquisition function

    * ``LCB`` :
    * ``EI`` :
    * ``PI`` :
    * ``gp_hedge`` : (default)
"""


import signal

from deephyper.search.hps.optimizer import Optimizer
from deephyper.search import Search
from deephyper.search import util

logger = util.conf_logger("deephyper.search.hps.ambs")

SERVICE_PERIOD = 2  # Delay (seconds) between main loop iterations
CHECKPOINT_INTERVAL = 1  # How many jobs to complete between optimizer checkpoints
EXIT_FLAG = False


def on_exit(signum, stack):
    global EXIT_FLAG
    EXIT_FLAG = True


class AMBS(Search):
    def __init__(self, problem, run, evaluator, **kwargs):
        kwargs["cache_key"] = "to_dict"
        super().__init__(problem, run, evaluator, **kwargs)
        logger.info("Initializing AMBS")
        kwargs.pop("num_workers")
        self.optimizer = Optimizer(self.problem, self.evaluator.num_workers, **kwargs)

    @staticmethod
    def _extend_parser(parser):
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
            "--acq-xi",
            type=float,
            default=0.01,
            help='Controls how much improvement one wants over the previous best values. If set to be very high, then we are favouring exploration over exploitation and vice versa. Used when the acquisition is "EI", "PI".',
        )

        parser.add_argument(
            "--n-jobs",
            type=int,
            default=1,
            help="number of cores to use for the 'surrogate model' (learner), if n_jobs=-1 then it will use all cores available.",
        )
        return parser

    def main(self):
        timer = util.DelayTimer(max_minutes=None, period=SERVICE_PERIOD)
        chkpoint_counter = 0
        num_evals = 0

        logger.info(f"Generating {self.evaluator.num_workers} initial points...")
        XX = self.optimizer.ask_initial(n_points=self.evaluator.num_workers)
        self.evaluator.add_eval_batch(XX)

        # MAIN LOOP
        for elapsed_str in timer:
            logger.info(f"Elapsed time: {elapsed_str}")
            results = list(self.evaluator.get_finished_evals())
            num_evals += len(results)
            chkpoint_counter += len(results)
            if EXIT_FLAG or num_evals >= self.max_evals:
                break
            if results:
                logger.info(f"Refitting model with batch of {len(results)} evals")
                self.optimizer.tell(results)
                logger.info(
                    f"Drawing {len(results)} points with strategy {self.optimizer.strategy}"
                )
                # ! 'ask' is written as a generator because asking for a large batch is
                # ! slow. We get better performance when ask is batched. The RF is
                # ! constantly re-fitting during the call to ask. So it becomes slow
                # ! when there are a large number of workers.
                for batch in self.optimizer.ask(n_points=len(results)):
                    self.evaluator.add_eval_batch(batch)
            if chkpoint_counter >= CHECKPOINT_INTERVAL:
                self.evaluator.dump_evals()
                chkpoint_counter = 0

        logger.info("Hyperopt driver finishing")
        self.evaluator.dump_evals()


if __name__ == "__main__":
    args = AMBS.parse_args()
    search = AMBS(**vars(args))
    signal.signal(signal.SIGINT, on_exit)
    signal.signal(signal.SIGTERM, on_exit)
    search.main()
