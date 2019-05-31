"""Asynchronous Model-Based Search.

Arguments of AMBS :
* ``learner``

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

logger = util.conf_logger('deephyper.search.hps.ambs')

SERVICE_PERIOD = 2          # Delay (seconds) between main loop iterations
CHECKPOINT_INTERVAL = 1    # How many jobs to complete between optimizer checkpoints
EXIT_FLAG = False


def on_exit(signum, stack):
    global EXIT_FLAG
    EXIT_FLAG = True


class AMBS(Search):
    def __init__(self, problem, run, evaluator, **kwargs):
        super().__init__(problem, run, evaluator, **kwargs)
        logger.info("Initializing AMBS")
        self.optimizer = Optimizer(self.problem, self.num_workers, self.args)

    @staticmethod
    def _extend_parser(parser):
        parser.add_argument('--learner',
                            default='RF',
                            choices=["RF", "ET", "GBRT", "DUMMY", "GP"],
                            help='type of learner (surrogate model)'
                            )
        parser.add_argument('--liar-strategy',
                            default="cl_max",
                            choices=["cl_min", "cl_mean", "cl_max"],
                            help='Constant liar strategy'
                            )
        parser.add_argument('--acq-func',
                            default="gp_hedge",
                            choices=["LCB", "EI", "PI", "gp_hedge"],
                            help='Acquisition function type'
                            )
        return parser

    def main(self):
        timer = util.DelayTimer(max_minutes=None, period=SERVICE_PERIOD)
        chkpoint_counter = 0
        num_evals = 0

        init_XX = self.problem.starting_point
        logger.info(
            f"The space has {len(init_XX)} starting points which will be used for the first iteration of the search.")

        # ! if we have more starting points than available workers
        if len(init_XX) > self.num_workers:
            batch = init_XX[:self.num_workers]
            init_XX = init_XX[self.num_workers:]
            self.evaluator.add_eval_batch(batch)
        else:
            logger.info(
                f"Starting points exhausted, generating {self.num_workers-len(init_XX)} other initial points...")
            self.evaluator.add_eval_batch(init_XX)
            for batch in range(self.num_workers-len(init_XX)):
                self.evaluator.add_eval_batch(batch)
            init_XX = []

        # * MAIN LOOP
        for elapsed_str in timer:
            logger.info(f"Elapsed time: {elapsed_str}")
            results = list(self.evaluator.get_finished_evals())
            num_evals += len(results)
            chkpoint_counter += len(results)
            if EXIT_FLAG or num_evals >= self.args.max_evals:
                break
            if results:
                logger.info(
                    f"Refitting model with batch of {len(results)} evals")
                self.optimizer.tell(results)
                logger.info(
                    f"Drawing {len(results)} points with strategy {self.optimizer.strategy}")

                # ! 'ask' is written as a generator because asking for a large batch is
                # ! slow. We get better performance when ask is batched. The RF is
                # ! constantly re-fitting during the call to ask. So it becomes slow
                # ! when there are a large number of workers.
                if len(init_XX) == 0:  # * starting point list exhausted.
                    for batch in self.optimizer.ask(n_points=len(results)):
                        self.evaluator.add_eval_batch(batch)
                else:
                    if len(init_XX) > self.num_workers:
                        batch = init_XX[:self.num_workers]
                        init_XX = init_XX[self.num_workers:]
                        self.evaluator.add_eval_batch(batch)
                    else:
                        logger.info("Starting points exhausted!")
                        self.evaluator.add_eval_batch(init_XX)
                        for batch in range(self.num_workers-len(init_XX)):
                            self.evaluator.add_eval_batch(batch)
                        init_XX = []

            if chkpoint_counter >= CHECKPOINT_INTERVAL:
                self.evaluator.dump_evals()
                chkpoint_counter = 0

        logger.info('Hyperopt driver finishing')
        self.evaluator.dump_evals()


if __name__ == "__main__":
    args = AMBS.parse_args()
    search = AMBS(**vars(args))
    signal.signal(signal.SIGINT, on_exit)
    signal.signal(signal.SIGTERM, on_exit)
    search.main()
