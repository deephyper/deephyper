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


import math
import signal

import numpy as np

import skopt
from deephyper.search import Search, util
from deephyper.core.logs.logging import JsonMessage as jm

dhlogger = util.conf_logger("deephyper.search.hps.ambs")

SERVICE_PERIOD = 2  # Delay (seconds) between main loop iterations
CHECKPOINT_INTERVAL = 1  # How many jobs to complete between optimizer checkpoints
EXIT_FLAG = False


def on_exit(signum, stack):
    global EXIT_FLAG
    EXIT_FLAG = True


class AMBS(Search):
    def __init__(
        self,
        problem,
        run,
        evaluator,
        surrogate_model="RF",
        acq_func="LCB",
        kappa=1.96,
        xi=0.001,
        liar_strategy="cl_max",
        n_jobs=32,
        **kwargs,
    ):
        kwargs["cache_key"] = "to_dict"
        super().__init__(problem, run, evaluator, **kwargs)
        dhlogger.info("Initializing AMBS")
        dhlogger.info(f"kappa={kappa}, xi={xi}")

        self.n_initial_points = self.evaluator.num_workers

        self.opt = skopt.Optimizer(
            dimensions=self.problem.space,
            base_estimator=self.get_surrogate_model(surrogate_model, n_jobs),
            acq_func=acq_func,
            acq_optimizer="sampling",
            acq_func_kwargs={"xi": xi, "kappa": kappa},
            n_initial_points=self.n_initial_points,
            random_state=self.problem.seed,
            # model_queue_size=100,
        )

    @staticmethod
    def _extend_parser(parser):
        parser.add_argument(
            "--surrogate-model",
            default="RF",
            choices=["RF", "ET", "GBRT", "DUMMY", "GP"],
            help="Type of surrogate model (learner).",
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
            "--kappa",
            type=float,
            default=1.96,
            help='Controls how much of the variance in the predicted values should be taken into account. If set to be very high, then we are favouring exploration over exploitation and vice versa. Used when the acquisition is "LCB".',
        )

        parser.add_argument(
            "--xi",
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
        # timer = util.DelayTimer(max_minutes=None, period=SERVICE_PERIOD)
        # chkpoint_counter = 0

        num_evals_done = 0

        # Filling available nodes at start
        dhlogger.info(f"Generating {self.evaluator.num_workers} initial points...")
        self.evaluator.add_eval_batch(self.get_random_batch(size=self.n_initial_points))

        # Main loop
        while num_evals_done < self.max_evals:

            # Collecting finished evaluations
            new_results = list(self.evaluator.get_finished_evals())

            if len(new_results) > 0:
                stats = {"num_cache_used": self.evaluator.stats["num_cache_used"]}
                dhlogger.info(jm(type="env_stats", **stats))
                self.evaluator.dump_evals()

                num_received = len(new_results)
                num_evals_done += num_received

                # Transform configurations to list to fit optimizer
                opt_X = []
                opt_y = []
                for cfg, obj in new_results:
                    x = replace_nan(cfg.values())
                    opt_X.append(x)
                    opt_y.append(-obj)  #! maximizing

                self.opt.tell(opt_X, opt_y)  #! fit: costly
                new_X = self.opt.ask(n_points=len(new_results))

                new_batch = []
                for x in new_X:
                    new_cfg = self.to_dict(x)
                    new_batch.append(new_cfg)

                # submit_childs
                if len(new_results) > 0:
                    self.evaluator.add_eval_batch(new_batch)

    def get_surrogate_model(self, name: str, n_jobs: int = None):
        """Get a surrogate model from Scikit-Optimize.

        Args:
            name (str): name of the surrogate model.
            n_jobs (int): number of parallel processes to distribute the computation of the surrogate model.

        Raises:
            ValueError: when the name of the surrogate model is unknown.
        """
        accepted_names = ["RF", "ET", "GBRT", "GP", "DUMMY"]
        if not (name in accepted_names):
            raise ValueError(
                f"Unknown surrogate model {name}, please choose among {accepted_names}."
            )

        if name == "RF":
            surrogate = skopt.learning.RandomForestRegressor(n_jobs=n_jobs)
        elif name == "ET":
            surrogate = skopt.learning.ExtraTreesRegressor(n_jobs=n_jobs)
        elif name == "GBRT":
            surrogate = skopt.learning.GradientBoostingQuantileRegressor(n_jobs=n_jobs)
        else:  # for DUMMY and GP
            surrogate = name

        return surrogate

    def get_random_batch(self, size: int) -> list:
        batch = []
        points = self.opt.ask(n_points=size)
        for point in points:
            point_as_dict = self.to_dict(point)
            batch.append(point_as_dict)

        return batch

    def to_dict(self, x: list) -> dict:
        res = {}
        hps_names = self.problem.space.get_hyperparameter_names()
        for i in range(len(x)):
            res[hps_names[i]] = "nan" if isnan(x[i]) else x[i]
        return res


def isnan(x) -> bool:
    """Check if a value is NaN."""
    if isinstance(x, float):
        return math.isnan(x)
    elif isinstance(x, np.float64):
        return np.isnan(x)
    else:
        return False


def replace_nan(x):
    return [np.nan if x_i == "nan" else x_i for x_i in x]


if __name__ == "__main__":
    args = AMBS.parse_args()
    search = AMBS(**vars(args))
    signal.signal(signal.SIGINT, on_exit)
    signal.signal(signal.SIGTERM, on_exit)
    search.main()
