import math

import ConfigSpace as CS
import numpy as np
import skopt

from deephyper.problem import HpProblem
from deephyper.core.logs.logging import JsonMessage as jm
from deephyper.core.parser import add_arguments_from_signature
from deephyper.search import util
from deephyper.search.nas import NeuralArchitectureSearch
from skopt import Optimizer as SkOptimizer

dhlogger = util.conf_logger("deephyper.search.nas.ambsmixed")


class AMBSMixed(NeuralArchitectureSearch):
    def __init__(
        self,
        problem,
        run,
        evaluator,
        surrogate_model="RF",
        acq_func="LCB",
        kappa=1.96,
        xi=0.001,
        liar_strategy="cl_min",
        n_jobs=1,
        **kwargs,
    ):
        super().__init__(
            problem=problem,
            run=run,
            evaluator=evaluator,
            **kwargs,
        )

        self.n_jobs = int(n_jobs)  # parallelism of BO surrogate model estimator
        self.kappa = float(kappa)
        self.xi = float(xi)
        self.n_initial_points = self.evaluator.num_workers
        self.liar_strategy = liar_strategy

        # Setup
        na_search_space = self.problem.build_search_space()

        self.hp_space = self.problem._hp_space  #! hyperparameters
        self.hp_size = len(self.hp_space.space.get_hyperparameter_names())
        self.na_space = HpProblem(self.problem.seed)
        for i, vnode in enumerate(na_search_space.variable_nodes):
            self.na_space.add_hyperparameter(
                (0, vnode.num_ops - 1), name=f"vnode_{i:05d}"
            )

        self.space = CS.ConfigurationSpace(seed=self.problem.seed)
        self.space.add_configuration_space(
            prefix="1", configuration_space=self.hp_space.space
        )
        self.space.add_configuration_space(
            prefix="2", configuration_space=self.na_space.space
        )

        # Initialize opitmizer of hyperparameter space
        self.opt = SkOptimizer(
            dimensions=self.space,
            base_estimator=self.get_surrogate_model(surrogate_model, self.n_jobs),
            acq_func=acq_func,
            acq_optimizer="sampling",
            acq_func_kwargs={"xi": self.xi, "kappa": self.kappa},
            n_initial_points=self.n_initial_points,
        )

    @staticmethod
    def _extend_parser(parser):
        NeuralArchitectureSearch._extend_parser(parser)
        add_arguments_from_signature(parser, AMBSMixed)
        return parser

    def saved_keys(self, val: dict):
        res = {"id": val["id"], "arch_seq": str(val["arch_seq"])}
        hp_names = self.hp_space._space.get_hyperparameter_names()

        for hp_name in hp_names:
            if hp_name == "loss":
                res["loss"] = val["loss"]
            else:
                res[hp_name] = val["hyperparameters"][hp_name]

        return res

    def main(self):

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
                self.evaluator.dump_evals(saved_keys=self.saved_keys)

                num_received = len(new_results)
                num_evals_done += num_received

                # Transform configurations to list to fit optimizer
                opt_X = []
                opt_y = []
                for cfg, obj in new_results:
                    arch_seq = cfg["arch_seq"]
                    hp_val = self.problem.extract_hp_values(cfg)
                    x = replace_nan(hp_val + arch_seq)
                    opt_X.append(x)
                    opt_y.append(-obj)  #! maximizing

                self.opt.tell(opt_X, opt_y)  #! fit: costly
                new_X = self.opt.ask(
                    n_points=len(new_results), strategy=self.liar_strategy
                )

                new_batch = []
                for x in new_X:
                    new_cfg = self.problem.gen_config(
                        x[self.hp_size :], x[: self.hp_size]
                    )
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
        n_points = max(0, size - len(batch))
        if n_points > 0:
            points = self.opt.ask(n_points=n_points)
            for point in points:
                point_as_dict = self.problem.gen_config(
                    point[self.hp_size :], point[: self.hp_size]
                )
                batch.append(point_as_dict)
        return batch

    def to_dict(self, x: list) -> dict:
        hp_x = x[: self.hp_size]
        arch_seq = x[self.hp_size :]
        cfg = self.problem.space.copy()
        cfg["arch_seq"] = arch_seq
        return cfg


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
    args = AMBSMixed.parse_args()
    search = AMBSMixed(**vars(args))
    search.main()
