import math
import logging

import ConfigSpace as CS
import numpy as np
import skopt
from deephyper.problem import HpProblem
from deephyper.search.nas.base import NeuralArchitectureSearch


class AMBSMixed(NeuralArchitectureSearch):
    def __init__(
        self,
        problem,
        evaluator,
        random_state=None,
        log_dir=".",
        verbose=0,
        surrogate_model: str = "RF",
        acq_func: str = "LCB",
        kappa: float = 1.96,
        xi: float = 0.001,
        liar_strategy: str = "cl_min",
        n_jobs: int = 1,
        **kwargs,
    ):
        super().__init__(problem, evaluator, random_state, log_dir, verbose)

        self._n_initial_points = self._evaluator.num_workers
        self._liar_strategy = liar_strategy

        # Setup
        na_search_space = self._problem.build_search_space()

        self.hp_space = self._problem._hp_space  #! hyperparameters
        self.hp_size = len(self.hp_space.space.get_hyperparameter_names())
        self.na_space = HpProblem(self._problem.seed)
        for i, vnode in enumerate(na_search_space.variable_nodes):
            self.na_space.add_hyperparameter(
                (0, vnode.num_ops - 1), name=f"vnode_{i:05d}"
            )

        self._space = CS.ConfigurationSpace(seed=self._problem.seed)
        self._space.add_configuration_space(
            prefix="1", configuration_space=self.hp_space.space
        )
        self._space.add_configuration_space(
            prefix="2", configuration_space=self.na_space.space
        )

        self._opt = None
        self._opt_kwargs = dict(
            dimensions=self._space,
            base_estimator=self.get_surrogate_model(surrogate_model, n_jobs),
            acq_func=acq_func,
            acq_optimizer="sampling",
            acq_func_kwargs={"xi": float(xi), "kappa": float(kappa)},
            n_initial_points=self._n_initial_points,
            random_state=self._random_state,
        )

    def _setup_optimizer(self):
        self._opt = skopt.Optimizer(**self._opt_kwargs)

    def saved_keys(self, job):

        res = {"arch_seq": str(job.config["arch_seq"])}
        hp_names = self._problem._hp_space._space.get_hyperparameter_names()

        for hp_name in hp_names:
            if hp_name == "loss":
                res["loss"] = job.config["loss"]
            else:
                res[hp_name] = job.config["hyperparameters"][hp_name]

        return res

    def _search(self, max_evals, timeout):

        if self._opt is None:
            self._setup_optimizer()

        num_evals_done = 0

        # Filling available nodes at start
        logging.info(f"Generating {self._evaluator.num_workers} initial points...")
        self._evaluator.submit(self.get_random_batch(size=self._n_initial_points))

        # Main loop
        while max_evals < 0 or num_evals_done < max_evals:

            # Collecting finished evaluations
            new_results = list(self._evaluator.gather("BATCH", size=1))
            num_received = len(new_results)

            if num_received > 0:

                self._evaluator.dump_evals(
                    saved_keys=self.saved_keys, log_dir=self._log_dir
                )
                num_evals_done += num_received

                if num_evals_done >= max_evals:
                    break

                # Transform configurations to list to fit optimizer
                opt_X = []
                opt_y = []
                for cfg, obj in new_results:
                    arch_seq = cfg["arch_seq"]
                    hp_val = self._problem.extract_hp_values(cfg)
                    x = replace_nan(hp_val + arch_seq)
                    opt_X.append(x)
                    opt_y.append(-obj)  #! maximizing

                self._opt.tell(opt_X, opt_y)  #! fit: costly
                new_X = self._opt.ask(
                    n_points=len(new_results), strategy=self._liar_strategy
                )

                new_batch = []
                for x in new_X:
                    new_cfg = self._problem.gen_config(
                        x[self.hp_size :], x[: self.hp_size]
                    )
                    new_batch.append(new_cfg)

                # submit_childs
                if len(new_results) > 0:
                    self._evaluator.submit(new_batch)

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
            points = self._opt.ask(n_points=n_points)
            for point in points:
                point_as_dict = self._problem.gen_config(
                    point[self.hp_size :], point[: self.hp_size]
                )
                batch.append(point_as_dict)
        return batch


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