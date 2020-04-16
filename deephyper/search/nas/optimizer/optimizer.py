from sys import float_info
from skopt import Optimizer as SkOptimizer
from skopt.learning import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingQuantileRegressor,
)

# from numpy import inf
import numpy as np
from deephyper.search import util

# from deephyper.problem import HpProblem

logger = util.conf_logger("deephyper.search.hps.optimizer.optimizer")


class Optimizer:
    SEED = 12345

    def __init__(
        self,
        problem,
        num_workers,
        surrogate_model="RF",
        acq_func="gp_hedge",
        acq_kappa=1.96,
        liar_strategy="cl_max",
        n_jobs=-1,
        **kwargs,
    ):
        assert surrogate_model in [
            "RF",
            "ET",
            "GBRT",
            "GP",
            "DUMMY",
        ], f"Unknown scikit-optimize base_estimator: {surrogate_model}"
        if surrogate_model == "RF":
            base_estimator = RandomForestRegressor(n_jobs=n_jobs)
        elif surrogate_model == "ET":
            base_estimator = ExtraTreesRegressor(n_jobs=n_jobs)
        elif surrogate_model == "GBRT":
            base_estimator = GradientBoostingQuantileRegressor(n_jobs=n_jobs)
        else:
            base_estimator = surrogate_model

        self.space = problem.space
        cs_kwargs = self.space["create_search_space"].get("kwargs")
        if cs_kwargs is None:
            search_space = self.space["create_search_space"]["func"]()
        else:
            search_space = self.space["create_search_space"]["func"](**cs_kwargs)

        # // queue of remaining starting points
        # // self.starting_points = problem.starting_point
        n_init = np.inf if surrogate_model == "DUMMY" else num_workers

        self.starting_points = []  # ! EMPTY for now TODO

        # Building search space for SkOptimizer
        skopt_space = [(0, vnode.num_ops - 1) for vnode in search_space.variable_nodes]

        self._optimizer = SkOptimizer(
            skopt_space,
            base_estimator=base_estimator,
            acq_optimizer="sampling",
            acq_func=acq_func,
            acq_func_kwargs={"kappa": acq_kappa},
            random_state=self.SEED,
            n_initial_points=n_init,
        )

        assert liar_strategy in "cl_min cl_mean cl_max".split()
        self.strategy = liar_strategy
        self.evals = {}
        self.counter = 0
        logger.info("Using skopt.Optimizer with %s base_estimator" % surrogate_model)

    def _get_lie(self):
        if self.strategy == "cl_min":
            return min(self._optimizer.yi) if self._optimizer.yi else 0.0
        elif self.strategy == "cl_mean":
            return np.mean(self._optimizer.yi) if self._optimizer.yi else 0.0
        else:  # self.strategy == "cl_max"
            return max(self._optimizer.yi) if self._optimizer.yi else 0.0

    def _xy_from_dict(self):
        XX = list(self.evals.keys())
        YY = [-self.evals[x] for x in XX]
        return XX, YY

    def to_dict(self, x):
        cfg = self.space.copy()
        cfg["arch_seq"] = list(x)
        return cfg

    def _ask(self):
        if len(self.starting_points) > 0:
            x = self.starting_points.pop()
        else:
            x = self._optimizer.ask()
        y = self._get_lie()
        key = tuple(x)
        if key not in self.evals:
            self.counter += 1
            self._optimizer.tell(x, y)
            self.evals[key] = y
            logger.debug(f"_ask: {x} lie: {y}")
        else:
            logger.debug(f"Duplicate _ask: {x} lie: {y}")
        return self.to_dict(x)

    def ask(self, n_points=None, batch_size=20):
        if n_points is None:
            return self._ask()
        else:
            batch = []
            for _ in range(n_points):
                batch.append(self._ask())
                if len(batch) == batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch

    def ask_initial(self, n_points):
        if len(self.starting_points) > 0:
            XX = [
                self.starting_points.pop()
                for i in range(min(n_points, len(self.starting_points)))
            ]
            if len(XX) < n_points:
                XX += self._optimizer.ask(n_points=n_points - len(XX))
        else:
            XX = self._optimizer.ask(n_points=n_points)
        for x in XX:
            y = self._get_lie()
            key = tuple(x)
            if key not in self.evals:
                self.counter += 1
                self._optimizer.tell(x, y)
                self.evals[key] = y
        return [self.to_dict(x) for x in XX]

    def tell(self, xy_data):
        assert isinstance(xy_data, list), f"where type(xy_data)=={type(xy_data)}"
        minval = min(self._optimizer.yi) if self._optimizer.yi else 0.0
        for x, y in xy_data:
            key = tuple(x["arch_seq"])
            assert key in self.evals, f"where key=={key} and self.evals=={self.evals}"
            logger.debug(f"tell: {x} --> {key}: evaluated objective: {y}")
            self.evals[key] = y if y > float_info.min else minval

        self._optimizer.Xi = []
        self._optimizer.yi = []
        XX, YY = self._xy_from_dict()
        assert len(XX) == len(YY) == self.counter, (
            f"where len(XX)=={len(XX)},"
            f"len(YY)=={len(YY)}, self.counter=={self.counter}"
        )
        self._optimizer.tell(XX, YY)
        assert len(self._optimizer.Xi) == len(self._optimizer.yi) == self.counter, (
            f"where len(self._optimizer.Xi)=={len(self._optimizer.Xi)}, "
            f"len(self._optimizer.yi)=={len(self._optimizer.yi)},"
            f"self.counter=={self.counter}"
        )
