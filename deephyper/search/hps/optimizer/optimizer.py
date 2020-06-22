import functools
import math
from sys import float_info

import numpy as np
from numpy import inf

from deephyper.search import util
from skopt import Optimizer as SkOptimizer
from skopt.learning import (
    ExtraTreesRegressor,
    GradientBoostingQuantileRegressor,
    RandomForestRegressor,
)

logger = util.conf_logger("deephyper.search.hps.optimizer.optimizer")


def isnan(x):
    if isinstance(x, float):
        return math.isnan(x)
    elif isinstance(x, np.float64):
        return np.isnan(x)
    else:
        return False


def convert2np(x):
    if x == "nan":
        return np.nan
    elif type(x) is float:
        return np.float64(x)
    elif type(x) is int:
        return np.int64(x)
    elif type(x) is str:
        return np.str_(x)
    else:
        return x


def equals(x, y):
    if isnan(x) and isnan(y):
        return True
    else:
        return x == y


def equal_list(l1, l2):
    return functools.reduce(
        lambda i, j: i and j, map(lambda m, k: equals(m, k), l1, l2), True
    )


class Optimizer:
    SEED = 12345

    def __init__(
        self,
        problem,
        num_workers,
        surrogate_model="RF",
        acq_func="gp_hedge",
        acq_kappa=1.96,
        acq_xi=None,
        liar_strategy="cl_max",
        n_jobs=1,
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
        # queue of remaining starting points
        self.starting_points = problem.starting_point

        n_init = (
            inf
            if surrogate_model == "DUMMY"
            else max(num_workers, len(self.starting_points))
        )

        # Set acq_func_kwargs parameters
        acq_func_kwargs = {}
        if type(acq_kappa) is float:
            acq_func_kwargs["kappa"] = acq_kappa
        if type(acq_xi) is float:
            acq_func_kwargs["xi"] = acq_xi

        self._optimizer = SkOptimizer(
            dimensions=self.space,
            base_estimator=base_estimator,
            acq_optimizer="sampling",
            acq_func=acq_func,
            acq_func_kwargs=acq_func_kwargs,
            random_state=self.SEED,
            n_initial_points=n_init,
        )

        assert liar_strategy in "cl_min cl_mean cl_max".split()
        self.strategy = liar_strategy
        self.evals = {}
        self.counter = 0
        logger.info(f"Using skopt.Optimizer with {surrogate_model} base_estimator")

    def _get_lie(self):
        if self.strategy == "cl_min":
            return min(self._optimizer.yi) if self._optimizer.yi else 0.0
        elif self.strategy == "cl_mean":
            return np.mean(self._optimizer.yi) if self._optimizer.yi else 0.0
        else:
            return max(self._optimizer.yi) if self._optimizer.yi else 0.0

    def _xy_from_dict(self):
        XX = []
        for key in self.evals.keys():
            x = tuple(convert2np(k) for k in key)
            XX.append(x)
        YY = [-self.evals[x] for x in self.evals.keys()]  # ! "-" maximizing
        return XX, YY

    def dict_to_xy(self, xy_dict: dict):
        XX = []
        for key in xy_dict.keys():
            x = [convert2np(k) for k in key]
            XX.append(x)
        YY = [-xy_dict[x] for x in xy_dict.keys()]  # ! "-" maximizing
        return XX, YY

    def to_dict(self, x: list) -> dict:
        res = {}
        hps_names = self.space.get_hyperparameter_names()
        for i in range(len(x)):
            res[hps_names[i]] = "nan" if isnan(x[i]) else x[i]
        return res

    def _ask(self):
        if len(self.starting_points) > 0:
            x = self.starting_points.pop()
        else:
            x = self._optimizer.ask()
        y = self._get_lie()
        key = tuple(self.to_dict(x).values())
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
            x = [convert2np(xi) for xi in x]
            key = tuple(self.to_dict(x).values())
            if key not in self.evals:
                self.counter += 1
                self._optimizer.tell(x, y)
                self.evals[key] = y
        return [self.to_dict(x) for x in XX]

    def tell(self, xy_data):
        assert isinstance(xy_data, list), f"where type(xy_data)=={type(xy_data)}"
        minval = min(self._optimizer.yi) if self._optimizer.yi else 0.0
        xy_dict = {}
        for x, y in xy_data:
            key = tuple(x[k] for k in self.space)
            assert key in self.evals, f"where key=={key} and self.evals=={self.evals}"
            logger.debug(f"tell: {x} --> {key}: evaluated objective: {y}")
            self.evals[key] = y if y > np.finfo(np.float32).min else minval
            xy_dict[key] = y if y > np.finfo(np.float32).min else minval

        XX, YY = self.dict_to_xy(xy_dict)

        selection = [
            (xi, yi)
            for xi, yi in zip(self._optimizer.Xi, self._optimizer.yi)
            if not (
                any([equal_list(xi, x) for x in XX])
            )  # all([diff(xi, x) for x in XX])
        ]

        new_Xi, new_yi = list(zip(*selection)) if len(selection) > 0 else ([], [])
        new_Xi, new_yi = list(new_Xi), list(new_yi)

        self._optimizer.Xi = new_Xi
        self._optimizer.yi = new_yi
        self._optimizer.tell(XX, YY)

        assert len(self._optimizer.Xi) == len(self._optimizer.yi) == self.counter, (
            f"where len(self._optimizer.Xi)=={len(self._optimizer.Xi)}, "
            f"len(self._optimizer.yi)=={len(self._optimizer.yi)},"
            f"self.counter=={self.counter}"
        )


def diff(xi, x):
    for e_xi, e_x in zip(xi, x):
        if e_xi != e_x:
            return True
    return False
