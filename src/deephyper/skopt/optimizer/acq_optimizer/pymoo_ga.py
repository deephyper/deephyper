"""Continuous genetic-algorithm optimization for the acquisition function."""

import warnings

import numpy as np

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.population import Population
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultSingleObjectiveTermination
from pymoo.core.problem import Problem
from pymoo.config import Config
from sklearn.utils import check_random_state

Config.warnings["not_compiled"] = False


class PyMOORealProblem(Problem):
    """Pymoo continuous problem definition."""

    def __init__(self, n_var, xl, xu, acq_func, **kwargs):
        super().__init__(n_var=n_var, n_obj=1, xl=xl, xu=xu)
        self.acq_func = acq_func

    def _evaluate(self, x, out, *args, **kwargs):
        y = self.acq_func(x).reshape(-1)

        out["F"] = y


class GAPymooAcqOptimizer:
    """Continuous GA optimizer using Pymoo."""

    def __init__(
        self,
        space,
        x_init,
        y_init,
        pop_size: int = 100,
        random_state=None,
        termination_kwargs=None,
    ):
        self.space = space
        
        # Bounds in the transformed/normalized space
        transformed_bounds = self.space.transformed_bounds
        xl, xu = list(zip(*transformed_bounds))
        xl, xu = np.array(xl), np.array(xu)
        self.xl = np.array(xl)
        self.xu = np.array(xu)
        self.n_var = len(self.xl)

        if len(y_init) != len(x_init):
            raise ValueError("The initial x_init and y_init values should have the same size.")
        self.x_init = np.array(x_init)
        self.y_init = np.array(y_init).reshape(-1)
        assert len(y_init) == pop_size
        if len(y_init) != pop_size:
            raise ValueError("The initial x_init and y_init should have a size equal to pop_size.")
        
        self.pop_size = pop_size
        self.random_state = check_random_state(random_state)

        if termination_kwargs is None:
            termination_kwargs = {}
        default_termination_kwargs = {
            "xtol": 1e-8,
            "ftol": 1e-6,
            "period": 30,
            "n_max_gen": 1_000,
        }
        default_termination_kwargs.update(termination_kwargs)
        self.termination_kwargs = default_termination_kwargs

    def minimize(self, acq_func):
        """Minimize the acquisition function."""
        problem = PyMOORealProblem(
            n_var=self.n_var,
            xl=self.xl,
            xu=self.xu,
            acq_func=acq_func,
        )

        init_pop = Population.new(
            "X",
            self.x_init,
            "F",
            self.y_init.reshape(-1),
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            res = minimize(
                problem,
                algorithm=GA(pop_size=self.pop_size, sampling=init_pop),
                termination=DefaultSingleObjectiveTermination(**self.termination_kwargs),
                seed=self.random_state.randint(0, np.iinfo(np.int32).max),
                verbose=False,
            )

        return res.X
