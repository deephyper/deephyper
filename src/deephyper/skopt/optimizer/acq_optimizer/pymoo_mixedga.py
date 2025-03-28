"""Mixed-Integer genetic-algorithm optimization for the acquisition function."""

from collections import OrderedDict

import numpy as np
from ConfigSpace.forbidden import ForbiddenClause, ForbiddenConjunction, ForbiddenRelation
from ConfigSpace.util import deactivate_inactive_hyperparameters
from pymoo.config import Config
from pymoo.core.mixed import (
    MixedVariableDuplicateElimination,
    MixedVariableGA,
    MixedVariableMating,
)
from pymoo.core.population import Population
from pymoo.core.problem import ElementwiseProblem, Problem
from pymoo.core.repair import Repair
from pymoo.core.termination import Termination
from pymoo.core.variable import Choice, Integer, Real
from pymoo.optimize import minimize
from pymoo.termination.ftol import SingleObjectiveSpaceTermination
from pymoo.termination.max_eval import MaximumFunctionCallTermination
from pymoo.termination.max_gen import MaximumGenerationTermination
from pymoo.termination.robust import RobustTermination
from sklearn.utils import check_random_state

import deephyper.skopt.space as skopt_space

Config.warnings["not_compiled"] = False

# https://pymoo.org/customization/mixed.html
# https://pymoo.org/interface/problem.html


Config.warnings["not_compiled"] = False


def convert_space_to_pymoo_mixed(space):
    """Convert a DeepHyper space to a pymoo space.

    Optimizing in the source input space.

    Args:
        space (Space): from deephyper.skopt.space.

    Returns:
        dict: a pymoo space.
    """
    pymoo_space = OrderedDict()
    for dim in space.dimensions:
        if isinstance(dim, skopt_space.Real):
            pymoo_dim = Real(bounds=dim.bounds)
            pymoo_space[dim.name] = pymoo_dim
        elif isinstance(dim, skopt_space.Integer):
            pymoo_dim = Integer(bounds=dim.bounds)
            pymoo_space[dim.name] = pymoo_dim
        elif isinstance(dim, skopt_space.Categorical):
            options = dim.categories
            pymoo_dim = Choice(options=options)
            pymoo_space[dim.name] = pymoo_dim
        else:
            raise ValueError(f"Unknown dimension type {type(dim)}")
    return pymoo_space


class PyMOOMixedVectorizedProblem(Problem):
    """Pymoo mixed-integer problem definition (vectorized)."""

    def __init__(self, space, acq_func=None, **kwargs):
        super().__init__(vars=convert_space_to_pymoo_mixed(space), n_obj=1, **kwargs)
        self.space = space
        self.acq_func = acq_func
        self.vars_names = space.dimension_names

    def _evaluate(self, x, out, *args, **kwargs):
        if x.ndim == 2:
            x = x[0]

        # !Using np.array blindly can lead to errors by mapping types to string
        x = list(
            map(
                lambda xi: [xi[k] for k in self.vars_names],
                x,
            )
        )
        y = self.acq_func(x).reshape(-1)
        out["F"] = y


class PyMOOMixedElementWiseProblem(ElementwiseProblem):
    """Pymoo mixed-integer problem definition (element-wise)."""

    def __init__(self, space, acq_func=None, **kwargs):
        vars = convert_space_to_pymoo_mixed(space)
        super().__init__(vars=vars, n_obj=1, **kwargs)
        self.acq_func = acq_func

    def _evaluate(self, x, out, *args, **kwargs):
        x = np.array(list(x.values()))
        try:
            y = self.acq_func([x])[0]
        except ValueError:
            y = np.inf
        out["F"] = y


class DefaultMixedTermination(Termination):
    """Pymoo custom termination criteria for mixed-integer problem."""

    def __init__(self, f, n_max_gen=1000, n_max_evals=100000) -> None:
        super().__init__()
        self.f = f

        self.max_gen = MaximumGenerationTermination(n_max_gen)
        self.max_evals = MaximumFunctionCallTermination(n_max_evals)

        self.criteria = [self.f, self.max_gen, self.max_evals]

    def _update(self, algorithm):
        p = [criterion.update(algorithm) for criterion in self.criteria]
        return max(p)


class DefaultSingleObjectiveMixedTermination(DefaultMixedTermination):
    """Pymoo custom default single objectived mixed-integer termination criteria."""

    def __init__(self, ftol=1e-6, period=30, n_max_gen=1000, **kwargs) -> None:
        f = RobustTermination(SingleObjectiveSpaceTermination(ftol, only_feas=True), period=period)
        super().__init__(f, n_max_gen)


class ConfigSpaceRepair(Repair):
    """Pymoo repair operator for ConfigSpace conditions/forbiddens."""

    def __init__(self, space):
        super().__init__()
        self.space = space
        self.config_space = self.space.config_space

    def _do(self, problem, x, **kwargs):
        def deactivate_inactive_dimensions(x: dict):
            if len(self.config_space.forbidden_clauses) > 0:
                # Resolve forbidden
                max_trials = 10
                num_trials = 0
                while (num_trials < max_trials) and any(
                    f.is_forbidden_value(x) for f in self.config_space.forbidden_clauses
                ):
                    # The new x respect all forbiddens
                    x_new = dict(self.config_space.sample_configuration())
                    for forbidden in self.config_space.forbidden_clauses:
                        if forbidden.is_forbidden_value(x):
                            if isinstance(forbidden, ForbiddenConjunction):
                                dlcs = forbidden.dlcs
                            else:
                                dlcs = [forbidden]

                            for f in dlcs:
                                if isinstance(f, ForbiddenClause):
                                    x[f.hyperparameter.name] = x_new[f.hyperparameter.name]
                                elif isinstance(f, ForbiddenRelation):
                                    x[f.right.name] = x_new[f.right.name]
                                    x[f.left.name] = x_new[f.left.name]
                    num_trials += 1
                # No possible fix was found we override with a valid config
                if max_trials == num_trials:
                    x = x_new

            x = dict(deactivate_inactive_hyperparameters(x, self.config_space))
            for i, hps_name in enumerate(self.space.dimension_names):
                # If the parameter is inactive due to some conditions then we attribute the
                # lower bound value to break symmetries and enforce the same representation.
                x[hps_name] = x.get(hps_name, self.space.dimensions[i].bounds[0])
            return x

        if self.config_space:
            # If there are forbiddens they must be treated before conditions

            # Dealing with conditions
            x[:] = list(
                map(
                    lambda xi: deactivate_inactive_dimensions(xi),
                    x,
                )
            )

        return x


class MixedGAPymooAcqOptimizer:
    """Mixed-Integer GA optimizer using Pymoo."""

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
        self.x_init = np.array(x_init)
        self.y_init = np.array(y_init).reshape(-1)
        self.pop_size = pop_size
        self.random_state = check_random_state(random_state)

        if termination_kwargs is None:
            termination_kwargs = {}
        default_termination_kwargs = {
            "ftol": 1e-6,
            "period": 30,
            "n_max_gen": 1000,
        }
        default_termination_kwargs.update(termination_kwargs)
        self.termination_kwargs = default_termination_kwargs

    def minimize(self, acq_func):
        """Minimize the acquisition function."""
        problem = PyMOOMixedVectorizedProblem(
            space=self.space,
            acq_func=lambda x: acq_func(self.space.transform(x)),
        )

        init_pop = Population.new(
            "X",
            self.x_init,
            "F",
            self.y_init,
        )

        repair = ConfigSpaceRepair(self.space)
        eliminate_duplicates = MixedVariableDuplicateElimination()
        algorithm = MixedVariableGA(
            pop_size=self.pop_size,
            sampling=init_pop,
            mating=MixedVariableMating(
                eliminate_duplicates=eliminate_duplicates,
                repair=repair,
            ),
            repair=repair,
            eliminate_duplicates=eliminate_duplicates,
        )

        res_ga = minimize(
            problem,
            algorithm,
            termination=DefaultSingleObjectiveMixedTermination(**self.termination_kwargs),
            seed=self.random_state.randint(0, np.iinfo(np.int32).max),
            verbose=False,
        )

        res_X = [res_ga.X[name] for name in self.space.dimension_names]
        res_X = self.space.transform([res_X])[0]
        return res_X
