from collections import OrderedDict

import numpy as np
from pymoo.core.problem import ElementwiseProblem, Problem
from pymoo.core.termination import Termination
from pymoo.core.variable import Choice, Integer, Real
from pymoo.termination.ftol import SingleObjectiveSpaceTermination
from pymoo.termination.max_eval import MaximumFunctionCallTermination
from pymoo.termination.max_gen import MaximumGenerationTermination
from pymoo.termination.robust import RobustTermination

import deephyper.skopt.space as skopt_space


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
    def __init__(self, space, acq_func=None, **kwargs):
        super().__init__(vars=convert_space_to_pymoo_mixed(space), n_obj=1, **kwargs)
        self.acq_func = acq_func
        self.vars_names = space.dimension_names

    def _evaluate(self, x, out, *args, **kwargs):
        if x.ndim == 2:
            x = x[0]
        x = np.asarray(list(map(lambda xi: [xi[k] for k in self.vars_names], x)))
        y = self.acq_func(x).reshape(-1)
        out["F"] = y


class PyMOOMixedElementWiseProblem(ElementwiseProblem):
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


class PyMOORealProblem(Problem):
    def __init__(self, n_var, xl, xu, acq_func, **kwargs):
        super().__init__(n_var=n_var, n_obj=1, xl=xl, xu=xu)
        self.acq_func = acq_func

    def _evaluate(self, x, out, *args, **kwargs):
        y = self.acq_func(x).reshape(-1)

        out["F"] = y


class DefaultMixedTermination(Termination):
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
    def __init__(self, ftol=1e-6, period=30, n_max_gen=1000, **kwargs) -> None:
        f = RobustTermination(
            SingleObjectiveSpaceTermination(ftol, only_feas=True), period=period
        )
        super().__init__(f, n_max_gen)
