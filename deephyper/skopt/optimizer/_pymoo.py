import numpy as np
from collections import OrderedDict
from pymoo.core.problem import Problem
from pymoo.core.variable import Real, Integer, Choice
import deephyper.skopt.space as skopt_space


def convert_space_to_pymoo(space):
    """Convert a DeepHyper space to a pymoo space.

    Args:
        space (Space): from deephyper.skopt.space.

    Returns:
        dict: a pymoo space.
    """
    pymoo_space = OrderedDict()
    for dim in space.dimensions:
        if isinstance(dim, skopt_space.Real):
            pymoo_dim = Real(bounds=(dim.low, dim.high))
            pymoo_space[dim.name] = pymoo_dim
        elif isinstance(dim, skopt_space.Integer):
            pymoo_dim = Integer(bounds=(dim.low, dim.high))
            pymoo_space[dim.name] = pymoo_dim
        elif isinstance(dim, skopt_space.Categorical):
            if dim.transformed_size == 1:
                if dim.transform_ == "label":
                    options = [i for i in range(len(dim.categories))]
                elif dim.transform_ == "identity":
                    options = dim.categories
                else:
                    # TODO
                    print(f"{dim.name=}")
                    print(f"{dim.transform_=}")
                    raise NotImplementedError
                pymoo_dim = Choice(options=options)
                pymoo_space[dim.name] = pymoo_dim
            else:
                # TODO:
                print(f"{dim.name=}")
                print(f"{dim.transformed_size=}")
                raise NotImplementedError
        else:
            raise ValueError(f"Unknown dimension type {type(dim)}")
    return pymoo_space


class DeepHyperProblem(Problem):
    def __init__(self, space, acq_func=None, **kwargs):
        vars = convert_space_to_pymoo(space)
        super().__init__(vars=vars, n_obj=1, **kwargs)
        self.acq_func = acq_func

    def _evaluate(self, x, out, *args, **kwargs):
        if x.ndim == 2:
            x = x[0]
        x = np.array(list(map(lambda v: list(v.values()), x)))
        y = self.acq_func(x)
        out["F"] = y.reshape(-1, 1)
