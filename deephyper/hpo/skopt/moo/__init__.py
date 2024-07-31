"""DeepHyper's multiobjective features.

DeepHyper solves multiobjective problems via scalarization.
A *scalarization* is a function that reduces several objectives to a single
target, which can be attained using any of DeepHyper's existing search
strategies, such as :class:`deephyper.hpo.CBO` or :class:`deephyper.hpo.MPIDistributedBO`.

If the user knows the tradeoff point that they would like to attain *a priori*,
then DeepHyper can use a fixed scalarization by using one of our
5 fixed-weighting scalarization methods.
To do so, when initializing the :class:`deephyper.search.Search` class, set
``moo_scalarization_strategy=["Linear", "Chebyshev", "AugChebyshev", "PBI", "Quadratic"]``.
Then, set the ``moo_scalarization_weight`` to a list of length equal to the
number of objectives, which DeepHyper will use for scalarization.

To interrogate the *entire* Pareto front, and not just a single tradeoff point,
one will need to solve multiple *different* scalarizations in parallel.
This can be achieved by using one of the 5 scalarization strategies with
randomized weights.
When initializing the :class:`deephyper.search.Search` class, set
``moo_scalarization_weight=None``.
DeepHyper will randomly generate new scalarization weights for each
candidate point.
This can be slightly more expensive than solving with a fixed scalarization
since multiobjective problems are inherently
more difficult than single-objective problems, but using different
scalarizations presents additional opportunity for parallelism.
When training an *ensemble* of models for prediction, setting 2 objectives
that balance model complexity vs accuracy and using randomized scalarizations
may produce a more diverse set of models.

Based on our experience, we recommend the ``"[r]AugChebyshev"`` option for most
applications.
For additional information, the corresponding strategies are documented in
the 5 classes from this module.
We also provide 5 common multiobjective utility functions for calculating
the hypervolume performance metric and extracting
Pareto efficient/non dominated point sets.
"""

from ._hv import hypervolume
from ._multiobjective import (
    MoAugmentedChebyshevFunction,
    MoChebyshevFunction,
    MoLinearFunction,
    MoPBIFunction,
    MoQuadraticFunction,
    MoScalarFunction,
)
from ._pf import (
    is_pareto_efficient,
    non_dominated_set,
    non_dominated_set_ranked,
    pareto_front,
)

moo_functions = {
    "Linear": MoLinearFunction,
    "Chebyshev": MoChebyshevFunction,
    "AugChebyshev": MoAugmentedChebyshevFunction,
    "PBI": MoPBIFunction,
    "Quadratic": MoQuadraticFunction,
}

__all__ = [
    "hypervolume",
    "MoLinearFunction",
    "MoAugmentedChebyshevFunction",
    "MoChebyshevFunction",
    "MoPBIFunction",
    "MoQuadraticFunction",
    "MoScalarFunction",
    "is_pareto_efficient",
    "moo_functions",
    "non_dominated_set",
    "non_dominated_set_ranked",
    "pareto_front",
]
