from typing import Literal
from deephyper.hpo._cbo import CBO
from deephyper.hpo._solution import SolutionSelection

__all__ = ["ExperimentalDesignSearch"]


class ExperimentalDesignSearch(CBO):
    """Centralized Experimental Design Search.

    It follows a manager-workers architecture where the manager runs the sampling process and
    workers execute parallel evaluations of the black-box function.

    .. list-table::
        :widths: 25 25 25
        :header-rows: 1

        * - Single-Objective
          - Multi-Objectives
          - Failures
        * - ✅
          - ✅
          - ✅

    Example Usage:

        >>> max_evals = 100
        >>> search = ExperimentalDesignSearch(problem, evaluator, n_points=max_evals, design="grid")
        >>> results = search.search(max_evals=100)

    Args:
        problem (HpProblem):
            Hyperparameter problem describing the search space to explore.

        evaluator (Evaluator):
            An ``Evaluator`` instance responsible of distributing the tasks.

        random_state (int, optional):
            Random seed. Defaults to ``None``.

        log_dir (str, optional):
            Log directory where search's results are saved. Defaults to ``"."``.

        verbose (int, optional):
            Indicate the verbosity level of the search. Defaults to ``0``.

        stopper (Stopper, optional):
            a stopper to leverage multi-fidelity when evaluating the function. Defaults to
            ``None`` which does not use any stopper.

        checkpoint_history_to_csv (bool, optional):
            wether the results from progressively collected evaluations should be checkpointed
            regularly to disc as a csv. Defaults to ``True``.

        solution_selection (Literal["argmax_obs", "argmax_est"] | SolutionSelection, optional):
            the solution selection strategy. It can be a string where ``"argmax_obs"`` would
            select the argmax of observed objective values, and ``"argmax_est"`` would select the
            argmax of estimated objective values (through a predictive model).

        n_points (int, optional):
            Number of points to sample. Defaults to ``None``.

        design (str, optional):
            Experimental design to use, it can be one of:
            - ``"random"`` for uniform random numbers.
            - ``"sobol"`` for a Sobol' sequence.
            - ``"halton"`` for a Halton sequence.
            - ``"hammersly"`` for a Hammersly sequence.
            - ``"lhs"`` for a latin hypercube sequence.
            - ``"grid"`` for a uniform grid sequence.
            Defaults to ``"random"``.

        initial_points (list, optional):
            List of initial points to evaluate. Defaults to ``None``.
    """

    def __init__(
        self,
        problem,
        evaluator,
        random_state: int = None,
        log_dir: str = ".",
        verbose: int = 0,
        stopper=None,
        checkpoint_history_to_csv: bool = True,
        solution_selection: Literal["argmax_obs", "argmax_est"] | SolutionSelection = "argmax_obs",
        n_points: int = None,
        design: str = "random",
        initial_points=None,
    ):
        if n_points is None:
            raise ValueError("n_points must be specified for the ExperimentalDesignSearch.")
        super().__init__(
            problem,
            evaluator,
            random_state,
            log_dir,
            verbose,
            stopper,
            checkpoint_history_to_csv,
            solution_selection,
            n_initial_points=n_points,
            initial_points=initial_points,
            initial_point_generator=design,
            surrogate_model="DUMMY",
        )
