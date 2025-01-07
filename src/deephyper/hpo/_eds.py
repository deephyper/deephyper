from deephyper.hpo._cbo import CBO


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

        >>> from deephyper.hpo import HpProblem, ExperimentalDesignSearch
        >>> problem = HpProblem()
        >>> problem.add_hyperparameter((0.0, 1.0), "x")
        UniformFloatHyperparameter(name='x', default_value=0.5, meta=None, size=inf, lower=0.0, upper=1.0, log=False)
        >>> problem.add_hyperparameter((0.0, 1.0), "y")
        UniformFloatHyperparameter(name='y', default_value=0.5, meta=None, size=inf, lower=0.0, upper=1.0, log=False)
        >>> def run(job):
        ...     return sum(job.parameters.values())
        >>> max_evals = 100
        >>> search = ExperimentalDesignSearch(problem, run, n_points=max_evals, design="grid")
        >>> results = search.search(max_evals)

    Args:
        problem (HpProblem): Hyperparameter problem describing the search space to explore.

        evaluator (Evaluator): An ``Evaluator`` instance responsible of distributing the tasks.

        random_state (int, optional): Random seed. Defaults to ``None``.

        log_dir (str, optional): Log directory where search's results are saved. Defaults to
            ``"."``.

        verbose (int, optional): Indicate the verbosity level of the search. Defaults to ``0``.

        stopper (Stopper, optional): a stopper to leverage multi-fidelity when evaluating the
            function. Defaults to ``None`` which does not use any stopper.

        n_points (int, optional): Number of points to sample. Defaults to ``None``.

        design (str, optional): Experimental design to use. Defaults to ``"random"``.
            - ``"random"`` for uniform random numbers.
            - ``"sobol"`` for a Sobol' sequence.
            - ``"halton"`` for a Halton sequence.
            - ``"hammersly"`` for a Hammersly sequence.
            - ``"lhs"`` for a latin hypercube sequence.
            - ``"grid"`` for a uniform grid sequence.

        initial_points (list, optional): List of initial points to evaluate. Defaults to ``None``.
    """  # noqa: E501

    def __init__(
        self,
        problem,
        evaluator,
        random_state: int = None,
        log_dir: str = ".",
        verbose: int = 0,
        stopper=None,
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
            n_initial_points=n_points,
            initial_points=initial_points,
            initial_point_generator=design,
            surrogate_model="DUMMY",
        )
