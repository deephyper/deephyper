from deephyper.problem.base import BaseProblem


class HpProblem(BaseProblem):
    """Problem specification for Hyperparameter Search
    """

    def __init__(self, seed=None):
        super().__init__(seed=seed)
