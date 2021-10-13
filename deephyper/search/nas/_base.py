from deephyper.search import Search


class NeuralArchitectureSearch(Search):
    def __init__(
        self, problem, evaluator, random_state=None, log_dir=".", verbose=0, **kwargs
    ):
        super().__init__(problem, evaluator, random_state, log_dir, verbose)

        if self._problem._space["log_dir"] is None:
            self._problem._space["log_dir"] = self._log_dir