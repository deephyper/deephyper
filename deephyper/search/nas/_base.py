from deephyper.search._search import Search


class NeuralArchitectureSearch(Search):
    def __init__(
        self, problem, evaluator, random_state=None, log_dir=".", verbose=0, **kwargs
    ):
        super().__init__(problem, evaluator, random_state, log_dir, verbose)

        self._problem._space["log_dir"] = self._log_dir
        self._problem._space["verbose"] = self._verbose
        self._problem._space["seed"] = self._random_state.get_state()[1][0]

        # HPS search space
        self._problem._hp_space._space.seed(self._random_state.get_state()[1][0])

    def _add_default_keys(self, config: dict) -> dict:
        config["log_dir"] = self._log_dir
        config["seed"] = self._seed
        config["verbose"] = self._verbose
        return config
