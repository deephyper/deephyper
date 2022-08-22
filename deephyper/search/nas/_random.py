from deephyper.search.nas._base import NeuralArchitectureSearch


class Random(NeuralArchitectureSearch):
    """Random neural architecture search. This search algorithm is compatible with a ``NaProblem`` defining fixed or variable hyperparameters.

    Args:
        problem (NaProblem): Neural architecture search problem describing the search space to explore.
        evaluator (Evaluator): An ``Evaluator`` instance responsible of distributing the tasks.
        random_state (int or RandomState, optional): Random seed. Defaults to None.
        log_dir (str, optional): Log directory where search's results are saved. Defaults to ".".
        verbose (int, optional): Indicate the verbosity level of the search. Defaults to 0.
    """

    def __init__(
        self,
        problem,
        evaluator,
        random_state: int = None,
        log_dir: str = ".",
        verbose: int = 0,
        **kwargs
    ):
        super().__init__(problem, evaluator, random_state, log_dir, verbose)

        # NAS search space
        self._space_list = self._problem.build_search_space().choices()

    def _saved_keys(self, job):

        res = {"arch_seq": str(job.config["arch_seq"])}
        hp_names = self._problem._hp_space._space.get_hyperparameter_names()

        for hp_name in hp_names:
            if hp_name == "loss":
                res["loss"] = job.config["loss"]
            else:
                res[hp_name] = job.config["hyperparameters"][hp_name]

        return res

    def _search(self, max_evals, timeout):

        num_evals_done = 0

        # Filling available nodes at start
        batch = self._gen_random_batch(size=self._evaluator.num_workers)
        self._evaluator.submit(batch)

        # Main loop
        while max_evals < 0 or num_evals_done < max_evals:
            results = self._evaluator.gather("BATCH", 1)

            num_received = num_evals_done
            num_evals_done += len(results)
            num_received = num_evals_done - num_received

            # Filling available nodes
            if num_received > 0:
                self._evaluator.dump_evals(
                    saved_keys=self._saved_keys, log_dir=self._log_dir
                )

                if max_evals < 0 or num_evals_done < max_evals:
                    self._evaluator.submit(self._gen_random_batch(size=num_received))

    def _gen_random_batch(self, size: int) -> list:
        batch = []

        hp_values_samples = self._problem._hp_space._space.sample_configuration(size)
        if size == 1:
            hp_values_samples = [hp_values_samples]

        for i in range(size):
            arch_seq = self._gen_random_arch()
            hp_values = list(dict(hp_values_samples[i]).values())
            config = self._problem.gen_config(arch_seq, hp_values)
            config = self._add_default_keys(config)
            batch.append(config)

        return batch

    def _gen_random_arch(self) -> list:
        return [self._random_state.choice(b + 1) for (_, b) in self._space_list]
