from deephyper.search.nas.base import NeuralArchitectureSearch


class Random(NeuralArchitectureSearch):
    """Search class to run a full random neural architecture search. The search is filling every available nodes as soon as they are detected. The master job is using only 1 MPI rank.

    Args:
        problem (str): Module path to the Problem instance you want to use for the search (e.g. deephyper.benchmark.nas.linearReg.Problem).
        run (str): Module path to the run function you want to use for the search (e.g. deephyper.nas.run.quick).
        evaluator (str): value in ['balsam', 'subprocess', 'processPool', 'threadPool'].
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

        self.free_workers = self._evaluator.num_workers

        self.pb_dict = self._problem.space
        search_space = self._problem.build_search_space()

        self.space_list = [
            (0, vnode.num_ops - 1) for vnode in search_space.variable_nodes
        ]

    def saved_keys(self, job):
        res = {"arch_seq": str(job.config["arch_seq"])}
        return res

    def _search(self, max_evals, timeout):

        num_evals_done = 0
        available_workers = self._evaluator.num_workers

        # Filling available nodes at start
        batch = self.gen_random_batch(size=available_workers)
        self._evaluator.submit(batch)

        # Main loop
        while num_evals_done < max_evals:
            results = self._evaluator.gather("BATCH", 1)

            num_received = num_evals_done
            num_evals_done += len(results)
            num_received = num_evals_done - num_received

            # Filling available nodes
            if num_received > 0:
                self._evaluator.dump_evals(saved_keys=self.saved_keys)

                if num_evals_done < max_evals:
                    self._evaluator.submit(self.gen_random_batch(size=num_received))

    def gen_random_batch(self, size: int) -> list:
        batch = []
        for _ in range(size):
            cfg = self.pb_dict.copy()
            cfg["arch_seq"] = self.gen_random_arch()
            batch.append(cfg)
        return batch

    def gen_random_arch(self) -> list:
        return [self._random_state.choice(b + 1) for (_, b) in self.space_list]
