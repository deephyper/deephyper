import collections

from deephyper.search.nas.base import NeuralArchitectureSearch


class RegularizedEvolution(NeuralArchitectureSearch):
    """Regularized evolution.
    https://arxiv.org/abs/1802.01548
    Args:
        problem (str): Module path to the Problem instance you want to use for the search (e.g. deephyper.benchmark.nas.linearReg.Problem).
        run (str): Module path to the run function you want to use for the search (e.g. deephyper.nas.run.quick).
        evaluator (str): value in ['balsam', 'subprocess', 'processPool', 'threadPool'].
        population_size (int, optional): the number of individuals to keep in the population. Defaults to 100.
        sample_size (int, optional): the number of individuals that should participate in each tournament. Defaults to 10.
    """

    def __init__(
        self,
        problem,
        evaluator,
        random_state: int = None,
        log_dir: str = ".",
        verbose: int = 0,
        population_size: int = 100,
        sample_size: int = 10,
        **kwargs
    ):

        super().__init__(problem, evaluator, random_state, log_dir, verbose)

        # Setup
        self.pb_dict = self._problem.space
        search_space = self._problem.build_search_space()

        self.space_list = [
            (0, vnode.num_ops - 1) for vnode in search_space.variable_nodes
        ]
        self._population_size = int(population_size)
        self._sample_size = int(sample_size)
        self._population = collections.deque(maxlen=self._population_size)

    def saved_keys(self, job):
        res = {"arch_seq": str(job.config["arch_seq"])}
        return res

    def _search(self, max_evals, timeout):

        num_evals_done = 0

        # Filling available nodes at start
        batch = self.gen_random_batch(size=self._evaluator.num_workers)
        self._evaluator.submit(batch)

        # Main loop
        while max_evals < 0 or num_evals_done < max_evals:

            # Collecting finished evaluations
            new_results = self._evaluator.gather("BATCH", 1)
            num_received = len(new_results)

            if num_received > 0:
                self._population.extend(new_results)
                self._evaluator.dump_evals(
                    saved_keys=self.saved_keys, log_dir=self._log_dir
                )
                num_evals_done += num_received

                if num_evals_done >= max_evals:
                    break

                # If the population is big enough evolve the population
                if len(self._population) == self._population_size:

                    children_batch = []

                    # For each new parent/result we create a child from it
                    for _ in range(num_received):
                        # select_sample
                        indexes = self._random_state.choice(
                            self._population_size, self._sample_size, replace=False
                        )
                        sample = [self._population[i] for i in indexes]
                        # select_parent
                        parent = self.select_parent(sample)
                        # copy_mutate_parent
                        child = self.copy_mutate_arch(parent)
                        # add child to batch
                        children_batch.append(child)

                    # submit_childs
                    self._evaluator.submit(children_batch)

                else:  # If the population is too small keep increasing it
                    self._evaluator.submit(self.gen_random_batch(size=num_received))

    def select_parent(self, sample: list) -> list:
        cfg, _ = max(sample, key=lambda x: x[1])
        return cfg["arch_seq"]

    def gen_random_batch(self, size: int) -> list:
        batch = []
        for _ in range(size):
            cfg = self.pb_dict.copy()
            cfg["arch_seq"] = self.random_search_space()
            batch.append(cfg)
        return batch

    def random_search_space(self) -> list:
        return [self._random_state.choice(b + 1) for (_, b) in self.space_list]

    def copy_mutate_arch(self, parent_arch: list) -> dict:
        """
        # ! Time performance is critical because called sequentialy
        Args:
            parent_arch (list(int)): [description]
        Returns:
            dict: [description]
        """
        i = self._random_state.choice(len(parent_arch))
        child_arch = parent_arch[:]

        range_upper_bound = self.space_list[i][1]
        elements = [j for j in range(range_upper_bound + 1) if j != child_arch[i]]

        # The mutation has to create a different search_space!
        sample = self._random_state.choice(elements, 1)[0]

        child_arch[i] = sample
        cfg = self.pb_dict.copy()
        cfg["arch_seq"] = child_arch
        return cfg