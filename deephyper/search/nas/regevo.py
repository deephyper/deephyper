import os
import collections
import numpy as np
import json

from deephyper.search import util
from deephyper.search.nas import NeuralArchitectureSearch
from deephyper.core.parser import add_arguments_from_signature
from deephyper.core.logs.logging import JsonMessage as jm

#dhlogger = util.conf_logger("deephyper.search.nas.regevo")


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
        self, problem, evaluator, random_state=None, log_dir=".", verbose=0, population_size=100, sample_size=10, **kwargs
    ):

        super().__init__(problem, evaluator, random_state, log_dir, verbose)

        self.free_workers = self._evaluator.num_workers

        # Setup
        self.pb_dict = self._problem.space
        search_space = self._problem.build_search_space()

        self.space_list = [
            (0, vnode.num_ops - 1) for vnode in search_space.variable_nodes
        ]
        self.population_size = int(population_size)
        self.sample_size = int(sample_size)

    def saved_keys(self, job):
        res = {"arch_seq": str(job.config["arch_seq"])}
        return res

    def _search(self, max_evals, timeout):

        num_evals_done = 0
        # num_cyles_done = 0
        population = collections.deque(maxlen=self.population_size)

        # Filling available nodes at start
        batch = self.gen_random_batch(size=self.free_workers)
        self._evaluator.submit(batch)

        # Main loop
        while num_evals_done < max_evals:

            # Collecting finished evaluations
            new_results = self._evaluator.gather("BATCH", 1)

            if len(new_results) > 0:
                population.extend(new_results)
                #stats = {"num_cache_used": self._evaluator.stats["num_cache_used"]}
                #dhlogger.info(jm(type="env_stats", **stats))
                self._evaluator.dump_evals(saved_keys=self.saved_keys)

                num_received = len(new_results)
                num_evals_done += num_received

                if num_evals_done >= max_evals:
                    break

                # If the population is big enough evolve the population
                if len(population) == self.population_size:
                    children_batch = []
                    # For each new parent/result we create a child from it
                    for _ in range(len(new_results)):
                        # select_sample
                        indexes = self._random_state.choice(
                            self.population_size, self.sample_size, replace=False
                        )
                        sample = [population[i] for i in indexes]
                        # select_parent
                        parent = self.select_parent(sample)
                        # copy_mutate_parent
                        child = self.copy_mutate_arch(parent)
                        # add child to batch
                        children_batch.append(child)
                    # submit_childs
                    if len(new_results) > 0:
                        self._evaluator.submit(children_batch)
                else:  # If the population is too small keep increasing it
                    self._evaluator.submit(
                        self.gen_random_batch(size=len(new_results))
                    )

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