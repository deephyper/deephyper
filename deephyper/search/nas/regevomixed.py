import collections

import ConfigSpace as CS
import numpy as np
from deephyper.problem import HpProblem
from deephyper.search.nas.regevo import RegularizedEvolution


class RegularizedEvolutionMixed(RegularizedEvolution):
    def __init__(
        self,
        problem,
        evaluator,
        random_state: int = None,
        log_dir: str = ".",
        verbose: int = 0,
        population_size: int = 100,
        sample_size: int = 10,
        **kwargs,
    ):
        super().__init__(
            problem,
            evaluator,
            random_state,
            log_dir,
            verbose,
            population_size,
            sample_size,
        )

        # Setup
        na_search_space = self._problem.build_search_space()

        self.hp_space = self._problem._hp_space  #! hyperparameters
        self.hp_size = len(self.hp_space.space.get_hyperparameter_names())
        self.na_space = HpProblem(self._problem.seed)
        for i, vnode in enumerate(na_search_space.variable_nodes):
            self.na_space.add_hyperparameter(
                (0, vnode.num_ops - 1), name=f"vnode_{i:05d}"
            )

        self._space = CS.ConfigurationSpace(seed=self._problem.seed)
        self._space.add_configuration_space(
            prefix="1", configuration_space=self.hp_space.space
        )
        self._space.add_configuration_space(
            prefix="2", configuration_space=self.na_space.space
        )
        self._space_size = len(self._space.get_hyperparameter_names())

    def saved_keys(self, job):

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
        self._evaluator.submit(self.gen_random_batch(size=self._evaluator.num_workers))

        # Main loop
        while max_evals < 0 or num_evals_done < max_evals:

            # Collecting finished evaluations
            new_results = list(self._evaluator.gather("BATCH", size=1))
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
                        indexes = np.random.choice(
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

                    new_batch = self.gen_random_batch(size=num_received)

                    self._evaluator.submit(new_batch)

    def select_parent(self, sample: list) -> dict:
        cfg, _ = max(sample, key=lambda x: x[1])
        return cfg

    def gen_random_batch(self, size: int) -> list:

        sample = lambda hp, size: [hp.sample(self._space.random) for _ in range(size)]
        batch = []
        iterator = zip(*(sample(hp, size) for hp in self._space.get_hyperparameters()))

        for x in iterator:
            cfg = self._problem.gen_config(
                list(x[self.hp_size :]), list(x[: self.hp_size])
            )
            batch.append(cfg)

        return batch

    def copy_mutate_arch(self, parent_cfg: dict) -> dict:
        """
        # ! Time performance is critical because called sequentialy

        Args:
            parent_arch (list(int)): embedding of the parent's architecture.

        Returns:
            dict: embedding of the mutated architecture of the child.

        """

        hp_x = self._problem.extract_hp_values(parent_cfg)
        x = hp_x + parent_cfg["arch_seq"]
        i = np.random.choice(self._space_size)
        hp = self._space.get_hyperparameters()[i]
        x[i] = hp.sample(self._space.random)

        child_cfg = self._problem.gen_config(x[self.hp_size :], x[: self.hp_size])

        return child_cfg
