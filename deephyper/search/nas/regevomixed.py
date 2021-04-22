import collections
import copy

import ConfigSpace as CS
import numpy as np

from deephyper.problem import HpProblem
from deephyper.core.logs.logging import JsonMessage as jm
from deephyper.core.parser import add_arguments_from_signature
from deephyper.search import util
from deephyper.search.nas.regevo import RegularizedEvolution

dhlogger = util.conf_logger("deephyper.search.regevomixed")


class RegularizedEvolutionMixed(RegularizedEvolution):

    def __init__(
        self,
        problem,
        run,
        evaluator,
        population_size=100,
        sample_size=10,
        n_jobs=1,
        **kwargs,
    ):
        super().__init__(
            problem=problem,
            run=run,
            evaluator=evaluator,
            population_size=population_size,
            sample_size=sample_size,
            **kwargs,
        )

        self.n_jobs = int(n_jobs)

        # Setup
        na_search_space = self.problem.build_search_space()

        self.hp_space = self.problem._hp_space  #! hyperparameters
        self.hp_size = len(self.hp_space.space.get_hyperparameter_names())
        self.na_space = HpProblem(self.problem.seed)
        for i, vnode in enumerate(na_search_space.variable_nodes):
            self.na_space.add_hyperparameter(
                (0, vnode.num_ops - 1), name=f"vnode_{i:05d}"
            )

        self.space = CS.ConfigurationSpace(seed=self.problem.seed)
        self.space.add_configuration_space(
            prefix="1", configuration_space=self.hp_space.space
        )
        self.space.add_configuration_space(
            prefix="2", configuration_space=self.na_space.space
        )

    @staticmethod
    def _extend_parser(parser):
        RegularizedEvolution._extend_parser(parser)
        add_arguments_from_signature(parser, RegularizedEvolutionMixed)
        return parser

    def saved_keys(self, val: dict):
        res = {"id": val["id"], "arch_seq": str(val["arch_seq"])}
        hp_names = self.hp_space._space.get_hyperparameter_names()

        for hp_name in hp_names:
            if hp_name == "loss":
                res["loss"] = val["loss"]
            else:
                res[hp_name] = val["hyperparameters"][hp_name]

        return res

    def main(self):

        num_evals_done = 0
        population = collections.deque(maxlen=self.population_size)

        # Filling available nodes at start
        self.evaluator.add_eval_batch(self.gen_random_batch(size=self.free_workers))

        # Main loop
        while num_evals_done < self.max_evals:

            # Collecting finished evaluations
            new_results = list(self.evaluator.get_finished_evals())

            if len(new_results) > 0:
                population.extend(new_results)
                stats = {"num_cache_used": self.evaluator.stats["num_cache_used"]}
                dhlogger.info(jm(type="env_stats", **stats))
                self.evaluator.dump_evals(saved_keys=self.saved_keys)

                num_received = len(new_results)
                num_evals_done += num_received

                # If the population is big enough evolve the population
                if len(population) == self.population_size:
                    children_batch = []

                    # For each new parent/result we create a child from it
                    for _ in range(len(new_results)):
                        # select_sample
                        indexes = np.random.choice(
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
                        self.evaluator.add_eval_batch(children_batch)
                else:  # If the population is too small keep increasing it

                    new_batch = self.gen_random_batch(size=len(new_results))

                    self.evaluator.add_eval_batch(new_batch)

    def select_parent(self, sample: list) -> dict:
        cfg, _ = max(sample, key=lambda x: x[1])
        return cfg

    def gen_random_batch(self, size: int) -> list:

        sample = lambda hp, size: [hp.sample(self.space.random) for _ in range(size)]
        batch = []
        iterator = zip(*(sample(hp,size) for hp in self.space.get_hyperparameters()))

        for x in iterator:
            #! DeepCopy is critical for nested lists or dicts
            cfg = self.problem.gen_config(x[self.hp_size :], x[: self.hp_size])
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

        hp_x = self.problem.extract_hp_values(parent_cfg)
        x = hp_x + parent_cfg["arch_seq"]
        i = np.random.choice(self.hp_size)
        hp = self.space.get_hyperparameters()[i]
        x[i] = hp.sample(self.space.random)

        child_cfg = self.problem.gen_config(x[self.hp_size :], x[: self.hp_size])

        return child_cfg


if __name__ == "__main__":
    args = RegularizedEvolutionMixed.parse_args()
    search = RegularizedEvolutionMixed(**vars(args))
    search.main()
