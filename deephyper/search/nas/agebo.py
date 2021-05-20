import collections

import numpy as np
from skopt import Optimizer as SkOptimizer
from skopt.learning import RandomForestRegressor

from deephyper.core.logs.logging import JsonMessage as jm
from deephyper.core.parser import add_arguments_from_signature, str2bool
from deephyper.search import util
from deephyper.search.nas.regevo import RegularizedEvolution

dhlogger = util.conf_logger("deephyper.search.nas.agebo")


class AgEBO(RegularizedEvolution):
    """Aging evolution with Bayesian Optimization.

    This algorithm build on the 'Regularized Evolution' from https://arxiv.org/abs/1802.01548. It cumulates Hyperparameter optimization with bayesian optimisation and Neural architecture search with regularized evolution.

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
        run,
        evaluator,
        population_size=100,
        sample_size=10,
        n_jobs=1,
        kappa=0.001,
        xi=0.000001,
        acq_func="LCB",
        sync=False,
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
        if type(sync) is str:
            sync = str2bool(sync)
        self.mode = "sync" if sync else "async"

        self.n_jobs = int(n_jobs)  # parallelism of BO surrogate model estimator

        # Initialize Hyperaparameter space
        self.hp_space = self.problem._hp_space

        # Initialize opitmizer of hyperparameter space
        acq_func_kwargs = {"xi": float(xi), "kappa": float(kappa)}  # tiny exploration
        self.n_initial_points = self.free_workers

        self.hp_opt = SkOptimizer(
            dimensions=self.hp_space._space,
            base_estimator=RandomForestRegressor(n_jobs=self.n_jobs),
            acq_func=acq_func,
            acq_optimizer="sampling",
            acq_func_kwargs=acq_func_kwargs,
            n_initial_points=self.n_initial_points,
        )

    @staticmethod
    def _extend_parser(parser):
        RegularizedEvolution._extend_parser(parser)
        add_arguments_from_signature(parser, AgEBO)
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
            new_results = list(self.evaluator.get_finished_evals(mode=self.mode))

            if len(new_results) > 0:
                population.extend(new_results)
                stats = {"num_cache_used": self.evaluator.stats["num_cache_used"]}
                dhlogger.info(jm(type="env_stats", **stats))
                self.evaluator.dump_evals(saved_keys=self.saved_keys)

                num_received = len(new_results)
                num_evals_done += num_received

                hp_results_X, hp_results_y = [], []

                # If the population is big enough evolve the population
                if len(population) == self.population_size:
                    children_batch = []

                    # For each new parent/result we create a child from it
                    for new_i in range(len(new_results)):
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

                        # collect infos for hp optimization
                        new_i_hp_values = self.problem.extract_hp_values(
                            config=new_results[new_i][0]
                        )
                        new_i_y = new_results[new_i][1]
                        hp_results_X.append(new_i_hp_values)
                        hp_results_y.append(-new_i_y)

                    hp_results_y = np.minimum(hp_results_y, 1e3).tolist() #! TODO

                    self.hp_opt.tell(hp_results_X, hp_results_y)  #! fit: costly
                    new_hps = self.hp_opt.ask(n_points=len(new_results))

                    new_configs = []
                    for hp_values, child_arch_seq in zip(new_hps, children_batch):
                        new_config = self.problem.gen_config(child_arch_seq, hp_values)
                        new_configs.append(new_config)

                    # submit_childs
                    if len(new_results) > 0:
                        self.evaluator.add_eval_batch(new_configs)
                else:  # If the population is too small keep increasing it

                    # For each new parent/result we create a child from it
                    for new_i in range(len(new_results)):

                        new_i_hp_values = self.problem.extract_hp_values(
                            config=new_results[new_i][0]
                        )
                        new_i_y = new_results[new_i][1]
                        hp_results_X.append(new_i_hp_values)
                        hp_results_y.append(-new_i_y)

                    self.hp_opt.tell(hp_results_X, hp_results_y)  #! fit: costly
                    new_hps = self.hp_opt.ask(n_points=len(new_results))

                    new_batch = self.gen_random_batch(size=len(new_results), hps=new_hps)

                    self.evaluator.add_eval_batch(new_batch)

    def select_parent(self, sample: list) -> list:
        cfg, _ = max(sample, key=lambda x: x[1])
        return cfg["arch_seq"]

    def gen_random_batch(self, size: int, hps: list = None) -> list:
        batch = []
        if hps is None:
            points = self.hp_opt.ask(n_points=size)
            for hp_values in points:
                arch_seq = self.random_search_space()
                config = self.problem.gen_config(arch_seq, hp_values)
                batch.append(config)
        else:  # passed hps are used
            assert size == len(hps)
            for hp_values in hps:
                arch_seq = self.random_search_space()
                config = self.problem.gen_config(arch_seq, hp_values)
                batch.append(config)
        return batch

    def random_search_space(self) -> list:
        return [np.random.choice(b + 1) for (_, b) in self.space_list]

    def copy_mutate_arch(self, parent_arch: list) -> list:
        """
        # ! Time performance is critical because called sequentialy

        Args:
            parent_arch (list(int)): embedding of the parent's architecture.

        Returns:
            dict: embedding of the mutated architecture of the child.

        """
        i = np.random.choice(len(parent_arch))
        child_arch = parent_arch[:]

        range_upper_bound = self.space_list[i][1]
        elements = [j for j in range(range_upper_bound + 1) if j != child_arch[i]]

        # The mutation has to create a different search_space!
        sample = np.random.choice(elements, 1)[0]

        child_arch[i] = sample
        return child_arch


if __name__ == "__main__":
    args = AgEBO.parse_args()
    search = AgEBO(**vars(args))
    search.main()
