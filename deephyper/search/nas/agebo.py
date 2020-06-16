import collections
import json
import os

# import ConfigSpace as cs
import numpy as np
from skopt import Optimizer as SkOptimizer
from skopt.learning import GradientBoostingQuantileRegressor, RandomForestRegressor

from deephyper.core.logs.logging import JsonMessage as jm
from deephyper.core.parser import add_arguments_from_signature
from deephyper.evaluator.evaluate import Encoder
from deephyper.problem.base import check_hyperparameter
from deephyper.search import util
from deephyper.search.nas.regevo import RegularizedEvolution

dhlogger = util.conf_logger("deephyper.search.nas.agebo")

# def key(d):
#     return json.dumps(dict(arch_seq=d['arch_seq']), cls=Encoder)


class ExtendedRegularizedEvolution(RegularizedEvolution):
    """Regularized evolution.

    https://arxiv.org/abs/1802.01548

    Args:
        problem (str): Module path to the Problem instance you want to use for the search (e.g. deephyper.benchmark.nas.linearReg.Problem).
        run (str): Module path to the run function you want to use for the search (e.g. deephyper.search.nas.model.run.quick).
        evaluator (str): value in ['balsam', 'subprocess', 'processPool', 'threadPool'].
        population_size (int, optional): the number of individuals to keep in the population. Defaults to 100.
        sample_size (int, optional): the number of individuals that should participate in each tournament. Defaults to 10.
    """

    def __init__(
        self, problem, run, evaluator, population_size=100, sample_size=10, **kwargs
    ):
        super().__init__(
            problem=problem,
            run=run,
            evaluator=evaluator,
            population_size=population_size,
            sample_size=sample_size,
            **kwargs
        )

        # Initialize Hyperaparameter space
        # self.hp_space = cs.ConfigurationSpace(seed=42)
        # self.hp_space.add_hyperparameter(
        #     check_hyperparameter(
        #         self.problem.space["hyperparameters"]["learning_rate"], "learning_rate"
        #     )
        # )
        # self.hp_space.add_hyperparameter(
        #     check_hyperparameter(
        #         self.problem.space["hyperparameters"]["batch_size"], "batch_size"
        #     )
        # )

        self.hp_space = []
        self.hp_space.append(self.problem.space["hyperparameters"]["learning_rate"][:2])
        self.hp_space.append(self.problem.space["hyperparameters"]["batch_size"][:2])

        self.init_hp = {
            "learning_rate": self.problem.space["hyperparameters"]["learning_rate"][2],
            "batch_size": self.problem.space["hyperparameters"]["batch_size"][2],
        }

        # if surrogate_model == "RF":
        #     base_estimator = RandomForestRegressor(n_jobs=n_jobs)
        # elif surrogate_model == "ET":
        #     base_estimator = ExtraTreesRegressor(n_jobs=n_jobs)
        # elif surrogate_model == "GBRT":
        #     base_estimator = GradientBoostingQuantileRegressor(n_jobs=n_jobs)
        # else:
        #     base_estimator = surrogate_model

        # Initialize opitmizer of hyperparameter space
        # acq_func_kwargs = {"xi": 0.000001, "kappa": 0.001}
        self.prop_advised = 0.5
        self.n_initial_points = int(self.free_workers * self.prop_advised)
        self.hp_opt = SkOptimizer(
            dimensions=self.hp_space,
            base_estimator=GradientBoostingQuantileRegressor(n_jobs=n_jobs),
            acq_func="gp_hedge",
            acq_optimizer="sampling",
            n_initial_points=self.n_initial_points,  # Half Random - Half advised
            # model_queue_size=100,
        )

    @staticmethod
    def _extend_parser(parser):
        RegularizedEvolution._extend_parser(parser)
        add_arguments_from_signature(parser, ExtendedRegularizedEvolution)
        return parser

    def saved_keys(self, val: dict):
        res = {
            "learning_rate": val["hyperparameters"]["learning_rate"],
            "batch_size": val["hyperparameters"]["batch_size"],
            "arch_seq": str(val["arch_seq"]),
        }
        return res

    def main(self):

        num_evals_done = 0
        # num_cyles_done = 0
        population = collections.deque(maxlen=self.population_size)

        # best_y = 0  #! dum value
        # patience_limit_best_y = 50

        # Filling available nodes at start
        self.evaluator.add_eval_batch(
            self.gen_random_batch(size=self.free_workers, prop=self.prop_advised)
        )

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

                        # hpo
                        # collect infos for hp optimization
                        new_i_hps = new_results[new_i][0]["hyperparameters"]
                        new_i_y = new_results[new_i][1]
                        hp_new_i = [new_i_hps["learning_rate"], new_i_hps["batch_size"]]
                        hp_results_X.append(hp_new_i)
                        hp_results_y.append(-new_i_y)

                    self.hp_opt.tell(hp_results_X, hp_results_y)  #! fit: costly
                    new_hps = self.hp_opt.ask(n_points=len(new_results))

                    for hps, child in zip(new_hps, children_batch):
                        child["hyperparameters"]["learning_rate"] = hps[0]
                        child["hyperparameters"]["batch_size"] = hps[1]

                    # submit_childs
                    if len(new_results) > 0:
                        self.evaluator.add_eval_batch(children_batch)
                else:  # If the population is too small keep increasing it

                    # For each new parent/result we create a child from it
                    for new_i in range(len(new_results)):

                        new_i_hps = new_results[new_i][0]["hyperparameters"]
                        new_i_y = new_results[new_i][1]
                        hp_new_i = [new_i_hps["learning_rate"], new_i_hps["batch_size"]]
                        hp_results_X.append(hp_new_i)
                        hp_results_y.append(-new_i_y)

                    self.hp_opt.tell(hp_results_X, hp_results_y)  #! fit: costly
                    new_hps = self.hp_opt.ask(n_points=len(new_results))

                    new_batch = self.gen_random_batch(size=len(new_results), hps=new_hps)

                    self.evaluator.add_eval_batch(new_batch)

    def select_parent(self, sample: list) -> list:
        cfg, _ = max(sample, key=lambda x: x[1])
        return cfg["arch_seq"]

    def gen_random_batch(self, size: int, prop: float = 1.0, hps: list = None) -> list:
        batch = []
        if hps is None:
            n_random = int(size * prop)
            for _ in range(n_random):
                cfg = self.pb_dict.copy()
                # hyperparameters
                point = self.hp_opt.ask(n_points=1)[0]
                cfg["hyperparameters"]["learning_rate"] = point[0]
                cfg["hyperparameters"]["batch_size"] = point[1]

                # architecture dna
                cfg["arch_seq"] = self.random_search_space()
                batch.append(cfg)
            for _ in range(size - n_random):
                cfg = self.pb_dict.copy()
                # hyperparameters
                cfg["hyperparameters"]["learning_rate"] = self.init_hp["learning_rate"]
                cfg["hyperparameters"]["batch_size"] = self.init_hp["batch_size"]

                # architecture dna
                cfg["arch_seq"] = self.random_search_space()
                batch.append(cfg)
        else:  # passed hps are used
            assert size == len(hps)
            for point in hps:
                cfg = self.pb_dict.copy()
                # hyperparameters
                cfg["hyperparameters"]["learning_rate"] = point[0]
                cfg["hyperparameters"]["batch_size"] = point[1]

                # architecture dna
                cfg["arch_seq"] = self.random_search_space()
                batch.append(cfg)
        return batch

    def random_search_space(self) -> list:
        return [np.random.choice(b + 1) for (_, b) in self.space_list]

    def copy_mutate_arch(self, parent_arch: list) -> dict:
        """
        # ! Time performance is critical because called sequentialy

        Args:
            parent_arch (list(int)): [description]

        Returns:
            dict: [description]

        """
        i = np.random.choice(len(parent_arch))
        child_arch = parent_arch[:]

        range_upper_bound = self.space_list[i][1]
        elements = [j for j in range(range_upper_bound + 1) if j != child_arch[i]]

        # The mutation has to create a different search_space!
        sample = np.random.choice(elements, 1)[0]

        child_arch[i] = sample
        cfg = self.pb_dict.copy()
        cfg["arch_seq"] = child_arch
        return cfg


if __name__ == "__main__":
    args = ExtendedRegularizedEvolution.parse_args()
    search = ExtendedRegularizedEvolution(**vars(args))
    search.main()
