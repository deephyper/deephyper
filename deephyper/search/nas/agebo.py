import collections
import json
import os
import copy

# import ConfigSpace as cs
import numpy as np
from skopt import Optimizer as SkOptimizer
from skopt.learning import (
    GradientBoostingQuantileRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor,
)
from skopt.acquisition import gaussian_lcb
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from deephyper.core.logs.logging import JsonMessage as jm
from deephyper.core.parser import add_arguments_from_signature
from deephyper.evaluator.evaluate import Encoder
from deephyper.problem.base import check_hyperparameter
from deephyper.search import util
from deephyper.search.nas.regevo import RegularizedEvolution

dhlogger = util.conf_logger("deephyper.search.nas.agebo")

# def key(d):
#     return json.dumps(dict(arch_seq=d['arch_seq']), cls=Encoder)


class AgeBO(RegularizedEvolution):
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
        self,
        problem,
        run,
        evaluator,
        population_size=100,
        sample_size=10,
        plot="true",
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

        self.do_plot = plot == "true"
        self.n_jobs = int(n_jobs)

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
        self.hp_space.append(self.problem.space["hyperparameters"]["learning_rate"])

        # ploting
        lr_range = self.problem.space["hyperparameters"]["learning_rate"][:2]
        self.domain_x = np.linspace(*lr_range, 400).reshape(-1, 1)
        # self.hp_space.append(self.problem.space["hyperparameters"]["batch_size"][:2])

        # if surrogate_model == "RF":
        #     base_estimator = RandomForestRegressor(n_jobs=n_jobs)
        # elif surrogate_model == "ET":
        #     base_estimator = ExtraTreesRegressor(n_jobs=n_jobs)
        # elif surrogate_model == "GBRT":
        #     base_estimator = GradientBoostingQuantileRegressor(n_jobs=n_jobs)
        # else:
        #     base_estimator = surrogate_model

        # Initialize opitmizer of hyperparameter space
        acq_func_kwargs = {"xi": 0.000001, "kappa": 0.001}  # tiny exploration
        self.n_initial_points = self.free_workers

        self.hp_opt = SkOptimizer(
            dimensions=self.hp_space,
            # base_estimator=GradientBoostingQuantileRegressor(n_jobs=4),
            base_estimator=RandomForestRegressor(n_jobs=32),
            # base_estimator=RandomForestRegressor(n_jobs=self.n_jobs),
            acq_func="LCB",  # "gp_hedge",
            acq_optimizer="sampling",
            acq_func_kwargs=acq_func_kwargs,
            n_initial_points=self.n_initial_points,  # Half Random - Half advised
            # model_queue_size=100,
        )

    @staticmethod
    def _extend_parser(parser):
        RegularizedEvolution._extend_parser(parser)
        add_arguments_from_signature(parser, AgeBO)
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
        it = 0
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
                        # hp_new_i = [new_i_hps["learning_rate"], new_i_hps["batch_size"]]
                        hp_new_i = [new_i_hps["learning_rate"]]
                        hp_results_X.append(hp_new_i)
                        hp_results_y.append(-new_i_y)

                    self.hp_opt.tell(hp_results_X, hp_results_y)  #! fit: costly
                    new_hps = self.hp_opt.ask(n_points=len(new_results))

                    for hps, child in zip(new_hps, children_batch):
                        child["hyperparameters"]["learning_rate"] = hps[0]
                        # child["hyperparameters"]["batch_size"] = hps[1]

                    # submit_childs
                    if len(new_results) > 0:
                        self.evaluator.add_eval_batch(children_batch)
                else:  # If the population is too small keep increasing it

                    # For each new parent/result we create a child from it
                    for new_i in range(len(new_results)):

                        new_i_hps = new_results[new_i][0]["hyperparameters"]
                        new_i_y = new_results[new_i][1]
                        # hp_new_i = [new_i_hps["learning_rate"], new_i_hps["batch_size"]]
                        hp_new_i = [new_i_hps["learning_rate"]]
                        hp_results_X.append(hp_new_i)
                        hp_results_y.append(-new_i_y)

                    self.hp_opt.tell(hp_results_X, hp_results_y)  #! fit: costly
                    new_hps = self.hp_opt.ask(n_points=len(new_results))

                    new_batch = self.gen_random_batch(size=len(new_results), hps=new_hps)

                    self.evaluator.add_eval_batch(new_batch)

                try:
                    self.plot_optimizer(x=self.domain_x, it=it)
                    it += 1
                except:
                    pass

    # def ask(self, n_points=None, batch_size=20):
    #     if n_points is None:
    #         return self._ask()
    #     else:
    #         batch = []
    #         for _ in range(n_points):
    #             batch.append(self._ask())
    #             if len(batch) == batch_size:
    #                 yield batch
    #                 batch = []
    #         if batch:
    #             yield batch

    def select_parent(self, sample: list) -> list:
        cfg, _ = max(sample, key=lambda x: x[1])
        return cfg["arch_seq"]

    def gen_random_batch(self, size: int, hps: list = None) -> list:
        batch = []
        if hps is None:
            points = self.hp_opt.ask(n_points=size)
            for point in points:
                #! DeepCopy is critical for nested lists or dicts
                cfg = copy.deepcopy(self.pb_dict)

                # hyperparameters
                cfg["hyperparameters"]["learning_rate"] = point[0]
                # cfg["hyperparameters"]["batch_size"] = point[1]

                # architecture dna
                cfg["arch_seq"] = self.random_search_space()
                batch.append(cfg)
        else:  # passed hps are used
            assert size == len(hps)
            for point in hps:
                #! DeepCopy is critical for nested lists or dicts
                cfg = copy.deepcopy(self.pb_dict)

                # hyperparameters
                cfg["hyperparameters"]["learning_rate"] = point[0]
                # cfg["hyperparameters"]["batch_size"] = point[1]

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
        cfg = self.pb_dict.copy()
        cfg["arch_seq"] = child_arch
        return cfg

    def plot_optimizer(self, x, it=0):
        opt = self.hp_opt
        model = opt.models[-1]
        x_model = opt.space.transform(x.tolist())

        plt.figure(figsize=(6.4 * 2, 4.8))
        plt.subplot(1, 2, 1)
        # Plot Model(x) + contours
        y_pred, sigma = model.predict(x_model, return_std=True)
        y_pred *= -1
        plt.plot(x, y_pred, "g--", label=r"$\mu(x)$")
        plt.fill(
            np.concatenate([x, x[::-1]]),
            np.concatenate([y_pred - 1.9600 * sigma, (y_pred + 1.9600 * sigma)[::-1]]),
            alpha=0.2,
            fc="g",
            ec="None",
        )

        # Plot sampled points
        W = 10
        yi = np.array(opt.yi)[-W:] * -1
        Xi = opt.Xi[-W:]
        plt.plot(Xi, yi, "r.", markersize=8, label="Observations")

        plt.grid()
        plt.legend(loc="best")
        plt.xlim(0.001, 0.1)
        plt.ylim(0, 1)
        plt.xlabel("Learning Rate")
        plt.ylabel("Objective")
        plt.xscale("log")

        ax = plt.gca()
        ax.xaxis.set_major_locator(ticker.FixedLocator([0.001, 0.01, 0.1]))
        ax.xaxis.set_major_formatter(ticker.FixedFormatter(["0.001", "0.01", "0.1"]))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))

        # LCB
        plt.subplot(1, 2, 2)
        acq = gaussian_lcb(x_model, model) * -1
        plt.plot(x, acq, "b", label="UCB(x)")
        plt.fill_between(x.ravel(), 0.0, acq.ravel(), alpha=0.3, color="blue")

        plt.xlabel("Learning Rate")

        # Adjust plot layout
        plt.grid()
        plt.legend(loc="best")
        plt.xlim(0.001, 0.1)
        plt.ylim(0, 1)
        plt.xscale("log")

        ax = plt.gca()
        ax.xaxis.set_major_locator(ticker.FixedLocator([0.001, 0.01, 0.1]))
        ax.xaxis.set_major_formatter(ticker.FixedFormatter(["0.001", "0.01", "0.1"]))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))

        # Save Figure
        plt.savefig(f"opt-{it:05}.png", dpi=100)
        plt.close()


if __name__ == "__main__":
    args = AgeBO.parse_args()
    search = AgeBO(**vars(args))
    search.main()
