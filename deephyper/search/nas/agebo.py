import collections

import numpy as np
import skopt

from deephyper.core.parser import str2bool
from deephyper.search.nas.regevo import RegularizedEvolution


class AgEBO(RegularizedEvolution):
    """Aging evolution with Bayesian Optimization.

    This algorithm build on the 'Regularized Evolution' from https://arxiv.org/abs/1802.01548. It cumulates Hyperparameter optimization with bayesian optimisation and Neural architecture search with regularized evolution.
    """

    def __init__(
        self,
        problem,
        evaluator,
        random_state: int = None,
        log_dir: str = ".",
        verbose: int = 0,
        # RE
        population_size: int = 100,
        sample_size: int = 10,
        # BO
        surrogate_model: str = "RF",
        n_jobs: int = 1,
        kappa: float = 0.001,
        xi: float = 0.000001,
        acq_func: str = "LCB",
        liar_strategy: str = "cl_min",
        mode: str = "async",
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

        assert mode in ["sync", "async"]
        self.mode = mode

        self.n_jobs = int(n_jobs)  # parallelism of BO surrogate model estimator

        # Initialize opitmizer of hyperparameter space
        self._n_initial_points = self._evaluator.num_workers
        self._liar_strategy = liar_strategy

        self._hp_opt = None
        self._hp_opt_kwargs = dict(
            dimensions=self._problem._hp_space._space,
            base_estimator=self.get_surrogate_model(surrogate_model, n_jobs),
            acq_func=acq_func,
            acq_optimizer="sampling",
            acq_func_kwargs={"xi": float(xi), "kappa": float(kappa)},
            n_initial_points=self._n_initial_points,
            random_state=self._random_state,
        )

    def _setup_hp_optimizer(self):
        self._hp_opt = skopt.Optimizer(**self._hp_opt_kwargs)

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

        if self._hp_opt is None:
            self._setup_hp_optimizer()

        num_evals_done = 0
        population = collections.deque(maxlen=self._population_size)

        # Filling available nodes at start
        self._evaluator.submit(self.gen_random_batch(size=self._n_initial_points))

        # Main loop
        while max_evals < 0 or num_evals_done < max_evals:

            # Collecting finished evaluations
            if self.mode == "async":
                new_results = list(self._evaluator.gather("BATCH", size=1))
            else:
                new_results = list(self._evaluator.gather("ALL"))

            if len(new_results) > 0:
                population.extend(new_results)

                self._evaluator.dump_evals(
                    saved_keys=self.saved_keys, log_dir=self._log_dir
                )

                num_received = len(new_results)
                num_evals_done += num_received

                hp_results_X, hp_results_y = [], []

                # If the population is big enough evolve the population
                if len(population) == self._population_size:
                    children_batch = []

                    # For each new parent/result we create a child from it
                    for new_i in range(len(new_results)):
                        # select_sample
                        indexes = np.random.choice(
                            self._population_size, self._sample_size, replace=False
                        )
                        sample = [population[i] for i in indexes]

                        # select_parent
                        parent = self.select_parent(sample)

                        # copy_mutate_parent
                        child = self.copy_mutate_arch(parent)

                        # add child to batch
                        children_batch.append(child)

                        # collect infos for hp optimization
                        new_i_hp_values = self._problem.extract_hp_values(
                            config=new_results[new_i][0]
                        )
                        new_i_y = new_results[new_i][1]
                        hp_results_X.append(new_i_hp_values)
                        hp_results_y.append(-new_i_y)

                    hp_results_y = np.minimum(hp_results_y, 1e3).tolist()  #! TODO

                    self._hp_opt.tell(hp_results_X, hp_results_y)  #! fit: costly
                    new_hps = self._hp_opt.ask(
                        n_points=len(new_results), strategy=self._liar_strategy
                    )

                    new_configs = []
                    for hp_values, child_arch_seq in zip(new_hps, children_batch):
                        new_config = self._problem.gen_config(child_arch_seq, hp_values)
                        new_configs.append(new_config)

                    # submit_childs
                    if len(new_results) > 0:
                        self._evaluator.submit(new_configs)

                else:  # If the population is too small keep increasing it

                    # For each new parent/result we create a child from it
                    for new_i in range(len(new_results)):

                        new_i_hp_values = self._problem.extract_hp_values(
                            config=new_results[new_i][0]
                        )
                        new_i_y = new_results[new_i][1]
                        hp_results_X.append(new_i_hp_values)
                        hp_results_y.append(-new_i_y)

                    self._hp_opt.tell(hp_results_X, hp_results_y)  #! fit: costly
                    new_hps = self._hp_opt.ask(
                        n_points=len(new_results), strategy=self._liar_strategy
                    )

                    new_batch = self.gen_random_batch(size=len(new_results), hps=new_hps)

                    self._evaluator.submit(new_batch)

    def gen_random_batch(self, size: int, hps: list = None) -> list:
        batch = []
        if hps is None:
            points = self._hp_opt.ask(n_points=size)
            for hp_values in points:
                arch_seq = self.random_search_space()
                config = self._problem.gen_config(arch_seq, hp_values)
                batch.append(config)
        else:  # passed hps are used
            assert size == len(hps)
            for hp_values in hps:
                arch_seq = self.random_search_space()
                config = self._problem.gen_config(arch_seq, hp_values)
                batch.append(config)
        return batch

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

    def get_surrogate_model(self, name: str, n_jobs: int = None):
        """Get a surrogate model from Scikit-Optimize.

        Args:
            name (str): name of the surrogate model.
            n_jobs (int): number of parallel processes to distribute the computation of the surrogate model.

        Raises:
            ValueError: when the name of the surrogate model is unknown.
        """
        accepted_names = ["RF", "ET", "GBRT", "GP", "DUMMY"]
        if not (name in accepted_names):
            raise ValueError(
                f"Unknown surrogate model {name}, please choose among {accepted_names}."
            )

        if name == "RF":
            surrogate = skopt.learning.RandomForestRegressor(n_jobs=n_jobs)
        elif name == "ET":
            surrogate = skopt.learning.ExtraTreesRegressor(n_jobs=n_jobs)
        elif name == "GBRT":
            surrogate = skopt.learning.GradientBoostingQuantileRegressor(n_jobs=n_jobs)
        else:  # for DUMMY and GP
            surrogate = name

        return surrogate
