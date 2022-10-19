import collections

import deephyper.skopt
import numpy as np
from deephyper.search.nas._regevo import RegularizedEvolution

# Adapt minimization -> maximization with DeepHyper
MAP_liar_strategy = {
    "cl_min": "cl_max",
    "cl_max": "cl_min",
}
MAP_acq_func = {
    "UCB": "LCB",
}


class AgEBO(RegularizedEvolution):
    """`Aging evolution with Bayesian Optimization <https://arxiv.org/abs/2010.16358>`_.

    This algorithm build on the `Regularized Evolution <https://arxiv.org/abs/1802.01548>`_. It cumulates Hyperparameter optimization with Bayesian optimisation and Neural architecture search with regularized evolution.

    Args:
        problem (NaProblem): Neural architecture search problem describing the search space to explore.
        evaluator (Evaluator): An ``Evaluator`` instance responsible of distributing the tasks.
        random_state (int, optional): Random seed. Defaults to None.
        log_dir (str, optional): Log directory where search's results are saved. Defaults to ".".
        verbose (int, optional): Indicate the verbosity level of the search. Defaults to 0.
        population_size (int, optional): the number of individuals to keep in the population. Defaults to ``100``.
        sample_size (int, optional): the number of individuals that should participate in each tournament. Defaults to ``10``.
        n_initial_points (int, optional): Number of collected objectives required before fitting the surrogate-model. Defaults to ``10``.
        initial_points (List[Dict], optional): A list of initial points to evaluate where each point is a dictionnary where keys are names of hyperparameters and values their corresponding choice. Defaults to ``None`` for them to be generated randomly from the search space.
        surrogate_model (str, optional): Surrogate model used by the Bayesian optimization. Can be a value in ``["RF", "ET", "GBRT", "DUMMY"]``. Defaults to ``"RF"``.
        acq_func (str, optional): Acquisition function used by the Bayesian optimization. Can be a value in ``["UCB", "EI", "PI", "gp_hedge"]``. Defaults to ``"UCB"``.
        kappa (float, optional): Manage the exploration/exploitation tradeoff for the "UCB" acquisition function. Defaults to ``0.001`` for strong exploitation.
        xi (float, optional): Manage the exploration/exploitation tradeoff of ``"EI"`` and ``"PI"`` acquisition function. Defaults to ``0.000001`` for strong exploitation.
        n_points (int, optional): The number of configurations sampled from the search space to infer each batch of new evaluated configurations. Defaults to ``10000``.
        liar_strategy (str, optional): Definition of the constant value use for the Liar strategy. Can be a value in ``["cl_min", "cl_mean", "cl_max"]`` . Defaults to ``"cl_max"``.
        n_jobs (int, optional): Number of parallel processes used to fit the surrogate model of the Bayesian optimization. A value of ``-1`` will use all available cores. Defaults to ``1``.
        sync_communcation (bool, optional): Performs the search in a batch-synchronous manner. Defaults to ``False`` for asynchronous updates.
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
        n_initial_points: int = 10,
        initial_points=None,
        surrogate_model: str = "RF",
        acq_func: str = "UCB",
        kappa: float = 0.001,
        xi: float = 0.000001,
        n_points: int = 10000,
        liar_strategy: str = "cl_max",
        n_jobs: int = 1,
        sync_communication: bool = False,
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

        # Initialize opitmizer of hyperparameter space
        if len(self._problem._hp_space._space) == 0:
            raise ValueError(
                "No hyperparameter space was defined for this problem use 'RegularizedEvolution' instead!"
            )

        # check input parameters
        surrogate_model_allowed = ["RF", "ET", "GBRT", "DUMMY"]
        if not (surrogate_model in surrogate_model_allowed):
            raise ValueError(
                f"Parameter 'surrogate_model={surrogate_model}' should have a value in {surrogate_model_allowed}!"
            )

        acq_func_allowed = ["UCB", "EI", "PI", "gp_hedge"]
        if not (acq_func in acq_func_allowed):
            raise ValueError(
                f"Parameter 'acq_func={acq_func}' should have a value in {acq_func_allowed}!"
            )

        if not (np.isscalar(kappa)):
            raise ValueError(f"Parameter 'kappa' should be a scalar value!")

        if not (np.isscalar(xi)):
            raise ValueError("Parameter 'xi' should be a scalar value!")

        if not (type(n_points) is int):
            raise ValueError("Parameter 'n_points' shoud be an integer value!")

        liar_strategy_allowed = ["cl_min", "cl_mean", "cl_max"]
        if not (liar_strategy in liar_strategy_allowed):
            raise ValueError(
                f"Parameter 'liar_strategy={liar_strategy}' should have a value in {liar_strategy_allowed}!"
            )

        if not (type(n_jobs) is int):
            raise ValueError(f"Parameter 'n_jobs' should be an integer value!")

        self._n_initial_points = n_initial_points
        self._initial_points = []
        if initial_points is not None and len(initial_points) > 0:
            for point in initial_points:
                if isinstance(point, list):
                    self._initial_points.append(point)
                elif isinstance(point, dict):
                    self._initial_points.append(
                        [point[hp_name] for hp_name in problem.hyperparameter_names]
                    )
                else:
                    raise ValueError(
                        f"Initial points should be dict or list but {type(point)} was given!"
                    )
        self._liar_strategy = MAP_liar_strategy.get(liar_strategy, liar_strategy)

        base_estimator = self._get_surrogate_model(
            surrogate_model, n_jobs, random_state=self._random_state.randint(0, 2**32)
        )
        self._hp_opt = None
        self._hp_opt_kwargs = dict(
            acq_optimizer="sampling",
            acq_optimizer_kwargs={
                "n_points": n_points,
                "filter_duplicated": False,
            },
            dimensions=self._problem._hp_space._space,
            base_estimator=base_estimator,
            acq_func=MAP_acq_func.get(acq_func, acq_func),
            acq_func_kwargs={"xi": xi, "kappa": kappa},
            n_initial_points=self._n_initial_points,
            initial_points=self._initial_points,
            random_state=self._random_state,
        )

        self._gather_type = "ALL" if sync_communication else "BATCH"

    def _setup_hp_optimizer(self):
        self._hp_opt = deephyper.skopt.Optimizer(**self._hp_opt_kwargs)

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

        if self._hp_opt is None:
            self._setup_hp_optimizer()

        num_evals_done = 0
        population = collections.deque(maxlen=self._population_size)

        # Filling available nodes at start
        batch = self._gen_random_batch(size=self._evaluator.num_workers)
        self._evaluator.submit(batch)

        # Main loop
        while max_evals < 0 or num_evals_done < max_evals:

            # Collecting finished evaluations
            new_results = list(self._evaluator.gather(self._gather_type, size=1))

            if len(new_results) > 0:
                population.extend(new_results)

                self._evaluator.dump_evals(
                    saved_keys=self._saved_keys, log_dir=self._log_dir
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
                        indexes = self._random_state.choice(
                            self._population_size, self._sample_size, replace=False
                        )
                        sample = [population[i] for i in indexes]

                        # select_parent
                        parent = self._select_parent(sample)

                        # copy_mutate_parent
                        child = self._copy_mutate_arch(parent)

                        # add child to batch
                        children_batch.append(child)

                        # collect infos for hp optimization
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

                    new_batch = self._gen_random_batch(
                        size=len(new_results), hps=new_hps
                    )
                    self._evaluator.submit(new_batch)

    def _gen_random_batch(self, size: int, hps: list = None) -> list:
        batch = []
        if hps is None:
            points = self._hp_opt.ask(n_points=size)
            for hp_values in points:
                arch_seq = self._random_search_space()
                config = self._problem.gen_config(arch_seq, hp_values)
                batch.append(config)
        else:  # passed hps are used
            assert size == len(hps)
            for hp_values in hps:
                arch_seq = self._random_search_space()
                config = self._problem.gen_config(arch_seq, hp_values)
                batch.append(config)
        return batch

    def _copy_mutate_arch(self, parent_arch: list) -> list:
        """
        # ! Time performance is critical because called sequentialy

        Args:
            parent_arch (list(int)): embedding of the parent's architecture.

        Returns:
            dict: embedding of the mutated architecture of the child.

        """
        i = self._random_state.choice(len(parent_arch))
        child_arch = parent_arch[:]

        range_upper_bound = self.space_list[i][1]
        elements = [j for j in range(range_upper_bound + 1) if j != child_arch[i]]

        # The mutation has to create a different search_space!
        sample = self._random_state.choice(elements, 1)[0]

        child_arch[i] = sample
        return child_arch

    def _get_surrogate_model(
        self, name: str, n_jobs: int = None, random_state: int = None
    ):
        """Get a surrogate model from Scikit-Optimize.

        Args:
            name (str): name of the surrogate model.
            n_jobs (int): number of parallel processes to distribute the computation of the surrogate model.

        Raises:
            ValueError: when the name of the surrogate model is unknown.
        """
        accepted_names = ["RF", "ET", "GBRT", "DUMMY"]
        if not (name in accepted_names):
            raise ValueError(
                f"Unknown surrogate model {name}, please choose among {accepted_names}."
            )

        if name == "RF":
            surrogate = deephyper.skopt.learning.RandomForestRegressor(
                n_jobs=n_jobs, random_state=random_state
            )
        elif name == "ET":
            surrogate = deephyper.skopt.learning.ExtraTreesRegressor(
                n_jobs=n_jobs, random_state=random_state
            )
        elif name == "GBRT":
            surrogate = deephyper.skopt.learning.GradientBoostingQuantileRegressor(
                n_jobs=n_jobs, random_state=random_state
            )
        else:  # for DUMMY and GP
            surrogate = name

        return surrogate
