import logging

import ConfigSpace as CS
import numpy as np
import deephyper.skopt
from deephyper.problem import HpProblem
from deephyper.search.nas._base import NeuralArchitectureSearch

# Adapt minimization -> maximization with DeepHyper
MAP_liar_strategy = {
    "cl_min": "cl_max",
    "cl_max": "cl_min",
}
MAP_acq_func = {
    "UCB": "LCB",
}


class AMBSMixed(NeuralArchitectureSearch):
    """Asynchronous Model-Based Search baised on the `Scikit-Optimized Optimizer <https://scikit-optimize.github.io/stable/modules/generated/deephyper.skopt.Optimizer.html#deephyper.skopt.Optimizer>`_. It is extended to the case of joint hyperparameter and neural architecture search.

    Args:
        problem (NaProblem): Neural architecture search problem describing the search space to explore.
        evaluator (Evaluator): An ``Evaluator`` instance responsible of distributing the tasks.
        random_state (int, optional): Random seed. Defaults to None.
        log_dir (str, optional): Log directory where search's results are saved. Defaults to ".".
        verbose (int, optional): Indicate the verbosity level of the search. Defaults to 0.
        surrogate_model (str, optional): Surrogate model used by the Bayesian optimization. Can be a value in ``["RF", "ET", "GBRT", "DUMMY"]``. Defaults to ``"RF"``.
        acq_func (str, optional): Acquisition function used by the Bayesian optimization. Can be a value in ``["UCB", "EI", "PI", "gp_hedge"]``. Defaults to ``"UCB"``.
        kappa (float, optional): Manage the exploration/exploitation tradeoff for the "UCB" acquisition function. Defaults to ``1.96`` for a balance between exploitation and exploration.
        xi (float, optional): Manage the exploration/exploitation tradeoff of ``"EI"`` and ``"PI"`` acquisition function. Defaults to ``0.001`` for a balance between exploitation and exploration.
        n_points (int, optional): The number of configurations sampled from the search space to infer each batch of new evaluated configurations. Defaults to ``10000``.
        liar_strategy (str, optional): Definition of the constant value use for the Liar strategy. Can be a value in ``["cl_min", "cl_mean", "cl_max"]`` . Defaults to ``"cl_max"``.
        n_jobs (int, optional): Number of parallel processes used to fit the surrogate model of the Bayesian optimization. A value of ``-1`` will use all available cores. Defaults to ``1``.
    """

    def __init__(
        self,
        problem,
        evaluator,
        random_state=None,
        log_dir=".",
        verbose=0,
        surrogate_model: str = "RF",
        acq_func: str = "UCB",
        kappa: float = 1.96,
        xi: float = 0.001,
        n_points: int = 10000,
        liar_strategy: str = "cl_max",
        n_jobs: int = 1,
        **kwargs,
    ):
        super().__init__(problem, evaluator, random_state, log_dir, verbose)

        # Setup the search space
        na_search_space = self._problem.build_search_space()

        self.hp_space = self._problem._hp_space  # !hyperparameters
        self.hp_size = len(self.hp_space.space.get_hyperparameter_names())
        self.na_space = HpProblem()
        self.na_space._space.seed(self._random_state.get_state()[1][0])
        for i, vnode in enumerate(na_search_space.variable_nodes):
            self.na_space.add_hyperparameter(
                (0, vnode.num_ops - 1), name=f"vnode_{i:05d}"
            )

        self._space = CS.ConfigurationSpace(seed=self._random_state.get_state()[1][0])
        self._space.add_configuration_space(
            prefix="1", configuration_space=self.hp_space.space
        )
        self._space.add_configuration_space(
            prefix="2", configuration_space=self.na_space.space
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
            raise ValueError("Parameter 'kappa' should be a scalar value!")

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
            raise ValueError("Parameter 'n_jobs' should be an integer value!")

        self._n_initial_points = self._evaluator.num_workers
        self._liar_strategy = MAP_liar_strategy.get(liar_strategy, liar_strategy)

        base_estimator = self._get_surrogate_model(
            surrogate_model, n_jobs, random_state=self._random_state.get_state()[1][0]
        )

        self._opt = None
        self._opt_kwargs = dict(
            dimensions=self._space,
            base_estimator=base_estimator,
            acq_func=MAP_acq_func.get(acq_func, acq_func),
            acq_optimizer="sampling",
            acq_func_kwargs={"xi": xi, "kappa": kappa, "n_points": n_points},
            n_initial_points=self._n_initial_points,
            random_state=self._random_state,
        )

    def _setup_optimizer(self):
        self._opt = deephyper.skopt.Optimizer(**self._opt_kwargs)

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

        if self._opt is None:
            self._setup_optimizer()

        num_evals_done = 0

        # Filling available nodes at start
        logging.info(f"Generating {self._evaluator.num_workers} initial points...")
        self._evaluator.submit(self._get_random_batch(size=self._n_initial_points))

        # Main loop
        while max_evals < 0 or num_evals_done < max_evals:

            # Collecting finished evaluations
            new_results = list(self._evaluator.gather("BATCH", size=1))
            num_received = len(new_results)

            if num_received > 0:

                self._evaluator.dump_evals(
                    saved_keys=self._saved_keys, log_dir=self._log_dir
                )
                num_evals_done += num_received

                if num_evals_done >= max_evals:
                    break

                # Transform configurations to list to fit optimizer
                opt_X = []
                opt_y = []
                for cfg, obj in new_results:
                    arch_seq = cfg["arch_seq"]
                    hp_val = self._problem.extract_hp_values(cfg)
                    x = replace_nan(hp_val + arch_seq)
                    opt_X.append(x)
                    opt_y.append(-obj)  # !maximizing

                self._opt.tell(opt_X, opt_y)  # !fit: costly
                new_X = self._opt.ask(
                    n_points=len(new_results), strategy=self._liar_strategy
                )

                new_batch = []
                for x in new_X:
                    new_cfg = self._problem.gen_config(
                        x[self.hp_size :], x[: self.hp_size]
                    )
                    new_batch.append(new_cfg)

                # submit_childs
                if len(new_results) > 0:
                    self._evaluator.submit(new_batch)

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

    def _get_random_batch(self, size: int) -> list:
        batch = []
        n_points = max(0, size - len(batch))
        if n_points > 0:
            points = self._opt.ask(n_points=n_points)
            for point in points:
                point_as_dict = self._problem.gen_config(
                    point[self.hp_size :], point[: self.hp_size]
                )
                batch.append(point_as_dict)
        return batch


def replace_nan(x):
    """
    :meta private:
    """
    return [np.nan if x_i == "nan" else x_i for x_i in x]
