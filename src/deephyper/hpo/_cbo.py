import functools
import logging
import numbers
import time
import warnings
from typing import Dict, List

import ConfigSpace as CS
import ConfigSpace.hyperparameters as csh
import numpy as np
import pandas as pd
from sklearn.base import is_regressor

import deephyper.core.exceptions
import deephyper.skopt
from deephyper.analysis.hpo import filter_failed_objectives
from deephyper.evaluator import HPOJob
from deephyper.hpo._problem import convert_to_skopt_space
from deephyper.hpo._search import Search
from deephyper.hpo.gmm import GMMSampler
from deephyper.skopt.moo import (
    MoScalarFunction,
    moo_functions,
    non_dominated_set,
    non_dominated_set_ranked,
)

# Adapt minimization -> maximization with DeepHyper
MAP_multi_point_strategy = {
    "cl_min": "cl_max",
    "cl_max": "cl_min",
    "qUCB": "qLCB",
    "qUCBd": "qLCBd",
}

MAP_acq_func = {"UCB": "LCB", "UCBd": "LCBd"}

MAP_filter_failures = {"min": "max"}


# schedulers
def scheduler_periodic_exponential_decay(i, eta_0, num_dim, period, rate, delay):
    """Periodic exponential decay scheduler for exploration-exploitation.

    Args:
        i (int): current iteration.
        eta_0 (float): initial value of the parameters ``[kappa, xi]`` to decay.
        num_dim (int): number of dimensions of the search space.
        period (int): period of the decay.
        rate (float): rate of the decay.
        delay (int): delay of the decay (decaying starts after ``delay`` iterations).

    Returns:
        tuple: an iterable of length 2 with the updated values at iteration ``i`` for
        ``[kappa, xi]``.
    """
    eta_i = eta_0 * np.exp(-rate * ((i - 1 - delay) % period))
    return eta_i


def scheduler_bandit(i, eta_0, num_dim, delta=0.05, lamb=0.2, delay=0):
    """Bandit scheduler for exploration-exploitation. Only valid for the UCB acquisition function.

    Args:
        i (int): current iteration.
        eta_0 (float): initial value of the parameters ``[kappa, xi]`` to decay.
        num_dim (int): number of dimensions of the search space.
        delta (float): confidence level.
        lamb (float): factor of the initial scheduler. Defaults to ``0.2``.
        delay (int): delay of the scheduler (decaying starts after ``delay`` iterations).

    Returns:
        tuple: an iterable of length 2 with the updated values at iteration ``i`` for
        ``[kappa, xi]``.
    """
    i = np.maximum(i + 1 - delay, 1)
    beta_i = 2 * np.log(num_dim * i**2 * np.pi**2 / (6 * delta)) * lamb
    beta_i = np.sqrt(beta_i)
    eta_i = eta_0[:]
    eta_i[0] = beta_i
    return eta_i


class CBO(Search):
    """Centralized Bayesian Optimisation Search.

    It follows a manager-workers architecture where the manager runs the Bayesian optimization
    loop and workers execute parallel evaluations of the black-box function.

    .. list-table::
        :widths: 25 25 25
        :header-rows: 1

        * - Single-Objective
          - Multi-Objectives
          - Failures
        * - ✅
          - ✅
          - ✅

    Example Usage:

        >>> search = CBO(problem, evaluator)
        >>> results = search.search(max_evals=100, timeout=120)

    Args:
        problem (HpProblem): Hyperparameter problem describing the search space to explore.

        evaluator (Evaluator): An ``Evaluator`` instance responsible of distributing the tasks.

        random_state (int, optional): Random seed. Defaults to ``None``.

        log_dir (str, optional): Log directory where search's results are saved. Defaults to
            ``"."``.

        verbose (int, optional): Indicate the verbosity level of the search. Defaults to ``0``.

        stopper (Stopper, optional): a stopper to leverage multi-fidelity when evaluating the
            function. Defaults to ``None`` which does not use any stopper.

        surrogate_model (Union[str,sklearn.base.RegressorMixin], optional): Surrogate model used by
            the Bayesian optimization. Can be a value in ``["RF", "GP", "ET", "MF", "GBRT",
            "DUMMY"]`` or a sklearn regressor. ``"ET"`` is for Extremely Randomized Trees which is
            the best compromise between speed and quality when performing a lot of parallel
            evaluations, i.e., reaching more than hundreds of evaluations. ``"GP"`` is for Gaussian-
            Process which is the best choice when maximizing the quality of iteration but quickly
            slow down when reaching hundreds of evaluations, also it does not support conditional
            search space. ``"RF"`` is for Random-Forest, slower than extremely randomized trees but
            with better mean estimate and worse epistemic uncertainty quantification capabilities.
            ``"GBRT"`` is for Gradient-Boosting Regression Tree, it has better mean estimate than
            other tree-based method worse uncertainty quantification capabilities and slower than
            ``"RF"``. Defaults to ``"ET"``.

        surrogate_model_kwargs (dict, optional): Additional parameters to pass to the surrogate
            model. Defaults to ``None``.

        acq_func (str, optional): Acquisition function used by the Bayesian optimization. Can be a
            value in ``["UCB", "EI", "PI", "gp_hedge"]``. Defaults to ``"UCB"``.

        acq_optimizer (str, optional): Method used to minimze the acquisition function. Can be a
            value in ``["sampling", "lbfgs", "ga", "mixedga"]``. Defaults to ``"auto"``.

        acq_optimizer_freq (int, optional): Frequency of optimization calls for the acquisition
            function. Defaults to ``10``, using optimizer every ``10`` surrogate model updates.

        kappa (float, optional): Manage the exploration/exploitation tradeoff for the "UCB"
            acquisition function. Defaults to ``1.96`` which corresponds to 95% of the confidence
            interval.

        xi (float, optional): Manage the exploration/exploitation tradeoff of ``"EI"`` and ``"PI"``
            acquisition function. Defaults to ``0.001``.

        n_points (int, optional): The number of configurations sampled from the search space to
            infer each batch of new evaluated configurations.

        filter_duplicated (bool, optional): Force the optimizer to sample unique points until the
            search space is "exhausted" in the sens that no new unique points can be found given
            the sampling size ``n_points``. Defaults to ``True``.

        update_prior (bool, optional): Update the prior of the surrogate model with the new
            evaluated points. Defaults to ``False``. Should be set to ``True`` when all objectives
            and parameters are continuous.

        update_prior_quantile (float, optional): The quantile used to update the prior.
            Defaults to ``0.1``.

        multi_point_strategy (str, optional): Definition of the constant value use for the Liar
            strategy. Can be a value in ``["cl_min", "cl_mean", "cl_max", "qUCB", "qUCBd"]``. All
            ``"cl_..."`` strategies follow the constant-liar scheme, where if $N$ new points are
            requested, the surrogate model is re-fitted $N-1$ times with lies (respectively, the
            minimum, mean and maximum objective found so far; for multiple objectives, these are
            the minimum, mean and maximum of the individual objectives) to infer the acquisition
            function. Constant-Liar strategy have poor scalability because of this repeated re-
            fitting. The ``"qUCB"`` strategy is much more efficient by sampling a new $kappa$ value
            for each new requested point without re-fitting the model.

        n_jobs (int, optional): Number of parallel processes used to fit the surrogate model of the
            Bayesian optimization. A value of ``-1`` will use all available cores. Not used in
            ``surrogate_model`` if passed as own sklearn regressor. Defaults to ``1``.

        n_initial_points (int, optional): Number of collected objectives required before fitting
            the surrogate-model. Defaults to ``10``.

        initial_point_generator (str, optional): Sets an initial points generator. Can be either
            ``["random", "sobol", "halton", "hammersly", "lhs", "grid"]``. Defaults to ``"random"``.

        initial_points (List[Dict], optional): A list of initial points to evaluate where each
            point is a dictionnary where keys are names of hyperparameters and values their
            corresponding choice. Defaults to ``None`` for them to be generated randomly from
            the search space.

        filter_failures (str, optional): Replace objective of failed configurations by ``"min"``
            or ``"mean"``. If ``"ignore"`` is passed then failed configurations will be
            filtered-out and not passed to the surrogate model. For multiple objectives, failure of
            any single objective will lead to treating that configuration as failed and each of
            these multiple objective will be replaced by their individual ``"min"`` or ``"mean"``
            of past configurations. Defaults to ``"min"`` to replace failed configurations
            objectives by the running min of all objectives.

        max_failures (int, optional): Maximum number of failed configurations allowed before
            observing a valid objective value when ``filter_failures`` is not equal to
            ``"ignore"``. Defaults to ``100``.

        moo_lower_bounds (list, optional): List of lower bounds on the interesting range of
            objective values. Must be the same length as the number of obejctives. Defaults to
            ``None``, i.e., no bounds. Can bound only a single objective by providing ``None``
            for all other values. For example, ``moo_lower_bounds=[None, 0.5, None]`` will explore
            all tradeoffs for the objectives at index 0 and 2, but only consider scores for
            objective 1 that exceed 0.5.

        moo_scalarization_strategy (str, optional): Scalarization strategy used in multiobjective
            optimization. Can be a value in ``["Linear", "Chebyshev", "AugChebyshev", "PBI",
            "Quadratic"]``. Defaults to ``"Chebyshev"``. Typically, randomized methods should be
            used to capture entire Pareto front, unless there is a known target solution a priori.
            Additional details on each scalarization can be found in :mod:`deephyper.skopt.moo`.

        moo_scalarization_weight (list, optional): Scalarization weights to be used in
            multiobjective optimization with length equal to the number of objective functions.
            Defaults to ``None`` for randomized weights. Only set if you want to fix the
            scalarization weights for a multiobjective HPS.

        scheduler (dict, callable, optional): a function to manage the value of ``kappa, xi`` with
            iterations. Defaults to ``None`` which does not use any scheduler. The periodic
            exponential decay scheduler can be used with  ``scheduler={"type":
            "periodic-exp-decay", "period": 30, "rate": 0.1}``. The scheduler can also be a
            callable function with signature ``scheduler(i, eta_0, **kwargs)`` where ``i`` is the
            current iteration, ``eta_0`` is the initial value of ``[kappa, xi]`` and ``kwargs`` are
            other fixed parameters of the function. Instead of fixing the decay ``"rate"`` the
            final ``kappa`` or ``xi`` can be used ``{"type": "periodic-exp-decay", "period": 25,
            "kappa_final": 1.96}``.

        objective_scaler (str, optional): a way to map the objective space to some other support
            for example to normalize it. Defaults to ``"auto"`` which automatically set it to
            "identity" for any surrogate model except "RF" which will use "quantile-uniform".
    """

    def __init__(
        self,
        problem,
        evaluator,
        random_state: int = None,
        log_dir: str = ".",
        verbose: int = 0,
        stopper=None,
        surrogate_model="ET",
        surrogate_model_kwargs: dict = None,  # TODO: documentation
        acq_func: str = "UCBd",
        acq_optimizer: str = "auto",
        acq_optimizer_freq: int = 10,
        kappa: float = 1.96,
        xi: float = 0.001,
        n_points: int = 10_000,
        filter_duplicated: bool = True,
        update_prior: bool = False,
        update_prior_quantile: float = 0.1,
        multi_point_strategy: str = "cl_max",
        n_jobs: int = 1,  # 32 is good for Theta
        n_initial_points: int = 10,
        initial_point_generator: str = "random",
        initial_points=None,
        filter_failures: str = "min",
        max_failures: int = 100,
        moo_lower_bounds=None,
        moo_scalarization_strategy: str = "Chebyshev",
        moo_scalarization_weight=None,
        scheduler=None,
        objective_scaler="auto",
        **kwargs,
    ):
        super().__init__(problem, evaluator, random_state, log_dir, verbose, stopper)
        # get the __init__ parameters
        self._init_params = locals()

        # check input parameters
        if type(n_jobs) is not int:
            raise ValueError(f"Parameter n_jobs={n_jobs} should be an integer value!")

        surrogate_model_allowed = [
            # Trees
            "RF",
            "ET",
            "TB",
            "RS",
            "MF",
            # Other models
            "GBRT",
            "GP",
            "HGBRT",
            # Random Search
            "DUMMY",
        ]
        if surrogate_model in surrogate_model_allowed:
            base_estimator = self._get_surrogate_model(
                surrogate_model,
                n_jobs=n_jobs,
                random_state=self._random_state.randint(0, 2**31),
                surrogate_model_kwargs=surrogate_model_kwargs,
            )
        elif is_regressor(surrogate_model):
            base_estimator = surrogate_model
        else:
            raise ValueError(
                f"Parameter 'surrogate_model={surrogate_model}' should have a value in "
                f"{surrogate_model_allowed}, or be a sklearn regressor!"
            )

        acq_func_allowed = [
            "UCB",
            "EI",
            "PI",
            "MES",
            "gp_hedge",
            "UCBd",
            "EId",
            "PId",
            "MESd",
            "gp_hedged",
        ]
        if acq_func not in acq_func_allowed:
            raise ValueError(
                f"Parameter 'acq_func={acq_func}' should have a value in {acq_func_allowed}!"
            )

        if not (np.isscalar(kappa)):
            raise ValueError("Parameter 'kappa' should be a scalar value!")

        if not (np.isscalar(xi)):
            raise ValueError("Parameter 'xi' should be a scalar value!")

        if type(n_points) is not int:
            raise ValueError("Parameter 'n_points' shoud be an integer value!")

        if type(filter_duplicated) is not bool:
            raise ValueError(
                f"Parameter filter_duplicated={filter_duplicated} should be a boolean value!"
            )

        if type(max_failures) is not int:
            raise ValueError(f"Parameter max_failures={max_failures} should be an integer value!")

        # Initialize lower bounds for objectives
        if moo_lower_bounds is None:
            self._moo_upper_bounds = None
        elif isinstance(moo_lower_bounds, list) and all(
            [isinstance(lbi, numbers.Number) or lbi is None for lbi in moo_lower_bounds]
        ):
            self._moo_upper_bounds = [
                -lbi if isinstance(lbi, numbers.Number) else None for lbi in moo_lower_bounds
            ]
        else:
            raise ValueError(
                f"Parameter 'moo_lower_bounds={moo_lower_bounds}' is invalid. Must be None or "
                f"a list"
            )

        moo_scalarization_strategy_allowed = list(moo_functions.keys())
        if not (
            moo_scalarization_strategy in moo_scalarization_strategy_allowed
            or isinstance(moo_scalarization_strategy, MoScalarFunction)
        ):
            raise ValueError(
                f"Parameter 'moo_scalarization_strategy={moo_scalarization_strategy}' should have a"
                f" value in {moo_scalarization_strategy_allowed} or be a subclass of "
                f"deephyper.skopt.moo.MoScalarFunction!"
            )
        self._moo_scalarization_strategy = moo_scalarization_strategy
        self._moo_scalarization_weight = moo_scalarization_weight

        multi_point_strategy_allowed = [
            "cl_min",
            "cl_mean",
            "cl_max",
            "topk",
            "boltzmann",
            "qUCB",
            "qUCBd",
        ]
        if multi_point_strategy not in multi_point_strategy_allowed:
            raise ValueError(
                f"Parameter multi_point_strategy={multi_point_strategy} should have a value "
                f"in {multi_point_strategy_allowed}!"
            )

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

        self._multi_point_strategy = MAP_multi_point_strategy.get(
            multi_point_strategy, multi_point_strategy
        )
        self._fitted = False

        # Map the ConfigSpace to Skop Space
        self._opt_space = convert_to_skopt_space(
            self._problem.space, surrogate_model=surrogate_model
        )

        self._opt = None
        self._opt_kwargs = dict(
            dimensions=self._opt_space,
            base_estimator=base_estimator,
            # optimizer
            initial_point_generator=initial_point_generator,
            acq_optimizer=acq_optimizer,
            acq_optimizer_kwargs={
                "n_points": n_points,
                "filter_duplicated": filter_duplicated,
                "update_prior": update_prior,
                "update_prior_quantile": 1 - update_prior_quantile,
                "n_jobs": n_jobs,
                "filter_failures": MAP_filter_failures.get(filter_failures, filter_failures),
                "max_failures": max_failures,
                "acq_optimizer_freq": acq_optimizer_freq,
            },
            # acquisition function
            acq_func=MAP_acq_func.get(acq_func, acq_func),
            acq_func_kwargs={"xi": xi, "kappa": kappa},
            n_initial_points=self._n_initial_points,
            initial_points=self._initial_points,
            random_state=self._random_state,
            moo_upper_bounds=self._moo_upper_bounds,
            moo_scalarization_strategy=self._moo_scalarization_strategy,
            moo_scalarization_weight=self._moo_scalarization_weight,
            objective_scaler=objective_scaler,
        )

        # Scheduler policy
        scheduler = {"type": "bandit"} if scheduler is None else scheduler
        self.scheduler = None
        if type(scheduler) is dict:
            scheduler = scheduler.copy()
            scheduler_type = scheduler.pop("type", None)
            assert scheduler_type in ["periodic-exp-decay", "bandit"]

            if scheduler_type == "periodic-exp-decay":
                rate = scheduler.get("rate", None)
                period = scheduler.get("period", None)

                # Automatically retrieve the "decay rate" of the scheduler by solving
                # the equation: eta_0 * exp(-rate * period) = eta_final
                if rate is None:
                    if "UCB" in acq_func:
                        kappa_final = scheduler.pop("kappa_final", 0.1)
                        rate = -1 / period * np.log(kappa_final / kappa)
                    elif "EI" in acq_func or "PI" in acq_func:
                        xi_final = scheduler.pop("xi_final", 0.0001)
                        rate = -1 / period * np.log(xi_final / xi)
                    else:
                        rate = 0.1

                scheduler_params = {
                    "period": period,
                    "rate": rate,
                    "delay": n_initial_points,
                }
                scheduler_func = scheduler_periodic_exponential_decay

            elif scheduler_type == "bandit":
                scheduler_params = {
                    "delta": 0.05,
                    "lamb": 0.2,
                    "delay": n_initial_points,
                }
                scheduler_func = scheduler_bandit

            scheduler_params.update(scheduler)
            eta_0 = np.array([kappa, xi])
            self.scheduler = functools.partial(
                scheduler_func,
                eta_0=eta_0,
                num_dim=len(self._problem),
                **scheduler_params,
            )
            logging.info(
                f"Set up scheduler '{scheduler_type}' with parameters '{scheduler_params}'"
            )
        elif callable(scheduler):
            self.scheduler = functools.partial(scheduler, eta_0=np.array([kappa, xi]))
            logging.info(f"Set up scheduler '{scheduler}'")

        self._num_asked = 0

    def _setup_optimizer(self):
        if self._fitted:
            self._opt_kwargs["n_initial_points"] = 0
        self._opt = deephyper.skopt.Optimizer(**self._opt_kwargs)

    def _apply_scheduler(self, i):
        """Apply scheduler policy and update corresponding values in Optimizer."""
        if self.scheduler is not None:
            kappa, xi = self.scheduler(i)
            values = {"kappa": kappa, "xi": xi}
            logging.info(f"Updated exploration-exploitation policy with {values} from scheduler")
            self._opt.acq_func_kwargs.update(values)

    def _ask(self, n: int = 1) -> List[Dict]:
        """Ask the search for new configurations to evaluate.

        Args:
            n (int, optional): The number of configurations to ask. Defaults to 1.

        Returns:
            List[Dict]: a list of hyperparameter configurations to evaluate.
        """
        new_X = self._opt.ask(n_points=n, strategy=self._multi_point_strategy)
        new_samples = [self._to_dict(x) for x in new_X]
        self._num_asked += n
        return new_samples

    def _tell(self, results: List[HPOJob]):
        """Tell the search the results of the evaluations.

        Args:
            results (List[HPOJob]): a dictionary containing the results of the evaluations.
        """
        # Transform configurations to list to fit optimizer
        logging.info("Transforming received configurations to list...")
        t1 = time.time()

        opt_X = []  # input configuration
        opt_y = []  # objective value
        # for cfg, obj in new_results:
        for job_i in results:
            cfg, obj = job_i
            x = list(cfg.values())

            if isinstance(obj, numbers.Number) or all(
                isinstance(obj_i, numbers.Number) for obj_i in obj
            ):
                opt_X.append(x)
                opt_y.append(np.negative(obj).tolist())  # !maximizing
            elif (type(obj) is str and "F" == obj[0]) or any(
                type(obj_i) is str and "F" == obj_i[0] for obj_i in obj
            ):
                if self._opt_kwargs["acq_optimizer_kwargs"]["filter_failures"] == "ignore":
                    continue
                else:
                    opt_X.append(x)
                    opt_y.append("F")

        logging.info(f"Transformation took {time.time() - t1:.4f} sec.")

        # apply scheduler
        self._apply_scheduler(self._num_asked)

        if len(opt_y) > 0:
            logging.info("Fitting the optimizer...")
            t1 = time.time()
            self._opt.tell(opt_X, opt_y)
            logging.info(f"Fitting took {time.time() - t1:.4f} sec.")

    def _search(self, max_evals, timeout, max_evals_strict=False):
        if self._opt is None:
            self._setup_optimizer()

        super()._search(max_evals, timeout, max_evals_strict)

    def _get_surrogate_model(
        self,
        name: str,
        n_jobs: int = 1,
        random_state: int = None,
        surrogate_model_kwargs: dict = None,
    ):
        """Get a surrogate model from Scikit-Optimize.

        Args:
            name (str): name of the surrogate model.
            n_jobs (int): number of parallel processes to distribute the computation of the
                surrogate model.
            random_state (int): random seed.
            surrogate_model_kwargs (dict): additional parameters to pass to the surrogate model.

        Returns:
            sklearn.base.RegressorMixin: a surrogate model capabable of predicting y_mean and y_std.

        Raises:
            ValueError: when the name of the surrogate model is unknown.
        """
        # Check if the surrogate model is supported
        accepted_names = ["RF", "ET", "TB", "RS", "GBRT", "DUMMY", "GP", "MF", "HGBRT"]
        if name not in accepted_names:
            raise ValueError(
                f"Unknown surrogate model {name}, please choose among {accepted_names}."
            )

        if surrogate_model_kwargs is None:
            surrogate_model_kwargs = {}

        # Define default surrogate model parameters
        if name in ["RF", "ET", "TB", "RS", "MF"]:
            default_surrogate_model_kwargs = dict(
                n_estimators=100,
                max_samples=0.8,
                min_samples_split=2,  # Aleatoric Variance will be 0
                n_jobs=n_jobs,
                random_state=random_state,
            )

            # From https://link.springer.com/article/10.1007/s10994-006-6226-1
            # We follow parameters indicated at: p. 8, Sec. 2.2.2
            # Model: Random Forest from L. Breiman
            if name == "RF":
                default_surrogate_model_kwargs["splitter"] = "best"
                default_surrogate_model_kwargs["max_features"] = "sqrt"
                default_surrogate_model_kwargs["bootstrap"] = True
            # Model: Extremely Randomized Forest
            elif name == "ET":
                default_surrogate_model_kwargs["splitter"] = "random"
                default_surrogate_model_kwargs["max_features"] = 1.0
                default_surrogate_model_kwargs["bootstrap"] = False
                default_surrogate_model_kwargs["max_samples"] = None
            # Model: Tree Bagging
            elif name == "TB":
                default_surrogate_model_kwargs["bootstrap"] = True
                default_surrogate_model_kwargs["splitter"] = "best"
                default_surrogate_model_kwargs["max_features"] = 1.0
            elif name == "RS":
                default_surrogate_model_kwargs["splitter"] = "best"
                default_surrogate_model_kwargs["bootstrap"] = False
                default_surrogate_model_kwargs["max_samples"] = None
                default_surrogate_model_kwargs["max_features"] = "sqrt"
            elif name == "MF":
                default_surrogate_model_kwargs["bootstrap"] = False
                default_surrogate_model_kwargs["max_samples"] = None

        elif name == "GBRT":
            default_surrogate_model_kwargs = dict(
                n_estimtaors=10,
                n_jobs=n_jobs,
                random_state=random_state,
            )
        elif name == "HGBRT":
            default_surrogate_model_kwargs = dict(
                n_jobs=n_jobs,
                random_state=random_state,
            )
        else:
            default_surrogate_model_kwargs = {}

        default_surrogate_model_kwargs.update(surrogate_model_kwargs)

        if name in ["RF", "TB", "RS", "ET"]:
            surrogate = deephyper.skopt.learning.RandomForestRegressor(
                **default_surrogate_model_kwargs,
            )

        # Model: Mondrian Forest
        elif name == "MF":
            try:
                surrogate = deephyper.skopt.learning.MondrianForestRegressor(
                    **default_surrogate_model_kwargs,
                )
            except AttributeError:
                raise deephyper.core.exceptions.MissingRequirementError(
                    "Installing 'deephyper/scikit-garden' is required to use MondrianForest (MF) "
                    "regressor as a surrogate model!"
                )
        # Model: Gradient Boosting Regression Tree (based on quantiles)
        # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
        elif name == "GBRT":
            from sklearn.ensemble import GradientBoostingRegressor

            gbrt = GradientBoostingRegressor(
                n_estimators=default_surrogate_model_kwargs.pop("n_estimators"),
                loss="quantile",
            )
            surrogate = deephyper.skopt.learning.GradientBoostingQuantileRegressor(
                base_estimator=gbrt,
                **default_surrogate_model_kwargs,
            )
        # Model: Histogram-based Gradient Boosting Regression Tree (based on quantiles)
        # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html
        elif name == "HGBRT":
            from sklearn.ensemble import HistGradientBoostingRegressor

            # Check wich parameters are categorical
            categorical_features = []
            for hp_name in self._problem.space:
                hp = self._problem.space.get_hyperparameter(hp_name)

                categorical_features.append(
                    isinstance(hp, csh.CategoricalHyperparameter)
                    # or isinstance(hp, csh.OrdinalHyperparameter)
                )

            gbrt = HistGradientBoostingRegressor(
                loss="quantile", categorical_features=categorical_features
            )
            surrogate = deephyper.skopt.learning.GradientBoostingQuantileRegressor(
                base_estimator=gbrt,
                **default_surrogate_model_kwargs,
            )
        else:  # for DUMMY and GP
            surrogate = name

        return surrogate

    def _return_cond(self, cond, cst_new):
        parent = cst_new.get_hyperparameter(cond.parent.name)
        child = cst_new.get_hyperparameter(cond.child.name)
        if type(cond) is CS.EqualsCondition:
            value = cond.value
            cond_new = CS.EqualsCondition(child, parent, cond.value)
        elif type(cond) is CS.GreaterThanCondition:
            value = cond.value
            cond_new = CS.GreaterThanCondition(child, parent, value)
        elif type(cond) is CS.NotEqualsCondition:
            value = cond.value
            cond_new = CS.GreaterThanCondition(child, parent, value)
        elif type(cond) is CS.LessThanCondition:
            value = cond.value
            cond_new = CS.GreaterThanCondition(child, parent, value)
        elif type(cond) is CS.InCondition:
            values = cond.values
            cond_new = CS.GreaterThanCondition(child, parent, values)
        else:
            print("Not supported type" + str(type(cond)))
        return cond_new

    def _return_forbid(self, cond, cst_new):
        if type(cond) is CS.ForbiddenEqualsClause or type(cond) is CS.ForbiddenInClause:
            hp = cst_new.get_hyperparameter(cond.hyperparameter.name)
            if type(cond) is CS.ForbiddenEqualsClause:
                value = cond.value
                cond_new = CS.ForbiddenEqualsClause(hp, value)
            elif type(cond) is CS.ForbiddenInClause:
                values = cond.values
                cond_new = CS.ForbiddenInClause(hp, values)
            else:
                print("Not supported type" + str(type(cond)))
        return cond_new

    def fit_surrogate(self, df):
        """Fit the surrogate model of the search from a checkpointed Dataframe.

        Args:
            df (str|DataFrame): a checkpoint from a previous search.

        Example Usage:

        >>> search = CBO(problem, evaluator)
        >>> search.fit_surrogate("results.csv")
        """
        if type(df) is not str and not isinstance(df, pd.DataFrame):
            raise ValueError("The argument 'df' should be a path to a CSV file or a DataFrame!")

        if type(df) is str and df[-4:] == ".csv":
            df = pd.read_csv(df)

        df, df_failures = filter_failed_objectives(df)

        self._fitted = True

        if self._opt is None:
            self._setup_optimizer()

        hp_names = [f"p:{name}" for name in self._problem.hyperparameter_names]
        try:
            x = df[hp_names].values.tolist()
            x += df_failures[hp_names].values.tolist()

            # check single or multiple objectives
            if "objective" in df.columns:
                y = df.objective.tolist()
            else:
                y = df.filter(regex=r"^objective_\d+$").values.tolist()
        except KeyError:
            raise ValueError("Incompatible dataframe 'df' to fit surrogate model of CBO.")

        y = [np.negative(yi).tolist() for yi in y] + ["F"] * len(df_failures)

        self._opt.tell(x, y)

    def fit_generative_model(
        self,
        df,
        q=0.90,
        verbose=False,
    ):
        """Fits a generative model for sampling during BO.

        Learn the distribution of hyperparameters for the top-``(1-q)x100%`` configurations and
        sample from this distribution. It can be used for transfer learning. For multiobjective
        problems, this function computes the top-``(1-q)x100%`` configurations in terms of their
        ranking with respect to pareto efficiency: all points on the first non-dominated pareto
        front have rank 1 and in general, points on the k'th non-dominated front have rank k.

        Example Usage:

        >>> search = CBO(problem, evaluator)
        >>> search.fit_surrogate("results.csv")

        Args:
            df (str|DataFrame): a dataframe or path to CSV from a previous search.

            q (float, optional): the quantile defined the set of top configurations used to bias
                the search. Defaults to ``0.90`` which select the top-10% configurations from
                ``df``.

            verbose (bool, optional): If set to ``True`` it will print the score of the generative
                model. Defaults to ``False``.

        Returns:
            model: the generative model.
        """
        if type(df) is str and df[-4:] == ".csv":
            df = pd.read_csv(df)
        assert isinstance(df, pd.DataFrame)

        if len(df) < 10:
            raise ValueError(
                f"The passed DataFrame contains only {len(df)} results when a minimum of "
                f"10 is required!"
            )

        # !avoid error linked to `n_components=10` a parameter of generative model used
        q_max = 1 - 10 / len(df)
        if q_max < q:
            warnings.warn(
                f"The value of q={q} is replaced by q_max={q_max} because a minimum of 10 samples "
                f"sare required to perform transfer-learning!",
                category=UserWarning,
            )
            q = q_max

        # check single or multiple objectives
        hp_cols = [k for k in df.columns if "p:" == k[:2]]
        if "objective" in df.columns:
            # filter failures
            if pd.api.types.is_string_dtype(df.objective):
                df = df[~df.objective.str.startswith("F")]
                df.objective = df.objective.astype(float)

            q_val = np.quantile(df.objective.values, q)
            req_df = df.loc[df["objective"] > q_val]
        else:
            # filter failures
            objcol = list(df.filter(regex=r"^objective_\d+$").columns)
            for col in objcol:
                if pd.api.types.is_string_dtype(df[col]):
                    df = df[~df[col].str.startswith("F")]
                    df[col] = df[col].astype(float)

            top = non_dominated_set_ranked(-np.asarray(df[objcol]), 1.0 - q)
            req_df = df.loc[top]

        req_df = req_df[["job_id"] + hp_cols]
        req_df = req_df.rename(columns={k: k[2:] for k in hp_cols if k.startswith("p:")})

        model = GMMSampler(self._problem.space, random_state=self._random_state)
        model.fit(req_df)

        self._opt_kwargs["model_sdv"] = model

        return model

    def fit_search_space(self, df, fac_numerical=0.125, fac_categorical=10):
        """Apply prior-guided transfer learning based on a DataFrame of results.

        Example Usage:

        >>> search = CBO(problem, evaluator)
        >>> search.fit_surrogate("results.csv")

        Args:
            df (str|DataFrame): a checkpoint from a previous search.

            fac_numerical (float): the factor used to compute the sigma of a truncated normal
                distribution based on ``sigma = max(1.0, (upper - lower) * fac_numerical)``. A
                small large factor increase exploration while a small factor increase exploitation
                around the best-configuration from the ``df`` parameter.

            fac_categorical (float): the weight given to a categorical feature part of the best
                configuration. A large weight ``> 1`` increase exploitation while a small factor
                close to ``1`` increase exploration.
        """
        if type(df) is str and df[-4:] == ".csv":
            df = pd.read_csv(df)
        assert isinstance(df, pd.DataFrame)

        # check single or multiple objectives
        if "objective" in df.columns:
            # filter failures
            if pd.api.types.is_string_dtype(df.objective):
                df = df[~df.objective.str.startswith("F")]
                df.objective = df.objective.astype(float)
        else:
            # filter failures
            objcol = df.filter(regex=r"^objective_\d+$").columns
            for col in objcol:
                if pd.api.types.is_string_dtype(df[col]):
                    df = df[~df[col].str.startswith("F")]
                    df[col] = df[col].astype(float)

        cst = self._problem.space
        if type(cst) is not CS.ConfigurationSpace:
            logging.error(f"{type(cst)}: not supported for trainsfer learning")

        res_df = df
        res_df_names = res_df.columns.values
        if "objective" in df.columns:
            best_index = np.argmax(res_df["objective"].values)
            best_param = res_df.iloc[best_index]
        else:
            best_index = non_dominated_set(-np.asarray(res_df[objcol]), return_mask=False)[0]
            best_param = res_df.iloc[best_index]

        cst_new = CS.ConfigurationSpace(seed=self._random_state.randint(0, 2**31))
        hp_names = list(cst.keys())
        for hp_name in hp_names:
            hp = cst[hp_name]
            if hp_name in res_df_names:
                if (
                    type(hp) is csh.UniformIntegerHyperparameter
                    or type(hp) is csh.UniformFloatHyperparameter
                ):
                    mu = best_param[hp.name]
                    lower = hp.lower
                    upper = hp.upper
                    sigma = max(1.0, (upper - lower) * fac_numerical)
                    if type(hp) is csh.UniformIntegerHyperparameter:
                        param_new = csh.NormalIntegerHyperparameter(
                            name=hp.name,
                            default_value=mu,
                            mu=mu,
                            sigma=sigma,
                            lower=lower,
                            upper=upper,
                        )
                    else:  # type is csh.UniformFloatHyperparameter:
                        param_new = csh.NormalFloatHyperparameter(
                            name=hp.name,
                            default_value=mu,
                            mu=mu,
                            sigma=sigma,
                            lower=lower,
                            upper=upper,
                        )
                    cst_new.add(param_new)
                elif (
                    type(hp) is csh.CategoricalHyperparameter
                    or type(hp) is csh.OrdinalHyperparameter
                ):
                    if type(hp) is csh.OrdinalHyperparameter:
                        choices = hp.sequence
                    else:
                        choices = hp.choices
                    weights = len(choices) * [1.0]
                    index = choices.index(best_param[hp.name])
                    weights[index] = fac_categorical
                    norm_weights = [float(i) / sum(weights) for i in weights]
                    param_new = csh.CategoricalHyperparameter(
                        name=hp.name, choices=choices, weights=norm_weights
                    )
                    cst_new.add(param_new)
                else:
                    logging.warning(f"Not fitting {hp} because it is not supported!")
                    cst_new.add(hp)
            else:
                logging.warning(f"Not fitting {hp} because it was not found in the dataframe!")
                cst_new.add(hp)

        # For conditions
        for cond in cst.conditions():
            if type(cond) is CS.AndConjunction or type(cond) is CS.OrConjunction:
                cond_list = []
                for comp in cond.components:
                    cond_list.append(self._return_cond(comp, cst_new))
                if type(cond) is CS.AndConjunction:
                    cond_new = CS.AndConjunction(*cond_list)
                elif type(cond) is CS.OrConjunction:
                    cond_new = CS.OrConjunction(*cond_list)
                else:
                    logging.warning(f"Condition {type(cond)} is not implemented!")
            else:
                cond_new = self._return_cond(cond, cst_new)
            cst_new.add(cond_new)

        # For forbiddens
        for cond in cst.forbidden_clauses:
            if type(cond) is CS.ForbiddenAndConjunction:
                cond_list = []
                for comp in cond.components:
                    cond_list.append(self._return_forbid(comp, cst_new))
                cond_new = CS.ForbiddenAndConjunction(*cond_list)
            elif type(cond) is CS.ForbiddenEqualsClause or type(cond) is CS.ForbiddenInClause:
                cond_new = self._return_forbid(cond, cst_new)
            else:
                logging.warning(f"Forbidden {type(cond)} is not implemented!")
            cst_new.add(cond_new)

        self._opt_kwargs["dimensions"] = cst_new

    def _to_dict(self, x: list) -> dict:
        """Transform a list of hyperparameter values to a ``dict``.

        The keys are hyperparameters names and values are hyperparameters values.

        Args:
            x (list): a list of hyperparameter values.

        Returns:
            dict: a dictionnary of hyperparameter names and values.
        """
        res = {}
        hps_names = self._problem.hyperparameter_names

        # to enforce native python types instead of numpy types
        x = map(lambda xi: getattr(xi, "tolist", lambda: xi)(), x)

        for hps_name, xi in zip(hps_names, x):
            res[hps_name] = xi

        return res
