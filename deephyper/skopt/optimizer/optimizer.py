import sys
import warnings
from math import log
import numbers

import ConfigSpace as CS
import numpy as np
import pandas as pd
from scipy.optimize import fmin_l_bfgs_b
from sklearn.base import clone, is_regressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.utils import check_random_state

from deephyper.core.utils.joblib_utils import Parallel, delayed

from ..acquisition import _gaussian_acquisition, gaussian_acquisition_1D
from ..learning import GaussianProcessRegressor
from ..moo import MoScalarFunction, moo_functions
from ..space import Categorical, Space
from ..utils import (
    check_x_in_space,
    cook_estimator,
    cook_initial_point_generator,
    cook_objective_scaler,
    create_result,
    has_gradients,
    is_2Dlistlike,
    is_listlike,
    normalize_dimensions,
)


class ExhaustedSearchSpace(RuntimeError):
    """ "Raised when the search cannot sample new points from the ConfigSpace."""

    def __str__(self):
        return "The search space is exhausted and cannot sample new unique points!"


class ExhaustedFailures(RuntimeError):
    """Raised when the search has seen ``max_failures`` failures without any valid objective value."""

    def __str__(self):
        return "The search has reached its quota of failures! Check if the type of failure is expected or the value of ``max_failures`` in the search algorithm."


def boltzmann_distribution(x, beta=1):
    x = np.exp(beta * x)
    x = x / np.sum(x)
    return x


OBJECTIVE_VALUE_FAILURE = "F"


class Optimizer(object):
    """Run bayesian optimisation loop in DeepHyper.

    An ``Optimizer`` represents the steps of a bayesian optimisation loop. To
    use it you need to provide your own loop mechanism. The various
    optimisers provided by ``skopt`` use this class under the hood.

    Do not call this class directly, it is used for "Ask" and "Tell" in
    DeepHyper's bayesian optimisation loop.

    Args:
        dimensions (list): List of search space dimensions.
            Each search dimension can be defined either as

            - a `(lower_bound, upper_bound)` tuple (for `Real` or `Integer`
              dimensions),
            - a `(lower_bound, upper_bound, "prior")` tuple (for `Real`
              dimensions),
            - as a list of categories (for `Categorical` dimensions), or
            - an instance of a `Dimension` object (`Real`, `Integer` or
              `Categorical`).

        base_estimator (str, optional): One of
            `"GP"`, `"RF"`, `"ET"`, `"GBRT"` or sklearn
            regressor, default: `"GP"`
            Should inherit from :obj:`sklearn.base.RegressorMixin`.
            In addition the `predict` method, should have an optional
            `return_std` argument, which returns `std(Y | x)` along with
            `E[Y | x]`.
            If base_estimator is one of ["GP", "RF", "ET", "GBRT"], a default
            surrogate model of the corresponding type is used corresponding to
            what is used in the minimize functions.

        n_random_starts (int, optional): default is 10
            .. deprecated:: 0.6 use `n_initial_points` instead.

        n_initial_points (int, optional):
            Number of evaluations of `func` with initialization points
            before approximating it with `base_estimator`. Initial point
            generator can be changed by setting `initial_point_generator`.
            Default is 10.

        initial_points (list, optional): default is None

        initial_point_generator (str, optional): InitialPointGenerator instance
            Default is `"random"`.
            Sets a initial points generator. Can be either

            - `"random"` for uniform random numbers,
            - `"sobol"` for a Sobol' sequence,
            - `"halton"` for a Halton sequence,
            - `"hammersly"` for a Hammersly sequence,
            - `"lhs"` for a latin hypercube sequence,
            - `"grid"` for a uniform grid sequence

        acq_func (string, optional): Default is `"gp_hedge"`.
            Function to minimize over the posterior distribution.
            Can be either

            - `"LCB"` for lower confidence bound.
            - `"EI"` for negative expected improvement.
            - `"PI"` for negative probability of improvement.
            - `"gp_hedge"` Probabilistically choose one of the above three
              acquisition functions at every iteration.

              - The gains `g_i` are initialized to zero.
              - At every iteration,

                - Each acquisition function is optimised independently to
                  propose an candidate point `X_i`.
                - Out of all these candidate points, the next point `X_best` is
                  chosen by :math:`softmax(\\eta g_i)`
                - After fitting the surrogate model with `(X_best, y_best)`,
                  the gains are updated such that :math:`g_i -= \\mu(X_i)`

            - `"EIps"` for negated expected improvement per second to take into
              account the function compute time. Then, the objective function is
              assumed to return two values, the first being the objective value and
              the second being the time taken in seconds.
            - `"PIps"` for negated probability of improvement per second. The
              return type of the objective function is assumed to be similar to
              that of `"EIps"`

        acq_optimizer (string, optional): `"sampling"` or `"lbfgs"`, default is `"auto"`
            Method to minimize the acquisition function. The fit model
            is updated with the optimal value obtained by optimizing `acq_func`
            with `acq_optimizer`.

            - If set to `"auto"`, then `acq_optimizer` is configured on the
              basis of the base_estimator and the space searched over.
              If the space is Categorical or if the estimator provided based on
              tree-models then this is set to be `"sampling"`.
            - If set to `"sampling"`, then `acq_func` is optimized by computing
              `acq_func` at `n_points` randomly sampled points.
            - If set to `"lbfgs"`, then `acq_func` is optimized by

              - Sampling `n_restarts_optimizer` points randomly.
              - `"lbfgs"` is run for 20 iterations with these points as initial
                points to find local minima.
              - The optimal of these local minima is used to update the prior.

        random_state (int, optional): RandomState instance, or None (default)
            Set random state to something other than None for reproducible
            results.

        n_jobs (int, optional): Default is 1.
            The number of jobs to run in parallel in the base_estimator,
            if the base_estimator supports n_jobs as parameter and
            base_estimator was given as string.
            If -1, then the number of jobs is set to the number of cores.

        acq_func_kwargs (dict, optional):
            Additional arguments to be passed to the acquisition function.

        acq_optimizer_kwargs (dict, optional):
            Additional arguments to be passed to the acquisition optimizer.

        model_queue_size (int or None, optional): Default is None.
            Keeps list of models only as long as the argument given. In the
            case of None, the list has no capped length.

        model_sdv (Model or None, optional): Default None
            A Model from Synthetic-Data-Vault.

        moo_scalarization_strategy (string, optional): Default is `"Chebyshev"`
            Function to convert multiple objectives into a single scalar value. Can be either

            - `"Linear"` for linear/convex combination.
            - `"Chebyshev"` for Chebyshev or weighted infinity norm.
            - `"AugChebyshev"` for Chebyshev norm augmented with a weighted 1-norm.
            - `"PBI"` for penalized boundary intersection.
            - `"Quadratic"` for quadratic combination (2-norm).

        moo_scalarization_weight (array, optional) Default is `None`.
            Scalarization weights to be used in multiobjective optimization with length equal to the number of objective functions.
            When set to `None`, a uniform weighting is generated.

    Attributes:
        Xi (list):
            Points at which objective has been evaluated.
        yi (scalar):
            Values of objective at corresponding points in `Xi`.
        models (list):
            Regression models used to fit observations and compute acquisition
            function.
        space (Space):
            An instance of `deephyper.skopt.space.Space`. Stores parameter search
            space used to sample points, bounds, and type of parameters.

    """

    def __init__(
        self,
        dimensions,
        base_estimator="gp",
        n_random_starts=None,
        n_initial_points=10,
        initial_points=None,
        initial_point_generator="random",
        n_jobs=1,
        acq_func="gp_hedge",
        acq_optimizer="auto",
        random_state=None,
        model_queue_size=None,
        acq_func_kwargs=None,
        acq_optimizer_kwargs=None,
        model_sdv=None,
        sample_max_size=-1,
        sample_strategy="quantile",
        moo_upper_bounds=None,
        moo_scalarization_strategy="Chebyshev",
        moo_scalarization_weight=None,
        objective_scaler="auto",
    ):
        args = locals().copy()
        del args["self"]
        self.specs = {"args": args, "function": "Optimizer"}
        self.rng = check_random_state(random_state)

        # Configure acquisition function

        # Store and creat acquisition function set
        self.acq_func = acq_func
        self.acq_func_kwargs = acq_func_kwargs

        allowed_acq_funcs = ["gp_hedge", "EI", "LCB", "qLCB", "PI", "EIps", "PIps"]
        if self.acq_func not in allowed_acq_funcs:
            raise ValueError(
                "expected acq_func to be in %s, got %s"
                % (",".join(allowed_acq_funcs), self.acq_func)
            )

        # treat hedging method separately
        if self.acq_func == "gp_hedge":
            self.cand_acq_funcs_ = ["EI", "LCB", "PI"]
            self.gains_ = np.zeros(3)
        else:
            self.cand_acq_funcs_ = [self.acq_func]

        if acq_func_kwargs is None:
            acq_func_kwargs = dict()
        self.eta = acq_func_kwargs.get("eta", 1.0)

        # Configure counters of points

        # Check `n_random_starts` deprecation first
        if n_random_starts is not None:
            warnings.warn(
                ("n_random_starts will be removed in favour of " "n_initial_points."),
                DeprecationWarning,
            )
            n_initial_points = n_random_starts

        if n_initial_points < 0:
            raise ValueError(
                "Expected `n_initial_points` >= 0, got %d" % n_initial_points
            )
        self._n_initial_points = n_initial_points
        self.n_initial_points_ = n_initial_points

        # Configure estimator

        # build base_estimator if doesn't exist
        if isinstance(base_estimator, str):
            base_estimator = cook_estimator(
                base_estimator,
                space=dimensions,
                random_state=self.rng.randint(0, np.iinfo(np.int32).max),
                n_jobs=n_jobs,
            )

        # check if regressor
        if not is_regressor(base_estimator) and base_estimator is not None:
            raise ValueError("%s has to be a regressor." % base_estimator)

        # treat per second acqusition function specially
        is_multi_regressor = isinstance(base_estimator, MultiOutputRegressor)
        if "ps" in self.acq_func and not is_multi_regressor:
            self.base_estimator_ = MultiOutputRegressor(base_estimator)
        else:
            self.base_estimator_ = base_estimator

        # preprocessing of target variable
        self.objective_scaler_ = objective_scaler
        self.objective_scaler = cook_objective_scaler(
            objective_scaler, self.base_estimator_
        )

        # Configure optimizer

        # decide optimizer based on gradient information
        if acq_optimizer == "auto":
            if has_gradients(self.base_estimator_):
                acq_optimizer = "lbfgs"
            else:
                acq_optimizer = "sampling"

        if acq_optimizer not in ["lbfgs", "sampling", "boltzmann_sampling"]:
            raise ValueError(
                "Expected acq_optimizer to be 'lbfgs' or "
                "'sampling' or 'softmax_sampling', got {0}".format(acq_optimizer)
            )

        if not has_gradients(self.base_estimator_) and not (
            "sampling" in acq_optimizer
        ):
            raise ValueError(
                "The regressor {0} should run with a 'sampling' "
                "acq_optimizer such as "
                "'sampling' or 'softmax_sampling'.".format(type(base_estimator))
            )
        self.acq_optimizer = acq_optimizer

        # record other arguments
        if acq_optimizer_kwargs is None:
            acq_optimizer_kwargs = dict()

        self.n_points = acq_optimizer_kwargs.get("n_points", 10000)
        self.n_restarts_optimizer = acq_optimizer_kwargs.get("n_restarts_optimizer", 5)
        self.n_jobs = acq_optimizer_kwargs.get("n_jobs", 1)
        self.update_prior = acq_optimizer_kwargs.get("update_prior", False)
        self.update_prior_quantile = acq_optimizer_kwargs.get(
            "update_prior_quantile", 0.9
        )
        self.filter_duplicated = acq_optimizer_kwargs.get("filter_duplicated", True)
        self.boltzmann_gamma = acq_optimizer_kwargs.get("boltzmann_gamma", 1)
        self.boltzmann_psucc = acq_optimizer_kwargs.get("boltzmann_psucc", 0)
        self.filter_failures = acq_optimizer_kwargs.get("filter_failures", "mean")
        self.max_failures = acq_optimizer_kwargs.get("max_failures", 100)
        self.acq_optimizer_kwargs = acq_optimizer_kwargs

        # Configure search space

        if type(dimensions) is CS.ConfigurationSpace:
            # Save the config space to do a real copy of the Optimizer
            self.config_space = dimensions
            self.config_space.seed(self.rng.get_state()[1][0])

            if isinstance(self.base_estimator_, GaussianProcessRegressor):
                raise RuntimeError("GP estimator is not available with ConfigSpace!")
        else:
            # normalize space if GP regressor
            if isinstance(self.base_estimator_, GaussianProcessRegressor):
                dimensions = normalize_dimensions(dimensions)

        # keep track of the generative model from sdv
        self.model_sdv = model_sdv

        self.space = Space(dimensions, model_sdv=self.model_sdv)

        self._initial_samples = [] if initial_points is None else initial_points[:]
        self._initial_point_generator = cook_initial_point_generator(
            initial_point_generator
        )

        if self._initial_point_generator is not None:
            transformer = self.space.get_transformer()
            self._initial_samples = (
                self._initial_samples
                + self._initial_point_generator.generate(
                    self.space.dimensions,
                    n_initial_points - len(self._initial_samples),
                    random_state=self.rng.randint(0, np.iinfo(np.int32).max),
                )
            )
            self.space.set_transformer(transformer)

        # record categorical and non-categorical indices
        self._cat_inds = []
        self._non_cat_inds = []
        for ind, dim in enumerate(self.space.dimensions):
            if isinstance(dim, Categorical):
                self._cat_inds.append(ind)
            else:
                self._non_cat_inds.append(ind)

        # Initialize storage for optimization
        if not isinstance(model_queue_size, (int, type(None))):
            raise TypeError(
                "model_queue_size should be an int or None, "
                "got {}".format(type(model_queue_size))
            )

        # For multiobjective optimization
        # TODO: would be nicer to factorize the moo code with `cook_moo_scaler(...)`

        # Initialize lower bounds for objectives
        if moo_upper_bounds is None:
            self._moo_upper_bounds = None
        elif isinstance(moo_upper_bounds, list) and all(
            [isinstance(lbi, numbers.Number) or lbi is None for lbi in moo_upper_bounds]
        ):
            self._moo_upper_bounds = moo_upper_bounds
        else:
            raise ValueError(
                f"Parameter 'moo_upper_bounds={moo_upper_bounds}' is invalid. Must be None or a list"
            )

        # Initialize the moo scalarization strategy
        moo_scalarization_strategy_allowed = list(moo_functions.keys()) + [
            f"r{s}" for s in moo_functions.keys()
        ]
        if not (
            moo_scalarization_strategy in moo_scalarization_strategy_allowed
            or isinstance(moo_scalarization_strategy, MoScalarFunction)
        ):
            raise ValueError(
                f"Parameter 'moo_scalarization_strategy={acq_func}' should have a value in {moo_scalarization_strategy_allowed}!"
            )
        self._moo_scalarization_strategy = moo_scalarization_strategy
        self._moo_scalarization_weight = moo_scalarization_weight
        self._moo_scalar_function = None

        self.max_model_queue_size = model_queue_size
        self.models = []
        self.Xi = []
        self.yi = []

        # Initialize cache for `ask` method responses
        # This ensures that multiple calls to `ask` with n_points set
        # return same sets of points. Reset to {} at every call to `tell`.
        self.cache_ = {}

        # to avoid duplicated samples
        self.sampled = []

        # for botlzmann strategy
        self._min_value = 0
        self._max_value = 0

        # parameters to stabilize the size of the dataset used to fit the surrogate model
        self._sample_max_size = sample_max_size
        self._sample_strategy = sample_strategy

        # parameters for multifidelity
        self._use_multifidelity = False
        self.bi = []
        self.bi_count = {}

    def copy(self, random_state=None):
        """Create a shallow copy of an instance of the optimizer.

        Parameters
        ----------
        random_state : int, RandomState instance, or None (default)
            Set the random state of the copy.
        """
        optimizer = Optimizer(
            dimensions=self.config_space
            if hasattr(self, "config_space")
            else self.space.dimensions,
            base_estimator=self.base_estimator_,
            n_initial_points=self.n_initial_points_,
            initial_point_generator=self._initial_point_generator,
            acq_func=self.acq_func,
            acq_optimizer=self.acq_optimizer,
            acq_func_kwargs=self.acq_func_kwargs,
            acq_optimizer_kwargs=self.acq_optimizer_kwargs,
            random_state=random_state,
            model_sdv=self.model_sdv,
            sample_max_size=self._sample_max_size,
            sample_strategy=self._sample_strategy,
            moo_upper_bounds=self._moo_upper_bounds,
            moo_scalarization_strategy=self._moo_scalarization_strategy,
            moo_scalarization_weight=self._moo_scalarization_weight,
            objective_scaler=self.objective_scaler_,
        )

        optimizer._initial_samples = self._initial_samples

        optimizer.sampled = self.sampled[:]

        if hasattr(self, "gains_"):
            optimizer.gains_ = np.copy(self.gains_)

        if self.Xi:
            optimizer._tell(self.Xi, self.yi)

        return optimizer

    def ask(self, n_points=None, strategy="cl_min"):
        """Query point or multiple points at which objective should be evaluated.

        n_points : int or None, default: None
            Number of points returned by the ask method.
            If the value is None, a single point to evaluate is returned.
            Otherwise a list of points to evaluate is returned of size
            n_points. This is useful if you can evaluate your objective in
            parallel, and thus obtain more objective function evaluations per
            unit of time.

        strategy : string, default: "cl_min"
            Method to use to sample multiple points (see also `n_points`
            description). This parameter is ignored if n_points = None.
            Supported options are `"cl_min"`, `"cl_mean"` or `"cl_max"`.

            - If set to `"cl_min"`, then constant liar strategy is used
               with lie objective value being minimum of observed objective
               values. `"cl_mean"` and `"cl_max"` means mean and max of values
               respectively. For details on this strategy see:

               https://hal.archives-ouvertes.fr/hal-00732512/document

               With this strategy a copy of optimizer is created, which is
               then asked for a point, and the point is told to the copy of
               optimizer with some fake objective (lie), the next point is
               asked from copy, it is also told to the copy with fake
               objective and so on. The type of lie defines different
               flavours of `cl_x` strategies.

        """
        if n_points is None or n_points == 1:
            x = self._ask()
            self.sampled.append(x)

            if n_points is None:
                return x
            else:
                return [x]

        if n_points > 0 and (
            self._n_initial_points > 0 or self.base_estimator_ is None
        ):
            if len(self._initial_samples) == 0:
                X = self._ask_random_points(size=n_points)
            else:
                n = min(len(self._initial_samples), n_points)
                X = self._initial_samples[:n]
                self._initial_samples = self._initial_samples[n:]
                X = X + self._ask_random_points(size=(n_points - n))
            self.sampled.extend(X)
            return X

        if self.acq_func == "qLCB":
            strategy = "qLCB"

        supported_strategies = [
            "cl_min",
            "cl_mean",
            "cl_max",
            "topk",
            "boltzmann",
            "qLCB",
        ]

        if not (isinstance(n_points, int) and n_points > 0):
            raise ValueError("n_points should be int > 0, got " + str(n_points))

        if strategy not in supported_strategies:
            raise ValueError(
                "Expected parallel_strategy to be one of "
                + str(supported_strategies)
                + ", "
                + "got %s" % strategy
            )

        # handle one-shot strategies (topk, softmax)
        if hasattr(self, "_last_X") and strategy in ["topk", "boltzmann"]:
            if strategy == "topk":
                idx = np.argsort(self._last_values)[:n_points]
                next_samples = self._last_X[idx].tolist()

                # to track sampled values and avoid duplicates
                self.sampled.extend(next_samples)

                return next_samples

            elif strategy == "boltmann":
                values = -self._last_values

                self._min_value = (
                    self._min_value
                    if self._min_value is None
                    else min(values.min(), self._min_value)
                )
                self._max_value = (
                    self._max_value
                    if self._max_value is None
                    else max(values.max(), self._max_value)
                )

                idx = [np.argmax(values)]
                max_trials = 100
                trials = 0

                while len(idx) < n_points:
                    t = len(self.sampled)
                    if t == 0:
                        beta = 0
                    else:
                        beta = (
                            self.boltzmann_gamma
                            * np.log(t)
                            / np.abs(self._max_value - self._min_value)
                        )

                    probs = boltzmann_distribution(values, beta)

                    new_idx = np.argmax(self.rng.multinomial(1, probs))

                    if (
                        self.filter_duplicated
                        and new_idx in idx
                        and trials < max_trials
                    ):
                        trials += 1
                    else:
                        idx.append(new_idx)
                        self.sampled.append(self._last_X[new_idx].tolist())

                return self._last_X[idx].tolist()
            else:
                raise ValueError(
                    f"'{strategy}' is not a valid multi-point acquisition strategy!"
                )

        # q-ACQ multi point acquisition for centralized setting
        if len(self.models) > 0 and strategy == "qLCB":
            X_s = self.space.rvs(
                n_samples=self.n_points, random_state=self.rng, n_jobs=self.n_jobs
            )
            X_s = self._filter_duplicated(X_s)
            X_c = self.space.imp_const.fit_transform(
                self.space.transform(X_s)
            )  # candidates

            mu, std = self.models[-1].predict(X_c, return_std=True)
            kappa = self.acq_func_kwargs.get("kappa", 1.96)
            kappas = self.rng.exponential(kappa, size=n_points - 1)
            X = [self._next_x]
            for kappa in kappas:
                values = mu - kappa * std
                idx = np.argmin(values)
                X.append(X_s[idx])
            return X

        # Caching the result with n_points not None. If some new parameters
        # are provided to the ask, the cache_ is not used.
        if (n_points, strategy) in self.cache_:
            return self.cache_[(n_points, strategy)]

        # Copy of the optimizer is made in order to manage the
        # deletion of points with "lie" objective (the copy of
        # optimizer is simply discarded)
        opt = self.copy(random_state=self.rng.randint(0, np.iinfo(np.int32).max))

        X = []
        for i in range(n_points):
            x = opt.ask()
            self.sampled.append(x)
            X.append(x)

            # the optimizer copy `opt` is discarded anyway
            if i == n_points - 1:
                break

            ti_available = "ps" in self.acq_func and len(opt.yi) > 0
            ti = [t for (_, t) in opt.yi] if ti_available else None

            opt_yi = self._filter_failures(opt.yi)

            if strategy == "cl_min":
                y_lie = np.min(opt_yi, axis=0) if opt_yi else 0.0  # CL-min lie
                t_lie = np.min(ti) if ti is not None else log(sys.float_info.max)
            elif strategy == "cl_mean":
                y_lie = np.mean(opt_yi, axis=0) if opt_yi else 0.0  # CL-mean lie
                t_lie = np.mean(ti) if ti is not None else log(sys.float_info.max)
            else:
                y_lie = np.max(opt_yi, axis=0) if opt_yi else 0.0  # CL-max lie
                t_lie = np.max(ti) if ti is not None else log(sys.float_info.max)

            # Lie to the optimizer.
            if "ps" in self.acq_func:
                # Use `_tell()` instead of `tell()` to prevent repeated
                # log transformations of the computation times.
                opt._tell(x, (y_lie, t_lie))
            else:
                opt._tell(x, y_lie)

        self.cache_ = {(n_points, strategy): X}  # cache_ the result

        return X

    def _filter_duplicated(self, samples):
        """Filter out duplicated values in ``samples``.

        Args:
            samples (list): the list of samples to filter.

        Returns:
            list: the filtered list of samples
        """

        if self.filter_duplicated:
            # check duplicated values

            if hasattr(self, "config_space"):
                hps_names = self.config_space.get_hyperparameter_names()
            else:
                hps_names = self.space.dimension_names

            df_samples = pd.DataFrame(data=samples, columns=hps_names, dtype="O")
            df_samples = df_samples[~df_samples.duplicated(keep="first")]

            if len(self.sampled) > 0:
                df_history = pd.DataFrame(data=self.sampled, columns=hps_names)
                df_merge = pd.merge(df_samples, df_history, on=None, how="inner")
                df_samples = pd.concat([df_samples, df_merge])
                df_samples = df_samples[~df_samples.duplicated(keep=False)]

            if len(df_samples) > 0:
                samples = df_samples.values.tolist()

        return samples

    def _filter_failures(self, yi):
        """Filter or replace failed objectives.

        Args:
            yi (list): a list of objectives.

        Returns:
            list: the filtered list.
        """
        if self.filter_failures in ["mean", "max"]:
            yi_no_failure = [v for v in yi if v != OBJECTIVE_VALUE_FAILURE]

            # when yi_no_failure is empty all configurations are failures
            if len(yi_no_failure) == 0:
                if len(yi) >= self.max_failures:
                    raise ExhaustedFailures
                # constant value for the acq. func. to return anything
                yi_failed_value = 0
            elif self.filter_failures == "mean":
                yi_failed_value = np.mean(yi_no_failure).tolist()
            else:
                yi_failed_value = np.max(yi_no_failure).tolist()

            yi = [v if v != OBJECTIVE_VALUE_FAILURE else yi_failed_value for v in yi]

        return yi

    def _sample(self, X, y):
        X = np.asarray(X, dtype="O")
        y = np.asarray(y)
        size = y.shape[0]

        if self._sample_max_size > 0 and size > self._sample_max_size:
            if self._sample_strategy == "quantile":
                quantiles = np.quantile(y, [0.10, 0.25, 0.50, 0.75, 0.90])
                int_size = self._sample_max_size // (len(quantiles) + 1)

                Xs, ys = [], []
                for i in range(len(quantiles) + 1):
                    if i == 0:
                        s = y < quantiles[i]
                    elif i == len(quantiles):
                        s = quantiles[i - 1] <= y
                    else:
                        s = (quantiles[i - 1] <= y) & (y < quantiles[i])

                    idx = np.where(s)[0]
                    idx = np.random.choice(idx, size=int_size, replace=True)
                    Xi = X[idx]
                    yi = y[idx]
                    Xs.append(Xi)
                    ys.append(yi)

                X = np.concatenate(Xs, axis=0)
                y = np.concatenate(ys, axis=0)

        X = X.tolist()
        y = y.tolist()
        return X, y

    def _ask_random_points(self, size=None):
        samples = self.space.rvs(
            n_samples=self.n_points, random_state=self.rng, n_jobs=self.n_jobs
        )

        samples = self._filter_duplicated(samples)

        if size is None:
            return samples[0]
        else:
            return samples[:size]

    def _ask(self):
        """Suggest next point at which to evaluate the objective.

        Return a random point while not at least `n_initial_points`
        observations have been `tell`ed, after that `base_estimator` is used
        to determine the next point.
        """
        if self._n_initial_points > 0 or self.base_estimator_ is None:
            # this will not make a copy of `self.rng` and hence keep advancing
            # our random state.
            if len(self._initial_samples) == 0:
                return self._ask_random_points()
            else:
                # The samples are evaluated starting form initial_samples[0]
                x = self._initial_samples[0]
                self._initial_samples = self._initial_samples[1:]
                return x

        else:
            if not self.models:
                raise RuntimeError(
                    "Random evaluations exhausted and no " "model has been fit."
                )

            next_x = self._next_x
            if next_x is not None:
                if not self.space.is_config_space:
                    min_delta_x = min(
                        [self.space.distance(next_x, xi) for xi in self.Xi]
                    )
                    if abs(min_delta_x) <= 1e-8:
                        warnings.warn(
                            "The objective has been evaluated " "at this point before."
                        )

            # return point computed from last call to tell()
            return next_x

    def tell(self, x, y, fit=True):
        """Record an observation (or several) of the objective function.

        Provide values of the objective function at points suggested by
        `ask()` or other points. By default a new model will be fit to all
        observations. The new model is used to suggest the next point at
        which to evaluate the objective. This point can be retrieved by calling
        `ask()`.

        To add observations without fitting a new model set `fit` to False.

        To add multiple observations in a batch pass a list-of-lists for `x`
        and a list of scalars for `y`.

        Parameters
        ----------
        x : list or list-of-lists
            Point at which objective was evaluated.

        y : scalar or list
            Value of objective at `x`.

        fit : bool, default: True
            Fit a model to observed evaluations of the objective. A model will
            only be fitted after `n_initial_points` points have been told to
            the optimizer irrespective of the value of `fit`.
        """
        if self.space.is_config_space:
            pass
        else:
            check_x_in_space(x, self.space)

        self._check_y_is_valid(x, y)

        # take the logarithm of the computation times
        if "ps" in self.acq_func:
            if is_2Dlistlike(x):
                y = [[val, log(t)] for (val, t) in y]
            elif is_listlike(x):
                y = list(y)
                y[1] = log(y[1])

        return self._tell(x, y, fit=fit)

    def _tell(self, x, y, fit=True):
        """Perform the actual work of incorporating one or more new points.
        See `tell()` for the full description.

        This method exists to give access to the internals of adding points
        by side stepping all input validation and transformation."""
        if "ps" in self.acq_func:
            if is_2Dlistlike(x):
                self.Xi.extend(x)
                self.yi.extend(y)
                self._n_initial_points -= len([v for v in self.yi if v != "F"])
            elif is_listlike(x):
                self.Xi.append(x)
                self.yi.append(y)
                if y != "F":
                    self._n_initial_points -= 1
        # if y isn't a scalar it means we have been handed a batch of points
        elif is_listlike(y) and is_2Dlistlike(x):
            self.Xi.extend(x)
            self.yi.extend(y)
            self._n_initial_points -= len([v for v in self.yi if v != "F"])
        elif is_listlike(x):
            self.Xi.append(x)
            self.yi.append(y)
            if y != "F":
                self._n_initial_points -= 1
        else:
            raise ValueError(
                "Type of arguments `x` (%s) and `y` (%s) "
                "not compatible." % (type(x), type(y))
            )

        # optimizer learned something new - discard cache
        self.cache_ = {}

        # after being "told" n_initial_points we switch from sampling
        # random points to using a surrogate model
        if fit and self._n_initial_points <= 0 and self.base_estimator_ is not None:
            transformed_bounds = self.space.transformed_bounds
            est = clone(self.base_estimator_)

            yi = self.yi

            # Convert multiple objectives to single scalar
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                if any(isinstance(v, list) for v in yi):
                    # Multi-Objective Optimization

                    yi = self._moo_scalarize(yi)

                else:
                    # Single-Objective Optimization

                    if "F" in yi:
                        # ! dtype="O" is key to avoid converting data to string
                        yi = np.asarray(yi, dtype="O")
                        mask_no_failures = np.where(yi != "F")
                        yi[mask_no_failures] = (
                            self.objective_scaler.fit_transform(
                                np.asarray(yi[mask_no_failures].tolist()).reshape(-1, 1)
                            )
                            .reshape(-1)
                            .tolist()
                        )
                        yi = yi.tolist()
                    else:
                        yi = (
                            self.objective_scaler.fit_transform(np.reshape(yi, (-1, 1)))
                            .reshape(-1)
                            .tolist()
                        )

            # Handle failures
            yi = self._filter_failures(yi)

            # handle size of the sample fit to the estimator
            Xi, yi = self._sample(self.Xi, yi)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # preprocessing of input space
                Xtt = self.space.imp_const.fit_transform(self.space.transform(Xi))
                Xtt = np.asarray(Xtt)

                # fit surrogate model
                est.fit(Xtt, yi)

                # update prior
                if self.update_prior:
                    self.space.update_prior(Xtt, yi, q=self.update_prior_quantile)

            if self.max_model_queue_size is None:
                self.models.append(est)
            elif len(self.models) < self.max_model_queue_size:
                self.models.append(est)
            else:
                # Maximum list size obtained, remove oldest model.
                self.models.pop(0)
                self.models.append(est)

            if hasattr(self, "next_xs_") and self.acq_func == "gp_hedge":
                self.gains_ -= est.predict(np.vstack(self.next_xs_))

            # even with BFGS as optimizer we want to sample a large number
            # of points and then pick the best ones as starting points
            X_s = self.space.rvs(
                n_samples=self.n_points, random_state=self.rng, n_jobs=self.n_jobs
            )

            X_s = self._filter_duplicated(X_s)

            X = self.space.imp_const.fit_transform(self.space.transform(X_s))

            self.next_xs_ = []
            for cand_acq_func in self.cand_acq_funcs_:
                values = _gaussian_acquisition(
                    X=X,
                    model=est,
                    y_opt=np.min(yi),
                    acq_func=cand_acq_func,
                    acq_func_kwargs=self.acq_func_kwargs,
                )

                # cache these values in case the strategy of ask is one-shot
                self._last_X = X
                self._last_values = values

                # Find the minimum of the acquisition function by randomly
                # sampling points from the space
                if self.acq_optimizer == "sampling":
                    next_x = X[np.argmin(values)]

                elif self.acq_optimizer == "boltzmann_sampling":
                    p = self.rng.uniform()
                    if p <= self.boltzmann_psucc:
                        next_x = X[np.argmin(values)]
                    else:
                        values = -values

                        self._min_value = (
                            self._min_value
                            if self._min_value is None
                            else min(values.min(), self._min_value)
                        )
                        self._max_value = (
                            self._max_value
                            if self._max_value is None
                            else max(values.max(), self._max_value)
                        )

                        t = len(self.Xi)
                        if t == 0:
                            beta = 0
                        else:
                            beta = (
                                self.boltzmann_gamma
                                * np.log(t)
                                / np.abs(self._max_value - self._min_value)
                            )

                        probs = boltzmann_distribution(values, beta)

                        idx = np.argmax(self.rng.multinomial(1, probs))

                        next_x = X[idx]

                # Use BFGS to find the mimimum of the acquisition function, the
                # minimization starts from `n_restarts_optimizer` different
                # points and the best minimum is used
                elif self.acq_optimizer == "lbfgs":
                    x0 = X[np.argsort(values)[: self.n_restarts_optimizer]]

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        results = Parallel(n_jobs=self.n_jobs)(
                            delayed(fmin_l_bfgs_b)(
                                gaussian_acquisition_1D,
                                x,
                                args=(
                                    est,
                                    np.min(yi),
                                    cand_acq_func,
                                    self.acq_func_kwargs,
                                ),
                                bounds=transformed_bounds,
                                approx_grad=False,
                                maxiter=20,
                            )
                            for x in x0
                        )

                    cand_xs = np.array([r[0] for r in results])
                    cand_acqs = np.array([r[1] for r in results])
                    next_x = cand_xs[np.argmin(cand_acqs)]

                # lbfgs should handle this but just in case there are
                # precision errors.
                if not self.space.is_categorical:
                    if not self.space.is_config_space:
                        transformed_bounds = np.asarray(transformed_bounds)
                        next_x = np.clip(
                            next_x,
                            transformed_bounds[:, 0],
                            transformed_bounds[:, 1],
                        )

                self.next_xs_.append(next_x)

            if self.acq_func == "gp_hedge":
                logits = np.array(self.gains_)
                logits -= np.max(logits)
                exp_logits = np.exp(self.eta * logits)
                probs = exp_logits / np.sum(exp_logits)
                next_x = self.next_xs_[np.argmax(self.rng.multinomial(1, probs))]
            else:
                next_x = self.next_xs_[0]

            # note the need for [0] at the end
            self._next_x = self.space.inverse_transform(next_x.reshape((1, -1)))[0]

        # Pack results
        result = create_result(
            self.Xi, self.yi, self.space, self.rng, models=self.models
        )

        result.specs = self.specs
        return result

    def _check_y_is_valid(self, x, y):
        """Check if the shape and types of x and y are consistent."""

        if "ps" in self.acq_func:
            if is_2Dlistlike(x):
                if not (np.ndim(y) == 2 and np.shape(y)[1] == 2):
                    raise TypeError("expected y to be a list of (func_val, t)")
            elif is_listlike(x):
                if not (np.ndim(y) == 1 and len(y) == 2):
                    raise TypeError("expected y to be (func_val, t)")

        # if y isn't a scalar it means we have been handed a batch of points
        elif is_listlike(y) and is_2Dlistlike(x):
            for y_value in y:
                if (
                    not isinstance(y_value, numbers.Number)
                    and not is_listlike(y_value)
                    and y_value != OBJECTIVE_VALUE_FAILURE
                ):
                    raise ValueError("expected y to be a 1-D or 2-D list of scalars")

        elif is_listlike(x):
            if not isinstance(y, numbers.Number) and not is_listlike(y):
                raise ValueError("`func` should return a scalar or tuple of scalars")

        else:
            raise ValueError(
                "Type of arguments `x` (%s) and `y` (%s) "
                "not compatible." % (type(x), type(y))
            )

    def run(self, func, n_iter=1):
        """Execute ask() + tell() `n_iter` times"""
        for _ in range(n_iter):
            x = self.ask()
            self.tell(x, func(x))

        result = create_result(
            self.Xi, self.yi, self.space, self.rng, models=self.models
        )
        result.specs = self.specs
        return result

    def update_next(self):
        """Updates the value returned by opt.ask(). Useful if a parameter
        was updated after ask was called."""
        self.cache_ = {}
        # Ask for a new next_x.
        # We only need to overwrite _next_x if it exists.
        if hasattr(self, "_next_x"):
            opt = self.copy(random_state=self.rng)
            self._next_x = opt._next_x

    def get_result(self):
        """Returns the same result that would be returned by opt.tell()
        but without calling tell

        Returns
        -------
        res : `OptimizeResult`, scipy object
            OptimizeResult instance with the required information.

        """
        result = create_result(
            self.Xi, self.yi, self.space, self.rng, models=self.models
        )
        result.specs = self.specs
        return result

    def _moo_scalarize(self, yi):
        has_failures = "F" in yi
        if has_failures:
            yi = np.asarray(yi, dtype="O")
            mask_no_failures = np.where(yi != "F")
            yi_filtered = yi[mask_no_failures]
        else:
            yi_filtered = np.asarray(yi)

        n_objectives = 1 if np.ndim(yi_filtered[0]) == 0 else len(yi_filtered[0])

        # Fit scaler
        self.objective_scaler.fit(yi_filtered)

        # Penality
        if self._moo_upper_bounds is not None:
            # y_max = yi_filtered.max(axis=0)
            # y_min = yi_filtered.min(axis=0)
            # upper_bounds = [
            #     m if b is None else b for m, b in zip(y_max, self._moo_upper_bounds)
            # ]
            # augmented_lower_bounds = [
            #     m if b is None else b for m, b in zip(y_min, self._moo_upper_bounds)
            # ]
            upper_bounds = [0.0 if b is None else b for b in self._moo_upper_bounds]

            # ! Strategy 1: penalty after scaling
            # upper_bounds = self.objective_scaler.transform([upper_bounds])[0]
            # yi_filtered = self.objective_scaler.transform(yi_filtered)
            # penalty = np.sum(2 * np.maximum(yi_filtered - upper_bounds, 0), axis=1)
            # print("-> y:", yi_filtered[-1])
            # print("-> p:", penalty[-1])
            # print()
            # yi_filtered = np.add(yi_filtered.T, penalty).T

            # ! Strategy 2: penalty before scaling
            # penalty = np.maximum(yi_filtered - upper_bounds, 0)
            # yi_filtered = yi_filtered + penalty
            # yi_filtered = self.objective_scaler.transform(yi_filtered)

            # ! Strategy 3: replace values above upper bounds with y_max
            # yi_filtered = self.objective_scaler.transform(yi_filtered)
            # y_max_scaled = self.objective_scaler.transform([y_max])[0]
            # upper_bounds = self.objective_scaler.transform([upper_bounds])[0]

            # *S.3.A: Replacing only the objective which breaks the bound
            # mask = yi_filtered > upper_bounds
            # yi_filtered[mask] = np.tile(y_max_scaled, yi_filtered.shape[0]).reshape(
            #     yi_filtered.shape
            # )[mask]

            # *S.3.B: Replacing all objectives if any of them breaks the bound
            # mask = (yi_filtered > upper_bounds).any(axis=1)
            # yi_filtered[mask] = y_max_scaled

            # Strategy 4: "leaky" penalty after scaling
            penalty_param = 2.0
            leak_param = self.rng.uniform()

            upper_bounds = self.objective_scaler.transform([upper_bounds])[0]
            augmented_lower_bounds = upper_bounds.copy()
            for i, b in enumerate(self._moo_upper_bounds):
                if b is None:
                    upper_bounds[i] = np.infty  # Allow to go higher
                    augmented_lower_bounds[i] = -np.infty  # No leaky rewards
            yi_filtered = self.objective_scaler.transform(yi_filtered)
            penalty = np.sum(
                penalty_param * np.maximum(yi_filtered - upper_bounds, 0), axis=1
            ) + np.sum(
                leak_param * np.minimum(yi_filtered - augmented_lower_bounds, 0), axis=1
            )
            # print("-> y:", yi_filtered[-1])
            # print("-> b:", upper_bounds)
            # print("-> p:", penalty[-1])
            # print()
            yi_filtered = np.add(yi_filtered.T, penalty).T
        else:
            yi_filtered = self.objective_scaler.transform(yi_filtered)

        # The object is created here because the number of objectives `n_objectives`
        # is inferred from observed data.
        if self._moo_scalar_function is None:
            if isinstance(self._moo_scalarization_strategy, str):
                self._moo_scalar_function = moo_functions[
                    self._moo_scalarization_strategy
                ](
                    n_objectives=n_objectives,
                    weight=self._moo_scalarization_weight,
                    random_state=self.rng,
                )
            elif isinstance(self._moo_scalarization_strategy, MoScalarFunction):
                self._moo_scalar_function = self._moo_scalarization_strategy

        self._moo_scalar_function.update_weight()

        # compute normalization constants
        self._moo_scalar_function.normalize(yi_filtered)
        yi_filtered = [self._moo_scalar_function.scalarize(y) for y in yi_filtered]

        if has_failures:
            yi[mask_no_failures] = yi_filtered
            return yi.tolist()
        else:
            return yi_filtered
