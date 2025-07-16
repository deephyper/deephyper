import abc
import logging
from typing import Any, Dict, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.stats as ss
from pydantic import BaseModel, ConfigDict, Field
from sklearn.base import BaseEstimator, is_regressor
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    ParameterGrid,
)
from sklearn.utils import check_random_state

from deephyper.evaluator import HPOJob
from deephyper.hpo._problem import HpProblem, convert_to_skopt_space
from deephyper.skopt.optimizer.acq_optimizer.pymoo_ga import GAPymooAcqOptimizer
from deephyper.skopt.utils import cook_estimator

logger = logging.getLogger(__name__)


class Solution(BaseModel):
    """Represents the estimated solution of a search.

    Attributes:
        parameters: The parameter configuration of the solution
        objective: The objective value(s) of the solution
        objective_std_al: Aleatoric uncertainty (if available)
        objective_std_ep: Epistemic uncertainty (if available)
    """

    parameters: Any = Field(description="Parameter configuration")
    objective: Any = Field(description="Objective value(s)")
    objective_std: Optional[float] = Field(None, description="Total uncertainty")
    objective_std_al: Optional[float] = Field(None, description="Aleatoric uncertainty")
    objective_std_ep: Optional[float] = Field(None, description="Epistemic uncertainty")

    model_config = ConfigDict(extra="allow")


class SolutionSelection(abc.ABC):
    """Base class for search solution selection strategies.

    This abstract base class defines the interface for different strategies
    to select the best solution from a set of evaluated configurations.
    """

    def __init__(self):
        self.solution = Solution(parameters=None, objective=None)
        # Value set by the SearchHistory
        self.num_objective: Optional[int] = None

    def update(self, jobs: Sequence[HPOJob]) -> None:
        """Update the solution based on new job results.

        Args:
            jobs: Sequence of completed HPO jobs
        """
        # Currently skipped for multi-objective optimization
        if self.num_objective is not None and self.num_objective > 1:
            logger.debug("Skipping solution update for multi-objective optimization")
            return

        if not jobs:
            logger.warning("No jobs provided for solution update")
            return

        self._update(jobs)
        logger.info(f"Updated search solution: {self.solution}")

    @abc.abstractmethod
    def _update(self, jobs: Sequence[HPOJob]) -> None:
        """Internal method to update the solution - must be implemented by subclasses."""
        ...


class ArgMaxObsSelection(SolutionSelection):
    """Selects the best solution based on maximum observed objective(s).

    This strategy simply picks the configuration with the highest observed
    objective value among all evaluated configurations.
    """

    def _update(self, jobs: Sequence[HPOJob]) -> None:
        """Update solution by selecting the job with maximum objective."""
        for job in jobs:
            # Handle failed evaluations
            if isinstance(job.objective, str):
                logger.debug(f"Skipping failed job with objective: {job.objective}")
                continue

            # Initialize or update solution if better objective found
            if self.solution.objective is None or (
                job.objective is not None and job.objective >= self.solution.objective
            ):
                self.solution = Solution(parameters=job.args, objective=job.objective)


def calibration_error(y, y_mean, y_var):
    empirical_error = np.square(y - y_mean)
    return np.mean(np.abs(empirical_error - y_var))


def gaussian_ll(y, mean, var):
    nll = 0.5 * np.mean(np.log(2 * np.pi * var) + np.square(y - mean) / var)
    return -nll


def gaussian_ll_score(model, X, y, eps=1e-6):
    try:
        y_mean, y_std, _ = model.predict(X, return_std=True, disentangled_std=True)
    except TypeError:
        y_mean, y_std = model.predict(X, return_std=True)
    y_var = np.maximum(y_std**2, eps)  # Ensure numerical stability
    return gaussian_ll(y, y_mean, y_var)


class ArgMaxEstSelection(SolutionSelection):
    """Selects solution using a surrogate model and acquisition optimizer.

    This strategy fits a surrogate model to the observed data and uses
    optimization to find the configuration that maximizes the predicted objective.
    """

    def __init__(
        self,
        problem: HpProblem,
        random_state: Optional[int] = None,
        model: Union[str, BaseEstimator] = "ET",
        model_kwargs: Optional[Dict[str, Any]] = None,
        optimizer: Literal["sampling", "ga"] = "sampling",
        filter_failures: Literal["mean", "max"] = "mean",
    ):
        """Initialize the estimator-based selection strategy.

        Args:
            problem: The hyperparameter optimization problem
            random_state: Random state for reproducibility
            model: Surrogate model name or instance
            model_kwargs: Additional arguments for model initialization
            optimizer: Optimization strategy for acquisition function
            filter_failures: Strategy for handling failed evaluations
        """
        super().__init__()
        self.problem = problem
        self.rng = check_random_state(random_state)
        self.optimizer = optimizer
        self.filter_failures = filter_failures

        self.parameters_list = []
        self.objective_list = []

        # Set default model parameters for Extra Trees
        if model == "ET" and model_kwargs is None:
            model_kwargs = {
                # "splitter": "random",
                "bootstrap": True,
                "min_samples_leaf": 3,
                "min_samples_split": 2,
                "n_estimators": 100,
                "n_jobs": 1,
            }

        elif model_kwargs is None:
            model_kwargs = {}

        self.skopt_space = convert_to_skopt_space(problem.space, surrogate_model=model)

        # Initialize surrogate model
        if isinstance(model, str):
            model = cook_estimator(
                model,
                space=self.skopt_space,
                random_state=self.rng.randint(0, np.iinfo(np.int32).max),
                **model_kwargs,
            )

        # if not is_regressor(model):
        #     raise ValueError(f"Model {model.__class__.__name__} must be a regressor.")

        self.model = model
        self.count_tune_model = 0

    def _fit_and_tune_model(self, X, y):
        param_grid = []
        # p_grid = {
        #     "n_estimators": [100],
        #     "bootstrap": [True],
        #     "min_samples_leaf": [1, 2, 4, 8],
        #     "min_samples_split": [2, 4, 8],
        #     "max_samples": [0.8, 0.9, 1.0],  # Only used if bootstrap=True
        #     "max_depth": [None, 10, 20],
        #     "max_features": ["sqrt", "log2", 0.5],
        # }

        # p_grid = {
        #     "bootstrap": [True],
        #     "min_samples_leaf": [1, 2, 4, 8, 12],
        #     "min_samples_split": [2, 4, 8, 12],
        #     "max_samples": [0.8, 0.9, 1.0],
        # }
        # param_grid += list(ParameterGrid(p_grid))
        # p_grid = {
        #     "bootstrap": [False],
        #     "min_samples_leaf": [1, 2, 4, 8, 12],
        #     "min_samples_split": [2, 4, 8, 12],
        # }
        p_grid = {
            "n_estimators": [100],
            "bootstrap": [False],
            "min_samples_leaf": [1, 2, 4, 8, 16],
            "min_samples_split": [2, 4, 8, 16, 32, 64],
            "max_depth": [None, 10, 20],
            "max_features": ["sqrt"],
        }
        # p_grid = {
        #     "learning_rate": [0.05],
        #     "max_depth": [2, 5],
        #     "min_samples_leaf": [1, 9],
        # }
        param_grid += list(ParameterGrid(p_grid))
        clf = GridSearchCV(
            estimator=self.model,
            param_grid=p_grid,
            cv=KFold(n_splits=4, shuffle=True),
            refit=True,
            # scoring="r2",
            scoring=gaussian_ll_score,
        )
        clf.fit(X, y)
        self.model = clf.best_estimator_

        try:
            y_mean, y_std, _ = self.model.predict(X, return_std=True, disentangled_std=True)
        except TypeError:
            y_mean, y_std = self.model.predict(X, return_std=True)

        r2_model = 1 - np.mean((y - y_mean) ** 2) / np.var(y)
        r2_ub = 1 - np.mean(y_std**2) / np.var(y)
        print("Final best model parameters:", clf.best_params_)
        print(f"Model     LL : {clf.best_score_:.3f}")
        print(f"Model     R2 : {r2_model:.3f}")
        print(f"Up. bound R2 : {r2_ub:.3f}")

        # The following evaluates the quality of the AL STD estimates
        p = ss.pearsonr(y_std**2, (y - y_mean) ** 2)
        print("Corr: ", p)

    def _filter_failures(self, yi: Sequence[Any]) -> Tuple[bool, Sequence[float]]:
        """Filter or replace failed objectives.

        Args:
            yi: List of objectives (may contain failure indicators)

        Returns:
            Tuple of (has_success, filtered_objectives)
        """
        has_success = True

        if self.filter_failures in ["mean", "max"]:
            yi_no_failure = [v for v in yi if v != "F"]

            # When all configurations are failures
            if len(yi_no_failure) == 0:
                yi_failed_value = 0.0
                has_success = False
                logger.warning("All evaluations failed, using default value of 0")
            else:
                if self.filter_failures == "mean":
                    yi_failed_value = float(np.mean(yi_no_failure))
                else:  # max
                    yi_failed_value = float(np.max(yi_no_failure))

                logger.debug(
                    f"Replacing {len(yi) - len(yi_no_failure)} failures with {yi_failed_value}"
                )

            yi = [v if v != "F" else yi_failed_value for v in yi]

        return has_success, yi

    def _update(self, jobs: Sequence[HPOJob]) -> None:
        """Update solution using surrogate model prediction."""
        if not jobs:
            return

        # Extract new data
        new_parameters = [list(job.args.values()) for job in jobs]
        self.objective_list.extend(job.objective for job in jobs)

        # Transform parameters to model space
        try:
            transformed_parameters = self.skopt_space.transform(new_parameters)
            self.parameters_list.extend(transformed_parameters)
        except Exception as e:
            logger.error(f"Failed to transform parameters: {e}")
            return

        # Handle failures
        has_success, objective_list = self._filter_failures(self.objective_list)
        if not has_success:
            logger.warning("No successful evaluations available for model fitting")
            return

        X, y = self.parameters_list, objective_list

        # Fit surrogate model
        if len(objective_list) >= 100 + self.count_tune_model:
            print("Tuning selection model...")
            self._fit_and_tune_model(X, y)
            self.count_tune_model = len(objective_list)
        else:
            try:
                self.model.fit(X, y)
                logger.debug(f"Fitted model with {len(self.parameters_list)} samples")
            except Exception as e:
                logger.error(f"Failed to fit surrogate model: {e}")
                return

        # Optimize acquisition function
        optimize_fn = {
            "sampling": self.optimize_sampling,
            "ga": self.optimize_ga,
        }.get(self.optimizer)

        if optimize_fn is None:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")

        try:
            result = optimize_fn()
            if len(result) == 4:
                parameters, objective_mean, objective_std_al, objective_std_ep = result
                self.solution = Solution(
                    parameters=dict(zip(self.skopt_space.dimension_names, parameters)),
                    objective=objective_mean,
                    objective_std_al=objective_std_al,
                    objective_std_ep=objective_std_ep,
                )
            else:
                parameters, objective_mean, objective_std = result
                self.solution = Solution(
                    parameters=dict(zip(self.skopt_space.dimension_names, parameters)),
                    objective=objective_mean,
                    objective_std=objective_std,
                )
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise e

    def acq_func(self, y_mean, y_std=None, y_std_al=None, y_std_ep=None):
        # MAXIMIZED
        if y_std is None and y_std_al is None and y_std_ep is None:
            return y_mean

        assert y_std is not None or (y_std_al is not None and y_std_ep is not None)

        if y_std is not None:
            return y_mean - 1.96 * y_std

        return y_mean - 1.96 * y_std_ep

    def optimize_sampling(self, n_samples: int = 10_000) -> Tuple[Any, float, float, float]:
        """Optimize using random sampling.

        Args:
            n_samples: Number of random samples to evaluate

        Returns:
            Tuple of (best_parameters, objective_mean, std_al, std_ep)
        """
        samples = self.skopt_space.rvs(n_samples=n_samples, random_state=self.rng, n_jobs=1)
        transformed = self.skopt_space.transform(samples)

        y_pred, y_std, y_std_al, y_std_ep = None, None, None, None
        try:
            y_pred, y_std_al, y_std_ep = self.model.predict(
                transformed,
                return_std=True,
                disentangled_std=True,
            )
        except TypeError:
            # Fallback if disentangled_std is not supported
            y_pred, y_std = self.model.predict(
                transformed,
                return_std=True,
            )
        scores = self.acq_func(y_pred, y_std, y_std_al, y_std_ep)
        idx = np.argmax(scores)

        # TODO: Experimental - select candidates with lower epistemic uncertainty
        # idx = np.argmax(y_pred - 1.96 * y_std_ep)

        best_parameters = self.skopt_space.inverse_transform([transformed[idx]])[0]

        if y_std is not None:
            return best_parameters, float(y_pred[idx]), float(y_std[idx])
        else:
            return best_parameters, float(y_pred[idx]), float(y_std_al[idx]), float(y_std_ep[idx])

    def optimize_ga(
        self,
        n_samples: int = 10_000,
        pop_size: int = 100,
        xtol: float = 1e-8,
        ftol: float = 1e-6,
        period: int = 30,
        n_max_gen: int = 1000,
    ) -> Tuple[Any, float]:
        """Optimize using genetic algorithm.

        Args:
            n_samples: Number of initial samples
            pop_size: Population size for GA
            xtol: Tolerance for parameter convergence
            ftol: Tolerance for objective convergence
            period: Period for convergence checking
            n_max_gen: Maximum number of generations

        Returns:
            Tuple of (best_parameters, objective_value)
        """
        # Generate initial population
        samples = self.skopt_space.rvs(n_samples=n_samples, random_state=self.rng, n_jobs=1)
        transformed = self.skopt_space.transform(samples)
        y_pred = self.model.predict(transformed)

        # Select top candidates for GA initialization
        top_idx = np.argsort(y_pred)[-pop_size:]

        acq_opt = GAPymooAcqOptimizer(
            space=self.skopt_space,
            x_init=transformed[top_idx],
            y_init=y_pred[top_idx],
            pop_size=pop_size,
            random_state=self.rng.randint(0, np.iinfo(np.int32).max),
            termination_kwargs={
                "xtol": xtol,
                "ftol": ftol,
                "period": period,
                "n_max_gen": n_max_gen,
            },
        )

        def acq_func(x):
            y_mean, y_std, y_std_al, y_std_ep = None, None, None, None
            try:
                y_mean, y_std_al, y_std_ep = self.model.predict(
                    x, return_std=True, disentangled_std=True
                )
            except TypeError:
                y_mean, y_std_al, y_std_ep = self.model.predict(
                    x,
                    return_std=True,
                )
            return -self.acq_func(y_mean, y_std, y_std_al, y_std_ep)

        # Minimize negative prediction (maximize prediction)
        x_sol = acq_opt.minimize(acq_func)
        best_parameters = self.skopt_space.inverse_transform([x_sol])[0]

        # Get prediction and uncertainty estimates
        try:
            y_mean, y_std_al, y_std_ep = self.model.predict(
                [x_sol], return_std=True, disentangled_std=True
            )
            return best_parameters, float(y_mean[0]), float(y_std_al[0]), float(y_std_ep[0])
        except TypeError:
            y_mean, y_std = self.model.predict(
                [x_sol],
                return_std=True,
            )
            return best_parameters, float(y_mean[0]), float(y_std[0])
