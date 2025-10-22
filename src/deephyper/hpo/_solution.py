import abc
import logging
from typing import Any, Literal, Optional, Sequence, Tuple, Union

import numpy as np
from numpy import ndarray
from pydantic import BaseModel, ConfigDict, Field
from sklearn.base import BaseEstimator
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
)
from sklearn.utils import check_random_state

from deephyper.evaluator import HPOJob
from deephyper.hpo._problem import HpProblem, convert_to_skopt_space
from deephyper.skopt.optimizer.acq_optimizer.pymoo_ga import GAPymooAcqOptimizer
from deephyper.skopt.utils import cook_estimator

logger = logging.getLogger(__name__)


class Solution(BaseModel):
    """Represents the solution of a search.

    Attributes:
        parameters: The parameter configuration of the solution.
        objective: The objective value(s) of the solution.
        objective_std: Total uncertainty (if available).
        objective_std_al: Aleatoric uncertainty (if available).
        objective_std_ep: Epistemic uncertainty (if available).
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
    to select the best solution from a set of evaluated parameter configurations.
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
    objective value among all evaluated configurations. If multiple maximums
    exists it will select the latest received result.
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


SCORING_FUNC_GRID_SEARCH = {
    "gaussian_nll": gaussian_ll_score,
    "r2": "r2",
}


class ArgMaxEstSelection(SolutionSelection):
    """Selects solution using a surrogate model and acquisition optimizer.

    This strategy fits a surrogate model to the observed data and uses
    optimization to find the configuration that maximizes the predicted objective.
    """

    def __init__(
        self,
        problem: HpProblem,
        random_state: int | None = None,
        model: Union[str, BaseEstimator] = "RF",
        model_kwargs: dict[str, Any] | None = None,
        optimizer: Literal["sampling", "ga"] = "ga",
        filter_failures: Literal["mean", "max"] = "mean",
        model_grid_search: bool = True,
        model_grid_search_period: int = 100,
        model_grid_search_score: Literal["r2", "gaussian_nll"] | None = None,
        noisy_objective: bool = False,
    ):
        """Initialize the estimator-based selection strategy.

        Args:
            problem (HpProblem): The hyperparameter optimization problem.

            random_state (int | None): Random state for reproducibility. Defaults to ``None``.

            model: Surrogate model name or instance.

            model_kwargs (dict): Additional arguments for model initialization.

            optimizer: Optimization strategy for the solution's acquisition function. Defaults to
                ``"ga"`` for Genetic Algorithm optimization.

            filter_failures: Strategy for handling failed evaluations (i.e., imputation strategy of
                missing values). Defaults to ``"mean"``.

            model_grid_search (bool): Activate or deactivate grid-search for the model. Defaults to
                ``True``.

            model_grid_search_period (int): The solution's model grid search will be triggered every
                ``model_grid_search_period`` new samples. Defaults to ``100``.

            model_grid_search_score (str): The score to use for model selection in grid search.
                Defaults to ``None``.

            noisy_objective (bool): Indicative if the objective observed is noisy or not. Defaults
                to ``False``.
        """
        super().__init__()
        self.problem = problem
        self.rng = check_random_state(random_state)
        self.optimizer = optimizer
        self.filter_failures = filter_failures
        self.model_grid_search = model_grid_search
        self.model_grid_search_period = model_grid_search_period
        if model_grid_search_score is None:
            self.model_grid_search_score = "gaussian_nll" if noisy_objective else "r2"
        else:
            self.model_grid_search_score = model_grid_search_score
        self.noisy_objective = noisy_objective

        self.parameters_list = []
        self.objective_list = []

        # Set default model parameters for the model
        if model == "RF" and model_kwargs is None:
            if self.noisy_objective:
                model_kwargs = {
                    "splitter": "random",
                    "bootstrap": True,
                    "min_samples_leaf": 8,
                    "min_samples_split": 4,
                    "n_estimators": 100,
                    "n_jobs": -1,
                    "max_features": 1.0 if len(self.problem) < 10 else "sqrt",
                }
            else:
                model_kwargs = {
                    "splitter": "best",
                    "bootstrap": False,
                    "min_samples_leaf": 1,
                    "min_samples_split": 2,
                    "n_estimators": 100,
                    "n_jobs": -1,
                    "max_features": 1.0 if len(self.problem) < 10 else "sqrt",
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

    def get_parameter_grid(self) -> dict:
        # Default grid for ExtraTrees
        p_grid = {
            "splitter": ["random", "best"],
            "n_estimators": [100],
            "bootstrap": [False, True],
            "min_samples_leaf": [1],
            "min_samples_split": [2, 4, 8, 16, 32],
            "max_depth": [None, 20],
        }
        return p_grid

    def evaluate(self, X, y):
        try:
            y_mean, y_std, _ = self.model.predict(X, return_std=True, disentangled_std=True)
        except TypeError:
            y_mean, y_std = self.model.predict(X, return_std=True)

        r2_model = 1 - np.mean((y - y_mean) ** 2) / np.var(y)
        r2_ub = 1 - np.mean(y_std**2) / np.var(y)

        # The following evaluates the quality of the AL STD estimates
        # p = ss.pearsonr(y_std**2, (y - y_mean) ** 2)
        return {
            "r2": r2_model,
            "r2_upper_bound": r2_ub,
            "callibration_error": calibration_error(y, y_mean, y_std**2),
            # "y_std_corr": {"statistic": p.statistic, "pvalue": p.pvalue},
        }

    def fit_and_tune_model(self, X, y):
        if self.model_grid_search:
            clf = GridSearchCV(
                estimator=self.model,
                param_grid=self.get_parameter_grid(),
                cv=KFold(n_splits=4, shuffle=True),
                refit=True,
                scoring=SCORING_FUNC_GRID_SEARCH[self.model_grid_search_score],
            )
            clf.fit(X, y)
            self.model = clf.best_estimator_
            print("Solution Selection - grid search tuned model parameters:", clf.best_params_)
        else:
            self.model.fit(X, y)

        scores = self.evaluate(X, y)
        print(f"Solution Selection - grid search tuned model scores: {scores}")

    def _filter_failures(self, yi: Sequence[Any]) -> Tuple[bool, Sequence[float]]:
        """Filter or replace failed objectives.

        Args:
            yi: List of objectives (may contain failure indicators)

        Returns:
            Tuple of (has_success, filtered_objectives)
        """
        has_success = True

        if self.filter_failures in ["mean", "max"]:
            yi_no_failure = [v for v in yi if not isinstance(v, str)]

            if len(yi) != len(yi_no_failure):
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

                yi = [v if not isinstance(v, str) else yi_failed_value for v in yi]

        return has_success, yi

    def _update(self, jobs: Sequence[HPOJob]) -> None:
        """Update solution using surrogate model prediction."""
        if not jobs:
            return

        def params_to_list(x: dict):
            return [x[k] for k in self.problem.hyperparameter_names]

        # Extract new data
        parameters_list = [params_to_list(job.args) for job in jobs]
        objective_list = [job.objective for job in jobs]
        self._update_from_lists(parameters_list, objective_list)

    def _update_from_lists(self, parameters_list, objective_list):
        # Transform parameters to model space
        try:
            transformed_parameters = self.skopt_space.transform(parameters_list)
            self.parameters_list.extend(transformed_parameters)
        except Exception as e:
            logger.error(f"Failed to transform parameters: {e}")
            return
        else:
            self.objective_list.extend(objective_list)

        # Handle failures
        has_success, objective_list = self._filter_failures(self.objective_list)
        if not has_success:
            logger.warning("No successful evaluations available for model fitting")
            return

        X, y = self.parameters_list, objective_list

        # Fit surrogate model
        if len(objective_list) >= self.model_grid_search_period + self.count_tune_model:
            print("Tuning selection model...")
            self.fit_and_tune_model(X, y)
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

    def acq_func(
        self,
        y_mean: ndarray,
        y_std: ndarray | None = None,
        y_std_al: ndarray | None = None,
        y_std_ep: ndarray | None = None,
    ):
        return y_mean
        # MAXIMIZED
        # if y_std is None and y_std_al is None and y_std_ep is None:
        #     return y_mean

        # assert y_std is not None or (y_std_al is not None and y_std_ep is not None)

        # if y_std is not None:
        #     return y_mean - 1.96 * y_std

        # y_stderr = y_std_ep / self.model.n_estimators**0.5
        # return y_mean - 1.96 * y_stderr  # 95% CI

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
