import abc
import logging
from typing import Any, Sequence, Optional, Literal

import numpy as np
from pydantic import BaseModel
from sklearn.base import is_regressor
from sklearn.utils import check_random_state

from deephyper.evaluator import HPOJob
from deephyper.hpo._problem import HpProblem, convert_to_skopt_space
from deephyper.skopt.optimizer.acq_optimizer.pymoo_ga import GAPymooAcqOptimizer
from deephyper.skopt.utils import cook_estimator

logger = logging.getLogger(__name__)


class Solution(BaseModel):
    """Represents the estimated solution of a search."""

    parameters: Any
    objective: Any


class SolutionSelection(abc.ABC):
    """Base class for search solution selection strategies."""

    def __init__(self):
        self.solution = Solution(parameters=None, objective=None)
        # Value set by the SearchHistory
        self.num_objective: int = None

    def update(self, jobs: Sequence[HPOJob]) -> None:
        # TODO: currently skipped and only applied for single objective optimization
        if self.num_objective > 1:
            return
        self._update(jobs)
        logger.info(f"Updated search solution: {self.solution}")

    @abc.abstractmethod
    def _update(self, jobs: Sequence[HPOJob]) -> None: ...


class ArgMaxObsSelection(SolutionSelection):
    """Selects the best solution based on maximum observed objective(s)."""

    def _update(self, jobs: Sequence[HPOJob]) -> None:
        for job in jobs:
            if type(job.objective) is str:
                self.solution = Solution(parameters=job.args, objective=None)
                continue

            if self.solution.objective is None or job.objective >= self.solution.objective:
                self.solution = Solution(parameters=job.args, objective=job.objective)


class ArgMaxEstSelection(SolutionSelection):
    """Selects solution using a surrogate model and acquisition optimizer."""

    def __init__(
        self,
        problem: HpProblem,
        random_state: Optional[int] = None,
        model: str = "RF",
        model_kwargs: Optional[dict] = None,
        optimizer: Literal["sampling", "ga"] = "sampling",
    ):
        super().__init__()
        self.problem = problem
        self.rng = check_random_state(random_state)
        self.optimizer = optimizer
        self.filter_failures = "mean"

        self.parameters_list = []
        self.objective_list = []

        if model == "RF" and model_kwargs is None:
            model_kwargs = dict(
                splitter="random",
                bootstrap=True,
                min_samples_leaf=3,
                n_estimators=100,
                n_jobs=1,
            )
        else:
            model_kwargs = {}

        self.skopt_space = convert_to_skopt_space(problem.space, surrogate_model=model)

        if isinstance(model, str):
            model = cook_estimator(
                model,
                space=self.skopt_space,
                random_state=self.rng.randint(0, np.iinfo(np.int32).max),
                **model_kwargs,
            )

        if not is_regressor(model):
            raise ValueError(f"Model {model.__class__.__name__} must be a regressor.")

        self.model = model

    def _filter_failures(self, yi):
        """Filter or replace failed objectives.

        Args:
            yi (list): a list of objectives.

        Returns:
            list: the filtered list.
        """
        has_success = True
        if self.filter_failures in ["mean", "max"]:
            yi_no_failure = [v for v in yi if v != "F"]

            # when yi_no_failure is empty all configurations are failures
            if len(yi_no_failure) == 0:
                yi_failed_value = 0
                has_success = False
            elif self.filter_failures == "mean":
                yi_failed_value = np.mean(yi_no_failure).tolist()
            else:
                yi_failed_value = np.max(yi_no_failure).tolist()

            yi = [v if v != "F" else yi_failed_value for v in yi]

        return has_success, yi

    def _update(self, jobs: Sequence[HPOJob]) -> None:
        if not jobs:
            return

        new_parameters = [list(job.args.values()) for job in jobs]
        self.objective_list.extend(job.objective for job in jobs)

        transformed_parameters = self.skopt_space.transform(new_parameters)
        self.parameters_list.extend(transformed_parameters)

        has_success, objective_list = self._filter_failures(self.objective_list)
        if not has_success:
            return

        self.model.fit(self.parameters_list, objective_list)

        optimize_fn = {
            "sampling": self.optimize_sampling,
            "ga": self.optimize_ga,
        }.get(self.optimizer)

        if not optimize_fn:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")

        parameters, objective = optimize_fn()
        self.solution = Solution(
            parameters={k: v for k, v in zip(self.skopt_space.dimension_names, parameters)},
            objective=objective,
        )

    def optimize_sampling(self, n_samples: int = 10_000) -> tuple[dict, float]:
        samples = self.skopt_space.rvs(n_samples=n_samples, random_state=self.rng, n_jobs=1)
        transformed = self.skopt_space.transform(samples)
        y_pred = self.model.predict(transformed)

        idx = np.argmax(y_pred)
        best_parameters = self.skopt_space.inverse_transform([transformed[idx]])[0]
        return best_parameters, y_pred[idx]

    def optimize_ga(
        self,
        n_samples: int = 10_000,
        pop_size: int = 100,
        xtol: float = 1e-8,
        ftol: float = 1e-6,
        period: int = 30,
        n_max_gen: int = 1000,
    ) -> tuple[dict, float]:
        samples = self.skopt_space.rvs(n_samples=n_samples, random_state=self.rng, n_jobs=1)
        transformed = self.skopt_space.transform(samples)
        y_pred = self.model.predict(transformed)

        top_idx = np.argsort(y_pred)[-pop_size:]
        acq_opt = GAPymooAcqOptimizer(
            space=self.skopt_space,
            x_init=transformed[top_idx],
            y_init=y_pred[top_idx],
            pop_size=pop_size,
            random_state=self.rng.randint(0, np.iinfo(np.int32).max),
            termination_kwargs=dict(xtol=xtol, ftol=ftol, period=period, n_max_gen=n_max_gen),
        )

        x_sol = acq_opt.minimize(acq_func=lambda x: -self.model.predict(x))
        best_parameters = self.skopt_space.inverse_transform([x_sol])[0]
        return best_parameters, self.model.predict([x_sol])[0]
