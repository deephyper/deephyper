from numbers import Number

import numpy as np

from deephyper.stopper._stopper import Stopper


class SuccessiveHalvingStopper(Stopper):
    """Stopper based on the Asynchronous Successive Halving algorithm (ASHA).

    .. list-table::
        :widths: 25 25 25
        :header-rows: 1

        * - Single-Objective
          - Multi-Objectives
          - Failures
        * - ✅
          - ❌
          - ✅

    The Successive Halving (SHA) was proposed in `Non-stochastic Best Arm
    IdentiÞcation and Hyperparameter Optimization
    <http://proceedings.mlr.press/v51/jamieson16.pdf>`_ in the context of a
    fixed number of hyperparameter configurations. The SHA algorithm was
    synchronous at the time and therefore not efficient when using parallel
    ressources. The Sucessive Halving algorithm was then extended to be
    asynchronous in `A System for Massively Parallel Hyperparameter Tuning
    <https://arxiv.org/abs/1810.05934>`_.

    Halving is a technique to reduce the number of configurations to evaluate
    by a factor of ``reduction_factor``. The halving schedule is following a
    geometric progression. The first halving step is done after ``min_steps``
    steps. The next halving step is done after ``min_steps *
    reduction_factor`` steps. The next halving step is done after
    ``min_steps * reduction_factor^2`` steps. And so on.

    Args:
        max_steps (int):
            The maximum number of steps to run the evaluation (e.g., number of epochs).
        min_steps (float, optional):
            The minimum number of steps to run the evaluation. Defaults to 1.
        reduction_factor (float, optional):
            At each halving step the current model is kept only if among the
            top-``1/reduction_factor*100``%. Defaults to 3.
        min_early_stopping_rate (float, optional):
            A parameter to delay the halving schedule. Defaults to 0.
        min_competing (int, optional):
            The minimum number of competitors necessary to check the top-k condition. Defaults to 0.
        min_fully_completed (int, optional):
            The minimum number of evaluation evaluated with ``max_steps``. Defaults to 1.
    """

    def __init__(
        self,
        max_steps: int,
        min_steps: float = 1,
        reduction_factor: float = 3,
        min_early_stopping_rate: float = 0,
        min_competing: int = 0,
        min_fully_completed=0,
        epsilon=1e-10,
    ) -> None:
        super().__init__(max_steps=max_steps)
        self.min_steps = min_steps
        self._reduction_factor = reduction_factor
        self._min_early_stopping_rate = min_early_stopping_rate
        self._min_competing = min_competing
        self._min_fully_completed = min_fully_completed
        self.epsilon = epsilon

        self._rung = 0
        self._list_completed_rung = []

    def _compute_halting_budget(self):
        return (self.min_steps - 1) + self._reduction_factor ** (
            self._min_early_stopping_rate + self._rung
        )

    def _get_competiting_objectives(self) -> list:
        search_id, _ = self.job.id.split(".")
        values = self.job.storage.load_metadata_from_all_jobs(
            search_id, f"_completed_rung_{self._rung}"
        )
        # Filter out non numerical values (e.g., "F" for failed jobs)
        values = [v for v in values if isinstance(v, Number)]
        return values

    def _num_fully_completed(self) -> int:
        search_id, _ = self.job.id.split(".")
        stopped = self.job.storage.load_metadata_from_all_jobs(search_id, "_completed")
        num = sum(int(s) for s in stopped)
        return num

    def observe(self, budget: float, objective: float):
        super().observe(budget, objective)
        self._budget = budget
        self._objective = objective

        # Compute when is the next Halting Budget
        halting_budget = self._compute_halting_budget()

        if self._budget >= halting_budget:
            # casting float to str to avoid numerical rounding of database
            # e.g. for Redis: The precision of the output is fixed at 17 digits
            # after the decimal point regardless of the actual internal precision
            # of the computation.
            self.job.storage.store_job_metadata(
                self.job.id, f"_completed_rung_{self._rung}", self._objective
            )
            self._list_completed_rung.append(self._rung)

        # A failure was observed consider all previous rungs as failed
        if not (isinstance(self._objective, Number)):
            for rung in self._list_completed_rung:
                self.job.storage.store_job_metadata(
                    self.job.id, f"_completed_rung_{rung}", self._objective
                )

    def stop(self) -> bool:
        # Enforce Pre-conditions Before Applying Successive Halving
        if super().stop():
            return True

        # Compute when is the next Halting Budget
        halting_budget = self._compute_halting_budget()

        if self._budget < halting_budget:
            return False

        # Check if the minimum number of fully completed budgets has been done
        if (
            self._min_fully_completed > 0
            and self._num_fully_completed() < self._min_fully_completed
        ):
            self._rung += 1
            return False

        # Check if the minimum number of competitors is verified
        competing_objectives = np.sort(self._get_competiting_objectives())
        num_competing = len(competing_objectives)

        if num_competing < self._min_competing:
            return True

        # Performe Successive Halving
        k = int(num_competing // self._reduction_factor)

        # Promote if best when there is less than reduction_factor competing values
        if k == 0:
            k = 1

        top_k_worst_objective = competing_objectives[-k]

        # The epsilon can vary depending on the numerical precision of the storage
        promotable = (self._objective + self.epsilon) >= top_k_worst_objective
        if promotable:
            self._rung += 1
        return not (promotable)
