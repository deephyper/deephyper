from numbers import Number

import numpy as np

from deephyper.stopper._stopper import Stopper


class MedianStopper(Stopper):
    """Stopper based on the median of observed objectives at similar budgets.

    .. list-table::
        :widths: 25 25 25
        :header-rows: 1

        * - Single-Objective
          - Multi-Objectives
          - Failures
        * - ✅
          - ❌
          - ❌
    """

    def __init__(
        self,
        max_steps: int,
        min_steps: int = 1,
        min_competing: int = 0,
        min_fully_completed: int = 0,
        interval_steps: int = 1,
        epsilon: float = 1e-10,
    ) -> None:
        super().__init__(max_steps=max_steps)

        self.min_steps = min_steps
        self._min_competing = min_competing
        self._min_fully_completed = min_fully_completed
        self._interval_steps = interval_steps
        self.epsilon = epsilon

        self._rung = 0

    def _is_halting_budget(self):
        if self.step < self.min_steps:
            return False
        else:
            return (self.step - self.min_steps) % self._interval_steps == 0

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
        self._budget = self.observed_budgets[-1]
        self._objective = self.observed_objectives[-1]

        if self._is_halting_budget():
            # casting float to str to avoid numerical rounding of database
            # e.g. for Redis: The precision of the output is fixed at 17 digits
            # after the decimal point regardless of the actual internal precision
            # of the computation.
            self.job.storage.store_job_metadata(
                self.job.id, f"_completed_rung_{self._rung}", self._objective
            )

    def stop(self) -> bool:
        # Enforce Pre-conditions
        if super().stop():
            return True

        if not (self._is_halting_budget()):
            return False

        if (
            self._min_fully_completed > 0
            and self._num_fully_completed() < self._min_fully_completed
        ):
            return False

        # Apply Median Pruning
        competing_objectives = np.sort(self._get_competiting_objectives())
        num_competing = len(competing_objectives)

        if num_competing < self._min_competing:
            return False

        median_objective = np.median(competing_objectives)

        promotable = self._objective + self.epsilon >= median_objective
        if promotable:
            self._rung += 1
        return not (promotable)
