import copy

import numpy as np

from deephyper.stopper._stopper import Stopper


class MedianStopper(Stopper):
    def __init__(
        self,
        min_budget: int = 1,
        min_competing: int = 0,
        min_fully_completed: int = 0,
        interval_steps: int = 1,
    ) -> None:
        super().__init__()
        self._min_budget = min_budget
        self._min_competing = min_competing
        self._min_fully_completed = min_fully_completed
        self._interval_steps = interval_steps

        self._rung = 0
        self._budget = min_budget
        self._objective = None

    def _is_halting_budget(self):
        if self._budget < self._min_budget:
            return False
        else:
            return (self._budget - self._min_budget) % self._interval_steps == 0

    def _get_competiting_objectives(self) -> list:
        search_id, _ = self.job.id.split(".")
        values = self.job.storage.load_metadata_from_all_jobs(
            search_id, f"completed_rung_{self._rung}"
        )
        values = [float(v) for v in values]
        return values

    def _num_fully_completed(self) -> int:
        search_id, _ = self.job.id.split(".")
        stopped = self.job.storage.load_metadata_from_all_jobs(search_id, "stopped")
        num = sum(int(not (s)) for s in stopped)
        return num

    def observe(self, budget: float, objective: float):
        self._budget = budget
        self._objective = objective

        if self._is_halting_budget():
            super().observe(budget, objective)

            # casting float to str to avoid numerical rounding of database
            # e.g. for Redis: The precision of the output is fixed at 17 digits
            # after the decimal point regardless of the actual internal precision
            # of the computation.
            self.job.storage.store_job_metadata(
                self.job.id, f"completed_rung_{self._rung}", str(self._objective)
            )

    def stop(self) -> bool:

        if not (self._is_halting_budget()):
            return False

        # check if configuration needs to be prune
        else:

            if self._num_fully_completed() < self._min_fully_completed:
                return False

            competing_objectives = np.sort(self._get_competiting_objectives())
            num_competing = len(competing_objectives)

            if num_competing < self._min_competing:
                return False

            median_objective = np.median(competing_objectives)

            promotable = self._objective >= median_objective
            if promotable:
                self._rung += 1
            return not (promotable)

    @property
    def observations(self) -> list:
        obs = copy.deepcopy(super().observations)
        if self._budget > obs[0][-1]:
            obs[0].append(self._budget)
            obs[1].append(self._objective)
        return obs
