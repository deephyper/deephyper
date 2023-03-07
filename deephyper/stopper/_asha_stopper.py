import numpy as np

from deephyper.stopper._stopper import Stopper


class SuccessiveHalvingStopper(Stopper):
    """Stopper based on the asynchronous successive halving algorithm."""

    def __init__(
        self,
        max_steps: int,
        min_steps: float = 1,
        reduction_factor: float = 3,
        min_early_stopping_rate: float = 0,
        min_competing: int = 0,
        min_fully_completed=0,
    ) -> None:
        super().__init__(max_steps=max_steps)
        self.min_steps = min_steps
        self._reduction_factor = reduction_factor
        self._min_early_stopping_rate = min_early_stopping_rate
        self._min_competing = min_competing
        self._min_fully_completed = min_fully_completed

        self._rung = 0

    def _compute_halting_budget(self):
        return (self.min_steps - 1) + self._reduction_factor ** (
            self._min_early_stopping_rate + self._rung
        )

    def _get_competiting_objectives(self) -> list:
        search_id, _ = self.job.id.split(".")
        values = self.job.storage.load_metadata_from_all_jobs(
            search_id, f"_completed_rung_{self._rung}"
        )
        values = [float(v) for v in values]
        return values

    def _num_fully_completed(self) -> int:
        search_id, _ = self.job.id.split(".")
        stopped = self.job.storage.load_metadata_from_all_jobs(search_id, "stopped")
        num = sum(int(not (s)) for s in stopped)
        return num

    def observe(self, budget: float, objective: float):
        super().observe(budget, objective)
        self._budget = budget
        self._objective = objective

        halting_budget = self._compute_halting_budget()

        if self._budget >= halting_budget:
            # casting float to str to avoid numerical rounding of database
            # e.g. for Redis: The precision of the output is fixed at 17 digits
            # after the decimal point regardless of the actual internal precision
            # of the computation.
            self.job.storage.store_job_metadata(
                self.job.id, f"_completed_rung_{self._rung}", str(self._objective)
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
        if self._num_fully_completed() < self._min_fully_completed:
            self._rung += 1
            return False

        # Check if the minimum number of competitors is verified
        competing_objectives = np.sort(self._get_competiting_objectives())
        num_competing = len(competing_objectives)

        if num_competing < self._min_competing:
            return True

        # Performe Successive Halving
        k = num_competing // self._reduction_factor

        # Promote if best when there is less than reduction_factor competing values
        if k == 0:
            k = 1

        top_k_worst_objective = competing_objectives[-k]

        promotable = self._objective >= top_k_worst_objective
        if promotable:
            self._rung += 1
        return not (promotable)
