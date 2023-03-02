import abc
import copy


class Stopper(abc.ABC):
    """An abstract class describing the interface of a Stopper.

    Args:
        max_steps (int): the maximum number of calls to ``observe(budget, objective)``.
    """

    def __init__(self, max_steps: int) -> None:
        assert max_steps > 0
        self.max_steps = max_steps
        self._count_steps = 0
        self.job = None

        # Initialize list to collect observations
        self.observed_budgets = []
        self.observed_objectives = []

    def to_json(self):
        """Returns a dict version of the stopper which can be saved as JSON."""
        json_format = type(self).__name__
        return json_format

    def transform_objective(self, objective: float):
        """Replaces the currently observed objective by the maximum objective observed from the
        start. Identity transformation by default."""
        # prev_objective = (
        #     self.observed_objectives[-1] if len(self.observed_objectives) > 0 else None
        # )
        # if prev_objective is not None:
        #     objective = max(prev_objective, objective)
        return objective

    @property
    def step(self):
        return self.observed_budgets[-1]

    def observe(self, budget: float, objective: float):
        self._count_steps += 1

        objective = self.transform_objective(objective)

        self.observed_budgets.append(budget)
        self.observed_objectives.append(objective)

    def stop(self) -> bool:
        return self._count_steps >= self.max_steps

    @property
    def observations(self) -> list:
        obs = [self.observed_budgets, self.observed_objectives]
        return copy.deepcopy(obs)

    @property
    def objective(self):
        return self.observations[-1][-1]
