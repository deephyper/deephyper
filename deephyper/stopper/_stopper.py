import abc


class Stopper(abc.ABC):
    def __init__(self) -> None:
        self.job = None
        self.observed_budgets = []
        self.observed_objectives = []

    def observe(self, budget: float, objective: float):
        self.observed_budgets.append(budget)
        self.observed_objectives.append(objective)

    def stop(self) -> bool:
        return False

    @property
    def observations(self) -> list:
        return [self.observed_budgets, self.observed_objectives]
