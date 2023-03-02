from deephyper.stopper._stopper import Stopper


class ConstantStopper(Stopper):
    """Constant stopping policy which will stop the evaluation of a configuration at a fixed step.

    Args:
        max_steps (int): the maximum number of steps which should be performed to evaluate the configuration fully.
        stop_step (int): the step at which to stop the evaluation.
    """

    def __init__(self, max_steps: int, stop_step: int) -> None:
        super().__init__(max_steps)
        self.stop_step = stop_step

    def stop(self) -> bool:
        return super().stop() or self.stop_step <= self._count_steps
