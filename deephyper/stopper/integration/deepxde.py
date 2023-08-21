from deepxde.callbacks import Callback


class DeepXDEStopperCallback(Callback):
    def __init__(self, job, mode="min", monitor="loss_test"):
        super().__init__()
        """Callback to use in conjonction with a DeepHyper ``RunningJob`` to stop the training when the ``Stopper`` is triggered.

        .. code-block:: python

            def run(job):
                callback = DeepXDEStopperCallback(job, ...)
                ...
                model.train(..., callbacks=[callback])
                ...

        Args:
            job (RunningJob): The running job created by DeepHyper.
            monitor (str, optional): The metric to monitor. It can be any metric collected in the ``History``. Defaults to "loss_test" (equivalent to validation loss).
            mode (str, optional): If the metric is maximized or minimized. Value in ``["max", "min"]``. Defaults to "min".
        """

        self.job = job
        self.monitor = monitor

        assert mode in ["max", "min"]
        self.mode = mode

        self.budget = 0
        self.stopped = False

    def on_epoch_end(self):
        self.budget += 1
        self.observe_and_stop(self.budget)

    def observe_and_stop(self, budget):
        objective = self.get_monitor_value()

        if self.mode == "min":
            objective = -objective

        self.job.record(budget, objective)
        if self.job.stopped():
            self.model.stop_training = True
            self.stopped = True

    def get_monitor_value(self):
        if self.monitor == "loss_train":
            result = sum(self.model.train_state.loss_train)
        elif self.monitor == "loss_test":
            result = sum(self.model.train_state.loss_test)
        else:
            raise ValueError("The specified monitor function is incorrect.")
        return result
