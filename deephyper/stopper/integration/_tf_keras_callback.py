import warnings

from tensorflow.keras.callbacks import Callback


class TFKerasStopperCallback(Callback):
    def __init__(self, job, monitor="val_loss", mode="min") -> None:
        """_summary_

        Args:
            job (RunningJob): The running job created by DeepHyper.
            monitor (str, optional): The metric to monitor. It can be any metric collected in the ``History``. Defaults to "val_loss".
            mode (str, optional): If the metric is maximized or minimized. Value in ``["max", "min"]``. Defaults to "max".
        """
        super().__init__()
        self.job = job
        self.monitor = monitor

        assert mode in ["max", "min"]
        self.mode = mode

        self.budget = 0

    def on_epoch_end(self, epoch, logs=None):

        self.budget += 1
        self.observe_and_stop(self.budget, logs)

    def observe_and_stop(self, budget, logs):

        if logs is None:
            return

        objective = logs.get(self.monitor)

        if objective is None:
            warnings.warn(
                f"Monitor {self.monitor} is not found in the history logs. Stopper will not be able to stop the training. Available logs are: {list(logs.keys())}"
            )
            return

        if self.mode == "min":
            objective = -objective

        self.job.record(budget, objective)
        if self.job.stopped():
            self.model.stop_training = True
