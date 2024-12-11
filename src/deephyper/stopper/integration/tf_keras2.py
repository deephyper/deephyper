"""Submodule of Tensorflow/Keras integrations for the ``Stopper``."""

import warnings

from tf_keras.callbacks import Callback

from deephyper.evaluator import RunningJob


class TFKerasStopperCallback(Callback):
    """Tensorflow/Keras callback to be used with a DeepHyper ``RunningJob``.

    This stops the training when the ``Stopper`` is triggered.

    .. code-block:: python

        def run(job):
            callback = TFKerasStopperCallback(job, ...)
            ...
            model.fit(..., callbacks=[callback])
            ...

    Args:
        job (RunningJob):
            The running job created by DeepHyper.
        monitor (str, optional):
            The metric to monitor. It can be any metric collected in the
            ``History``. Defaults to ``"val_loss"``.
        mode (str, optional):
            If the metric is maximized or minimized. Value in ``
            ["max", "min"]``. Defaults to ``"max"``.
    """

    def __init__(self, job: RunningJob, monitor: str = "val_loss", mode: str = "min"):
        super().__init__()
        self.job = job
        self.monitor = monitor

        assert mode in ["max", "min"]
        self.mode = mode

        self.budget = 0

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of an epoch during training.

        Args:
            epoch: Integer, index of epoch.
            logs: Dict, metric results for this training epoch, and for the
              validation epoch if validation is performed. Validation result
              keys are prefixed with `val_`. For training epoch, the values of
              the `Model`'s metrics are returned. Example:
              `{'loss': 0.2, 'accuracy': 0.7}`.
        """
        self.budget += 1
        self._observe_and_stop(self.budget, logs)

    def _observe_and_stop(self, budget, logs):
        if logs is None:
            return

        objective = logs.get(self.monitor)

        if objective is None:
            warnings.warn(
                f"Monitor {self.monitor} is not found in the history logs. Stopper will not be "
                "able to stop the training. Available logs are: {list(logs.keys())}"
            )
            return

        if self.mode == "min":
            objective = -objective

        self.job.record(budget, objective)
        if self.job.stopped():
            self.model.stop_training = True
