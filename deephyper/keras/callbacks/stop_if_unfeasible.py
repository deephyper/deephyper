import time

import tensorflow as tf


class StopIfUnfeasible(tf.keras.callbacks.Callback):
    def __init__(self, time_limit=600, patience=20):
        super().__init__()
        self.time_limit = time_limit
        self.timing = list()
        self.stopped = False  # boolean set to True if the model training has been stopped due to time_limit condition
        self.patience = patience

    def set_params(self, params):
        self.params = params
        if self.params["steps"] is None:
            self.steps = self.params["samples"] // self.params["batch_size"]
            self.steps = self.params["samples"] // self.params["batch_size"]
        if self.steps * self.params["batch_size"] < self.params["samples"]:
            self.steps += 1
        else:
            self.steps = self.params["steps"]

    def on_batch_begin(self, batch, logs=None):
        """Called at the beginning of a training batch in `fit` methods.
        Subclasses should override for any actions to run.

        Args:
            batch (int): index of batch within the current epoch.
            logs (dict): has keys `batch` and `size` representing the current
                batch number and the size of the batch.
        """
        self.timing.append(time.time())

    def on_batch_end(self, batch, logs=None):
        """Called at the end of a training batch in `fit` methods.
        Subclasses should override for any actions to run.

        Args:
            batch (int): index of batch within the current epoch.
            logs (dict): metric results for this batch.
        """
        self.timing[-1] = time.time() - self.timing[-1]
        self.avr_batch_time = sum(self.timing) / len(self.timing)
        self.estimate_training_time = sum(self.timing) + self.avr_batch_time * (
            self.steps - len(self.timing)
        )

        if (
            len(self.timing) >= self.patience
            and self.estimate_training_time > self.time_limit
        ):
            self.stopped = True
            self.model.stop_training = True
