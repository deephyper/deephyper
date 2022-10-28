import os
import traceback

import tensorflow as tf
import numpy as np
import ray

from deephyper.nas.metrics import selectMetric
from deephyper.ensemble import BaseEnsemble
from deephyper.nas.run._util import set_memory_growth_for_visible_gpus


def mse(y_true, y_pred):
    return tf.square(y_true - y_pred)


@ray.remote(num_cpus=1)
def model_predict(model_path, X, batch_size=32, verbose=0):
    """Perform an inference of the model located at ``model_path``.

    :meta private:

    Args:
        model_path (str): Path to the ``h5`` file to load to perform the inferencec.
        X (array): array of input data for which we perform the inference.
        batch_size (int, optional): Batch size used to perform the inferencec. Defaults to 32.
        verbose (int, optional): Verbose option. Defaults to 0.

    Returns:
        array: The prediction based on the provided input data.
    """

    # GPU Configuration if available
    set_memory_growth_for_visible_gpus(True)
    tf.keras.backend.clear_session()
    model_file = model_path.split("/")[-1]

    try:
        if verbose:
            print(f"Loading model {model_file}", flush=True)
        model = tf.keras.models.load_model(model_path, compile=False)
    except Exception:
        if verbose:
            print(f"Could not load model {model_file}", flush=True)
            traceback.print_exc()
        model = None

    if model:
        y = model.predict(X, batch_size=batch_size)
    else:
        y = None

    return y


class BaggingEnsemble(BaseEnsemble):
    """Ensemble based on uniform averaging of the predictions of each members.

    :meta private:

    Args:
        model_dir (str): Path to directory containing saved Keras models in .h5 format.
        loss (callable): a callable taking (y_true, y_pred) as input.
        size (int, optional): Number of unique models used in the ensemble. Defaults to 5.
        verbose (bool, optional): Verbose mode. Defaults to True.
        ray_address (str, optional): Address of the Ray cluster. If "auto" it will try to connect to an existing cluster. If "" it will start a local Ray cluster. Defaults to "".
        num_cpus (int, optional): Number of CPUs allocated to load one model and predict. Defaults to 1.
        num_gpus (int, optional): Number of GPUs allocated to load one model and predict. Defaults to None.
        batch_size (int, optional): Batch size used batchify the inference of loaded models. Defaults to 32.
        selection (str, optional): Selection strategy to build the ensemble. Value in ``["topk"]``. Default to ``topk``.
        mode (str, optional): Value in ``["regression", "classification"]``. Default to ``"regression"``.
    """

    def __init__(
        self,
        model_dir,
        loss=mse,
        size=5,
        verbose=True,
        ray_address="",
        num_cpus=1,
        num_gpus=None,
        batch_size=32,
        selection="topk",
        mode="regression",
    ):
        super().__init__(
            model_dir,
            loss,
            size,
            verbose,
            ray_address,
            num_cpus,
            num_gpus,
            batch_size,
        )
        assert selection in ["topk"]
        self.selection = selection
        assert mode in ["regression", "classification"]
        self.mode = mode

    def __repr__(self) -> str:
        out = super().__repr__()
        out += f"Mode: {self.mode}\n"
        out += f"Selection: {self.selection}\n"
        return out

    def fit(self, X, y):
        """Fit the current algorithm to the provided data.

        Args:
            X (array): The input data.
            y (array): The output data.

        Returns:
            BaseEnsemble: The current fitted instance.
        """
        X_id = ray.put(X)
        model_files = self._list_files_in_model_dir()

        def model_path(f):
            return os.path.join(self.model_dir, f)

        y_pred = ray.get(
            [
                model_predict.options(
                    num_cpus=self.num_cpus, num_gpus=self.num_gpus
                ).remote(model_path(f), X_id, self.batch_size, self.verbose)
                for f in model_files
            ]
        )
        y_pred = np.array([arr for arr in y_pred if arr is not None])

        members_indexes = topk(self.loss, y_true=y, y_pred=y_pred, k=self.size)
        self.members_files = [model_files[i] for i in members_indexes]

        return self

    def predict(self, X) -> np.ndarray:
        """Execute an inference of the ensemble for the provided data.

        Args:
            X (array): An array of input data.

        Returns:
            array: The prediction.
        """
        # make predictions
        X_id = ray.put(X)

        def model_path(f):
            os.path.join(self.model_dir, f)

        y_pred = ray.get(
            [
                model_predict.options(
                    num_cpus=self.num_cpus, num_gpus=self.num_gpus
                ).remote(model_path(f), X_id, self.batch_size, self.verbose)
                for f in self.members_files
            ]
        )
        y_pred = np.array([arr for arr in y_pred if arr is not None])

        y = aggregate_predictions(y_pred, regression=(self.mode == "regression"))

        return y

    def evaluate(self, X, y, metrics=None):
        """Compute metrics based on the provided data.

        Args:
            X (array): An array of input data.
            y (array): An array of true output data.
            metrics (callable, optional): A metric. Defaults to None.
        """
        scores = {}

        y_pred = self.predict(X)

        scores["loss"] = tf.reduce_mean(self.loss(y, y_pred)).numpy()
        if metrics:
            for metric_name in metrics:
                scores[metric_name] = apply_metric(metric_name, y, y_pred)

        return scores


class BaggingEnsembleRegressor(BaggingEnsemble):
    """Ensemble for regression based on uniform averaging of the predictions of each members.


    Args:
        model_dir (str): Path to directory containing saved Keras models in .h5 format.
        loss (callable): a callable taking (y_true, y_pred) as input.
        size (int, optional): Number of unique models used in the ensemble. Defaults to 5.
        verbose (bool, optional): Verbose mode. Defaults to True.
        ray_address (str, optional): Address of the Ray cluster. If "auto" it will try to connect to an existing cluster. If "" it will start a local Ray cluster. Defaults to "".
        num_cpus (int, optional): Number of CPUs allocated to load one model and predict. Defaults to 1.
        num_gpus (int, optional): Number of GPUs allocated to load one model and predict. Defaults to None.
        batch_size (int, optional): Batch size used batchify the inference of loaded models. Defaults to 32.
        selection (str, optional): Selection strategy to build the ensemble. Value in ``["topk"]``. Default to ``topk``.
    """

    def __init__(
        self,
        model_dir,
        loss=mse,
        size=5,
        verbose=True,
        ray_address="",
        num_cpus=1,
        num_gpus=None,
        selection="topk",
    ):
        super().__init__(
            model_dir,
            loss,
            size,
            verbose,
            ray_address,
            num_cpus,
            num_gpus,
            selection,
            mode="regression",
        )


class BaggingEnsembleClassifier(BaggingEnsemble):
    """Ensemble for classification based on uniform averaging of the predictions of each members.


    Args:
        model_dir (str): Path to directory containing saved Keras models in .h5 format.
        loss (callable): a callable taking (y_true, y_pred) as input.
        size (int, optional): Number of unique models used in the ensemble. Defaults to 5.
        verbose (bool, optional): Verbose mode. Defaults to True.
        ray_address (str, optional): Address of the Ray cluster. If "auto" it will try to connect to an existing cluster. If "" it will start a local Ray cluster. Defaults to "".
        num_cpus (int, optional): Number of CPUs allocated to load one model and predict. Defaults to 1.
        num_gpus (int, optional): Number of GPUs allocated to load one model and predict. Defaults to None.
        batch_size (int, optional): Batch size used batchify the inference of loaded models. Defaults to 32.
        selection (str, optional): Selection strategy to build the ensemble. Value in ``["topk"]``. Default to ``topk``.
    """

    def __init__(
        self,
        model_dir,
        loss=mse,
        size=5,
        verbose=True,
        ray_address="",
        num_cpus=1,
        num_gpus=None,
        selection="topk",
    ):
        super().__init__(
            model_dir,
            loss,
            size,
            verbose,
            ray_address,
            num_cpus,
            num_gpus,
            selection,
            mode="classification",
        )


def apply_metric(metric_name, y_true, y_pred) -> float:
    """Perform the computation of provided metric.

    :meta private:

    Args:
        metric_name (str|callable): If ``str`` then it needs to be a metric available in ``deephyper.nas.metrics``.
        y_true (array): Array of true predictions.
        y_pred (array): Array of predicted predictions

    Returns:
        float: a scalar value of the computed metric.
    """
    metric_func = selectMetric(metric_name)
    metric = tf.reduce_mean(
        metric_func(
            tf.convert_to_tensor(y_true, dtype=np.float32),
            tf.convert_to_tensor(y_pred, dtype=np.float32),
        )
    ).numpy()
    return metric


def aggregate_predictions(y_pred, regression=True):
    """Build an ensemble from predictions.

    :meta private:

    Args:
        ensemble_members (np.array): Indexes of selected members in the axis-0 of y_pred.
        y_pred (np.array): Predictions array of shape (n_models, n_samples, n_outputs).
        regression (bool): Boolean (True) if it is a regression (False) if it is a classification.
    Return:
        A TFP Normal Distribution in the case of regression and a np.array with average probabilities
        in the case of classification.
    """
    n = np.shape(y_pred)[0]
    y_pred = np.sum(y_pred, axis=0)
    if regression:
        agg_y_pred = y_pred / n
    else:  # classification
        agg_y_pred = np.argmax(y_pred, axis=1)
    return agg_y_pred


def topk(loss_func, y_true, y_pred, k=2) -> list:
    """Select the Top-k models to be part of the ensemble. A model can appear only once in the ensemble for this strategy.

    :meta private:

    Args:
        loss_func (callable): loss function.
        y_true (array): Array of true predictions.
        y_pred (array): Array of predicted predictions
        k (int, optional): Number of models composing the ensemble. Defaults to 2.

    Returns:
        list: a list of model indexes composing the ensembles.
    """
    # losses is of shape: (n_models, n_outputs)
    losses = tf.reduce_mean(loss_func(y_true, y_pred), axis=1).numpy()
    ensemble_members = np.argsort(losses, axis=0)[:k].reshape(-1).tolist()
    return ensemble_members
