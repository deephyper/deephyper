import argparse
import os
import traceback

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import ray

from deephyper.nas.metrics import selectMetric
from deephyper.core.parser import add_arguments_from_signature
from deephyper.ensemble import BaseEnsemble
from deephyper.nas.run.util import set_memory_growth_for_visible_gpus


def nll(y, rv_y):
    """Negative log likelihood for Tensorflow probability."""
    return -rv_y.log_prob(y)


cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)


@ray.remote(num_cpus=1)
def model_predict(model_path, X):
    import tensorflow as tf
    import tensorflow_probability as tfp

    # GPU Configuration if available
    set_memory_growth_for_visible_gpus(True)
    tf.keras.backend.clear_session()
    model_file = model_path.split("/")[-1]

    try:
        print(f"Loading model {model_file}", end="\n", flush=True)
        model = tf.keras.models.load_model(model_path, compile=False)
    except:
        print(f"Could not load model {model_file}", flush=True)
        traceback.print_exc()
        model = None

    if model:
        y_dist = model(X)
        if isinstance(y_dist, tfp.distributions.Distribution):
            y = np.concatenate([y_dist.loc, y_dist.scale], axis=1)
        else:
            y = y_dist
    else:
        y = None

    return y


class UQBaggingEnsemble(BaseEnsemble):
    def __init__(
        self,
        model_dir,
        loss=nll,
        size=5,
        verbose=True,
        ray_address="",
        num_cpus=1,
        num_gpus=None,
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
        )
        self.selection = selection
        assert mode in ["regression", "classification"]
        self.mode = mode

    @staticmethod
    def _extend_parser(parser) -> argparse.ArgumentParser:
        add_arguments_from_signature(parser, UQBaggingEnsemble)
        return parser

    def fit(self, X, y):
        X_id = ray.put(X)

        model_files = self.list_all_model_files()
        model_path = lambda f: os.path.join(self.model_dir, f)

        y_pred = ray.get(
            [
                model_predict.options(
                    num_cpus=self.num_cpus, num_gpus=self.num_gpus
                ).remote(model_path(f), X_id)
                for f in model_files
            ]
        )
        y_pred = np.array([arr for arr in y_pred if arr is not None])

        members_indexes = topk(self.loss, y_true=y, y_pred=y_pred, k=self.size)
        self.members_files = [model_files[i] for i in members_indexes]

    def predict(self, X) -> np.ndarray:
        # make predictions
        X_id = ray.put(X)
        model_path = lambda f: os.path.join(self.model_dir, f)

        y_pred = ray.get(
            [
                model_predict.options(
                    num_cpus=self.num_cpus, num_gpus=self.num_gpus
                ).remote(model_path(f), X_id)
                for f in self.members_files
            ]
        )
        y_pred = np.array([arr for arr in y_pred if arr is not None])

        y = aggregate_predictions(y_pred, regression=(self.mode == "regression"))

        return y

    def evaluate(self, X, y, metrics=None):
        scores = {}

        y_pred = self.predict(X)

        scores["loss"] = tf.reduce_mean(self.loss(y, y_pred)).numpy()
        if metrics:
            for metric_name in metrics:
                scores[metric_name] = apply_metric(metric_name, y, y_pred)

        return scores


class UQBaggingEnsembleRegressor(UQBaggingEnsemble):
    def __init__(
        self,
        model_dir,
        loss=nll,
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


class UQBaggingEnsembleClassifier(UQBaggingEnsemble):
    def __init__(
        self,
        model_dir,
        loss=cce,
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
    metric_func = selectMetric(metric_name)
    metric = tf.reduce_mean(
        metric_func(
            tf.convert_to_tensor(y_true, dtype=np.float64),
            tf.convert_to_tensor(y_pred, dtype=np.float64),
        )
    ).numpy()
    return metric


def aggregate_predictions(y_pred, regression=True):
    """Build an ensemble from predictions.

    Args:
        ensemble_members (np.array): Indexes of selected members in the axis-0 of y_pred.
        y_pred (np.array): Predictions array of shape (n_models, n_samples, n_outputs).
        regression (bool): Boolean (True) if it is a regression (False) if it is a classification.
    Return:
        A TFP Normal Distribution in the case of regression and a np.array with average probabilities
        in the case of classification.
    """
    if regression:
        # assuming first half are means, second half are std
        mid = np.shape(y_pred)[2] // 2
        loc = y_pred[:, :, :mid]
        scale = y_pred[:, :, mid:]

        mean_loc = np.mean(loc, axis=0)
        sum_loc_scale = np.square(loc) + np.square(scale)
        mean_scale = np.sqrt(np.mean(sum_loc_scale, axis=0) - np.square(mean_loc))

        return tfp.distributions.Normal(loc=mean_loc, scale=mean_scale)
    else:  # classification
        agg_y_pred = np.mean(y_pred[:, :, :], axis=0)
        return agg_y_pred


def topk(loss_func, y_true, y_pred, k=2):
    """Select the top-k models to be part of the ensemble. A model can appear only once in the ensemble for this strategy."""
    if np.shape(y_true)[-1] * 2 == np.shape(y_pred)[-1]: # regression
        mid = np.shape(y_true)[-1]
        y_pred = tfp.distributions.Normal(
            loc=y_pred[:, :, :mid], scale=y_pred[:, :, mid:]
        )
    # losses is of shape: (n_models, n_outputs)
    losses = tf.reduce_mean(loss_func(y_true, y_pred), axis=1).numpy()
    ensemble_members = np.argsort(losses, axis=0)[:k].reshape(-1).tolist()
    return ensemble_members