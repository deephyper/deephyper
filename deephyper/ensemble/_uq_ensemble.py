import os

import numpy as np

from deephyper.ensemble._ensemble import Ensemble


class UQEnsemble(Ensemble):
    """Ensemble with uncertainty quantification based on uniform averaging of the predictions of each members.

    :meta private:

    Args:
        model_dir (str): Path to directory containing saved Keras models in .h5 format.

        loss (callable): a callable taking (y_true, y_pred) as input.

        size (int, optional): Number of unique models used in the ensemble. Defaults to 5.
        verbose (bool, optional): Verbose mode. Defaults to True.

        batch_size (int, optional): Batch size used batchify the inference of loaded models. Defaults to 32.

        evaluator_method (str, optional): Method used to run the (parallel) inferences of the members of the ensemble. Defaults to ``"serial"`` for sequential in-process execution.

        evaluator_method_kwargs (dict, optional): Keyword arguments passed to the evaluator method. Defaults to ``None``.
    """

    @property
    def model_dir(self):
        return self._ensemble.model_dir

    @model_dir.setter
    def model_dir(self, value):
        self._ensemble.model_dir = value

    @property
    def loss(self):
        return self._ensemble.loss

    @loss.setter
    def loss(self, value):
        self._ensemble.loss = value

    @property
    def members_files(self):
        return self._ensemble.members_files

    @members_files.setter
    def members_files(self, value):
        self._ensemble.members_files = value

    @property
    def size(self):
        return self._ensemble.size

    @size.setter
    def size(self, value):
        self._ensemble.size = value

    @property
    def verbose(self):
        return self._ensemble.verbose

    @verbose.setter
    def verbose(self, value):
        self._ensemble.verbose = value

    @property
    def evaluator_method(self):
        return self._ensemble.evaluator_method

    @evaluator_method.setter
    def ray_address(self, value):
        self._ensemble.evaluator_method = value

    @property
    def evaluator_method_kwargs(self):
        return self._ensemble.evaluator_method_kwargs

    @property
    def batch_size(self):
        return self._ensemble.batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._ensemble.batch_size = value

    @property
    def selection(self):
        return self._ensemble.selection

    @selection.setter
    def selection(self, value):
        self._ensemble.selection = value

    @property
    def mode(self):
        return self._ensemble.mode

    @mode.setter
    def mode(self, value):
        self._ensemble.mode = value

    def fit(self, X, y):
        self._ensemble.fit(X, y)

    def predict(self, X) -> np.ndarray:
        return self._ensemble.predict(X)

    def evaluate(self, X, y, metrics=None, scaler_y=None):
        return self._ensemble.evaluate(X, y, metrics, scaler_y)

    def load_members_files(self, file: str = "ensemble.json") -> None:
        """Load the members composing an ensemble.

        Args:
            file (str, optional): Path of JSON file containing the ensemble members. All members needs to be accessible in ``model_dir``. Defaults to "ensemble.json".
        """
        self._ensemble.load_members_files(file)

    def save_members_files(self, file: str = "ensemble.json") -> None:
        """Save the list of file names of the members of the ensemble in a JSON file.

        Args:
            file (str, optional): Path JSON file where the file names are saved. Defaults to "ensemble.json".
        """
        self._ensemble.save_members_files(file)


class UQEnsembleRegressor(UQEnsemble):
    """Ensemble with uncertainty quantification for regression based on uniform averaging of the predictions of each members.

    Args:
        model_dir (str): Path to directory containing saved models.

        loss (callable): a callable taking (y_true, y_pred) as input. Defaults to ``"nll"``.

        size (int, optional): Number of unique models used in the ensemble. Defaults to ``5``.

        verbose (bool, optional): Verbose mode. Defaults to ``True``.

        batch_size (int, optional): Batch size used batchify the inference of loaded models. Defaults to 32.

        selection (str, optional): Selection strategy to build the ensemble. Value in ``["topk", "caruana"]``. Default to ``topk``.


        load_model_func (callable, optional): Function to load checkpointed models. It takes as input the path to the model file and return the loaded model. Defaults to ``None`` for default model loading strategy.

        evaluator_method (str, optional): Method used to run the (parallel) inferences of the members of the ensemble. Defaults to ``"serial"`` for sequential in-process execution.

        evaluator_method_kwargs (dict, optional): Keyword arguments passed to the evaluator method. Defaults to ``None``.
    """

    def __init__(
        self,
        model_dir,
        loss="nll",
        size=5,
        verbose=True,
        batch_size=32,
        selection="topk",
        load_model_func=None,
        evaluator_method="serial",
        evaluator_method_kwargs=None,
    ):

        nn_backend = os.environ.get("DEEPHYPER_NN_BACKEND", "torch")
        if nn_backend == "tf_keras2":
            from deephyper.ensemble._uq_ensemble_tf_keras2 import (
                _TFKerasUQEnsembleRegressor as _UQEnsemble,
            )

        elif nn_backend == "torch":
            from deephyper.ensemble._uq_ensemble_torch import (
                _TorchUQEnsembleRegressor as _UQEnsemble,
            )

        else:
            raise ValueError(f"Unsupported DEEPHYPER_NN_BACKEND='{nn_backend}'")

        self._ensemble = _UQEnsemble(
            model_dir=model_dir,
            loss=loss,
            size=size,
            verbose=verbose,
            batch_size=batch_size,
            selection=selection,
            load_model_func=load_model_func,
            evaluator_method=evaluator_method,
            evaluator_method_kwargs=evaluator_method_kwargs,
        )

    def predict_var_decomposition(self, X):
        r"""Execute an inference of the ensemble for the provided data with uncertainty quantification estimates. The **aleatoric uncertainty** corresponds to the expected value of learned variance of each model composing the ensemble :math:`\mathbf{E}[\sigma_\\theta^2(\mathbf{x})]`. The **epistemic uncertainty** corresponds to the variance of learned mean estimates of each model composing the ensemble :math:`\mathbf{V}[\mu_\\theta(\mathbf{x})]`.

        Args:
            X (array): An array of input data.

        Returns:
            y, u1, u2: where ``y`` is the mixture distribution, ``u1`` is the aleatoric component of the variance of ``y`` and ``u2`` is the epistemic component of the variance of ``y``.
        """
        return self._ensemble.predict_var_decomposition(X)


class UQEnsembleClassifier(UQEnsemble):
    """Ensemble with uncertainty quantification for classification based on uniform averaging of the predictions of each members.

    Args:
        model_dir (str): Path to directory containing saved models.

        loss (callable): a callable taking (y_true, y_pred) as input. Defaults to ``"cce"``.

        size (int, optional): Number of unique models used in the ensemble. Defaults to ``5``.

        verbose (bool, optional): Verbose mode. Defaults to ``True``.

        batch_size (int, optional): Batch size used batchify the inference of loaded models. Defaults to 32.

        selection (str, optional): Selection strategy to build the ensemble. Value in ``["topk", "caruana"]``. Default to ``topk``.

        load_model_func (callable, optional): Function to load checkpointed models. It takes as input the path to the model file and return the loaded model. Defaults to ``None`` for default model loading strategy.

        evaluator_method (str, optional): Method used to run the (parallel) inferences of the members of the ensemble. Defaults to ``"serial"`` for sequential in-process execution.

        evaluator_method_kwargs (dict, optional): Keyword arguments passed to the evaluator method. Defaults to ``None``.
    """

    def __init__(
        self,
        model_dir,
        loss="cce",
        size=5,
        verbose=True,
        batch_size=32,
        selection="topk",
        load_model_func=None,
        evaluator_method="serial",
        evaluator_method_kwargs=None,
    ):

        nn_backend = os.environ.get("DEEPHYPER_NN_BACKEND", "torch")
        if nn_backend == "tf_keras2":
            from deephyper.ensemble._uq_ensemble_tf_keras2 import (
                _TFKerasUQEnsembleClassifier as _UQEnsemble,
            )

        elif nn_backend == "torch":
            from deephyper.ensemble._uq_ensemble_torch import (
                _TorchUQEnsembleClassifier as _UQEnsemble,
            )

        else:
            raise ValueError(f"Unsupported DEEPHYPER_NN_BACKEND='{nn_backend}'")

        self._ensemble = _UQEnsemble(
            model_dir=model_dir,
            loss=loss,
            size=size,
            verbose=verbose,
            batch_size=batch_size,
            selection=selection,
            load_model_func=load_model_func,
            evaluator_method=evaluator_method,
            evaluator_method_kwargs=evaluator_method_kwargs,
        )
