import abc
import json
import os

import numpy as np

from deephyper.evaluator import Evaluator, RunningJob
from deephyper.evaluator.storage import NullStorage


def _wrapper_model_predict_func(job: RunningJob, model_predict_func: callable):
    return model_predict_func(**job.parameters)


class Ensemble(abc.ABC):
    """Base class for ensembles, every new ensemble algorithms needs to extend this class.

    Args:
        model_dir (str): Path to directory containing saved Keras models in .h5 format.
        loss (callable): a callable taking (y_true, y_pred) as input.
        size (int, optional): Number of unique models used in the ensemble. Defaults to 5.
        verbose (bool, optional): Verbose mode. Defaults to True.
        batch_size (int, optional): Batch size used to batchify the inference of loaded models. Defaults to ``32``.
        evaluator_method (str, optional): Method used to run the (parallel) inferences of the members of the ensemble. Defaults to ``"serial"`` for sequential in-process execution.
        evaluator_method_kwargs (dict, optional): Keyword arguments passed to the evaluator method. Defaults to ``None``.
    """

    def __init__(
        self,
        model_dir,
        loss,
        size=5,
        verbose=True,
        batch_size=32,
        evaluator_method="serial",
        evaluator_method_kwargs=None,
    ):
        self.model_dir = os.path.abspath(model_dir)
        self.loss = loss
        self.members_files = []
        self.size = size
        self.verbose = verbose
        self.batch_size = batch_size
        self._model_predict_func = None
        self._evaluator = None
        self.evaluator_method = evaluator_method
        self.evaluator_method_kwargs = (
            {} if evaluator_method_kwargs is None else evaluator_method_kwargs
        )

    def init_evaluator(self):
        """Initialize an evaluator for the ensemble.

        Returns:
            Evaluator: An evaluator instance.
        """
        method_kwargs = {
            "storage": NullStorage(),
            "run_function_kwargs": {"model_predict_func": self._model_predict_func},
        }
        method_kwargs.update(self.evaluator_method_kwargs)
        self._evaluator = Evaluator.create(
            run_function=_wrapper_model_predict_func,
            method=self.evaluator_method,
            method_kwargs=method_kwargs,
        )

    def predict_with_models(self, X, model_files: list):
        """Given a sequence of models, this method will execute the inference of each model on the provided data.

        Args:
            X (np.ndarray): The input data.
            model_files (list): the sequence of models.

        Returns:
            np.ndarray: the inference of shape (n_models, n_samples, n_outputs)
        """

        self._evaluator.submit(
            [
                {
                    "model_path": path,
                    "X": X,
                    "batch_size": self.batch_size,
                    "verbose": self.verbose,
                }
                for path in model_files
            ]
        )

        jobs_done = self._evaluator.gather("ALL")
        jobs_done = list(sorted(jobs_done, key=lambda j: int(j.id.split(".")[-1])))

        y_pred = [job.result for job in jobs_done]
        y_pred = np.array([arr for arr in y_pred if arr is not None])
        return y_pred

    def _list_files_in_model_dir(self):
        return [
            f
            for f in os.listdir(self.model_dir)
            if f.endswith("h5") or f.endswith("keras")
        ]

    @abc.abstractmethod
    def fit(self, X, y):
        """Fit the current algorithm to the provided data.

        Args:
            X (array): The input data.
            y (array): The output data.

        Returns:
            BaseEnsemble: The current fitted instance.
        """

    @abc.abstractmethod
    def predict(self, X):
        """Execute an inference of the ensemble for the provided data.

        Args:
            X (array): An array of input data.

        Returns:
            array: The prediction.
        """

    @abc.abstractmethod
    def evaluate(self, X, y, metrics=None):
        """Compute metrics based on the provided data.

        Args:
            X (array): An array of input data.
            y (array): An array of true output data.
            metrics (callable, optional): A metric. Defaults to None.
        """

    def load_members_files(self, file: str = "ensemble.json") -> None:
        """Load the members composing an ensemble.

        Args:
            file (str, optional): Path of JSON file containing the ensemble members. All members needs to be accessible in ``model_dir``. Defaults to "ensemble.json".
        """
        with open(file, "r") as f:
            self.members_files = json.load(f)

    def save_members_files(self, file: str = "ensemble.json") -> None:
        """Save the list of file names of the members of the ensemble in a JSON file.

        Args:
            file (str, optional): Path JSON file where the file names are saved. Defaults to "ensemble.json".
        """
        with open(file, "w") as f:
            json.dump(self.members_files, f)

    def load(self, file: str) -> None:
        """Load an ensemble from a save.

        Args:
            file (str): Path to the save of the ensemble.
        """
        self.load_members_files(file)

    def save(self, file: str = None) -> None:
        """Save an ensemble.

        Args:
            file (str): Path to the save of the ensemble.
        """
        self.save_members_files(file)
