import abc
import json
import os
import traceback

import ray
import tensorflow as tf


class BaseEnsemble(abc.ABC):
    """Base class for ensembles, every new ensemble algorithms needs to extend this class.

    Args:
        model_dir (str): Path to directory containing saved Keras models in .h5 format.
        loss (callable): a callable taking (y_true, y_pred) as input.
        size (int, optional): Number of unique models used in the ensemble. Defaults to 5.
        verbose (bool, optional): Verbose mode. Defaults to True.
        ray_address (str, optional): Address of the Ray cluster. If "auto" it will try to connect to an existing cluster. If "" it will start a local Ray cluster. Defaults to "".
        num_cpus (int, optional): Number of CPUs allocated to load one model and predict. Defaults to 1.
        num_gpus (int, optional): Number of GPUs allocated to load one model and predict. Defaults to None.
        batch_size (int, optional): Batch size used batchify the inference of loaded models. Defaults to 32.
    """

    def __init__(
        self,
        model_dir,
        loss,
        size=5,
        verbose=True,
        ray_address="",
        num_cpus=1,
        num_gpus=None,
        batch_size=32,
    ):
        self.model_dir = os.path.abspath(model_dir)
        self.loss = loss
        self.members_files = []
        self.size = size
        self.verbose = verbose
        self.ray_address = ray_address
        self.num_cpus = num_cpus
        self.num_gpus = num_gpus
        self.batch_size = batch_size

        if not (ray.is_initialized()):
            ray.init(address=self.ray_address)

    def __repr__(self) -> str:
        out = ""
        out += f"Model Dir: {self.model_dir}\n"
        out += f"Members files: {self.members_files}\n"
        out += f"Ensemble size: {len(self.members_files)}/{self.size}\n"
        return out

    def _list_files_in_model_dir(self):
        return [f for f in os.listdir(self.model_dir) if f[-2:] == "h5"]

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
