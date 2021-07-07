import traceback
import os
import json

import tensorflow as tf
import ray



class BaseEnsemble:
    """Base class of ensemble.

        Args:
            model_dir (str): Path to directory containing saved Keras models in .h5 format.
            loss (callable): a callable taking (y_true, y_pred) as input.
            size (int, optional): Size of the ensemble. Defaults to 5.
            verbose (bool, optional): Verbose mode. Defaults to True.
            ray_address (str, optional): Address of the Ray cluster. If "auto" it will try to connect to an existing cluster. If "" it will start a local Ray cluster. Defaults to "".
            num_cpus (int, optional): Number of CPUs allocated to load one model and predict. Defaults to 1.
            num_gpus (int, optional): Number of GPUs allocated to load one model and predict. Defaults to None.
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

        if not(ray.is_initialized()):
            ray.init(address=self.ray_address)

    def __repr__(self) -> str:
        out = ""
        out += f"Model Dir: {self.model_dir}\n"
        out += f"Members files: {self.members_files}\n"
        out += f"Ensemble size: {len(self.members_files)}/{self.size}\n"
        return out

    def _list_files_in_model_dir(self):
        return [f for f in os.listdir(self.model_dir) if f[-2:] == "h5"]

    def fit(self, X, y) -> None:
        raise NotImplementedError

    def load_members_files(self, file: str = "ensemble.json") -> None:
        with open(file, "r") as f:
            self.members_files = json.load(f)

    def save_members_files(self, file: str = "ensemble.json") -> None:
        with open(file, "w") as f:
            json.dump(self.members_files, f)

    def load(self, file: str) -> None:
        self.load_members_files(file)

    def save(self, file: str=None) -> None:
        self.save_members_files(file)
