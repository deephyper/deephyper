import abc
import os

from typing import List

import numpy as np


class Predictor(abc.ABC):
    """Base class that represents a model ``f(X) = y`` that can predict."""

    @abc.abstractmethod
    def predict(self, X: np.ndarray):
        """Predicts the target for the inputs.

        Args:
            X (np.ndarray): the inputs.

        Returns:
            np.ndarray: the predicted target.
        """


class PredictorLoader(abc.ABC):
    """Represents a loader for a predictor."""

    @abc.abstractmethod
    def load(self) -> Predictor:
        """Loads a predictor.

        Returns:
            Predictor: the loaded predictor.
        """


class PredictorFileLoader(PredictorLoader):
    """Represents a file loader for a predictor.

    Args:
        path_predictor_file (str): the path to the predictor file.
    """

    def __init__(self, path_predictor_file: str):
        self.path_predictor_file = path_predictor_file

    def __repr__(self) -> str:
        return f"{type(self).__name__}('{self.path_predictor_file}')"

    @staticmethod
    def find_predictor_files(path_directory: str, file_extension: str) -> List[str]:
        """Finds the predictor files in a directory given a specific extension.

        Args:
            path_directory (str): the directory path.
            file_extension (str): the file extension.

        Returns:
            List[str]: the list of predictor files found in the directory.
        """
        files = [
            os.path.join(path_directory, f)
            for f in os.listdir(path_directory)
            if f.endswith(file_extension)
        ]
        return files
