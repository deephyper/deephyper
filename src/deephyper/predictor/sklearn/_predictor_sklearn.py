import pickle

from typing import List

from sklearn.base import BaseEstimator, ClassifierMixin
from deephyper.predictor import Predictor, PredictorFileLoader


class SklearnPredictor(Predictor):
    """Represents a frozen Scikit-Learn model that can only predict."""

    def __init__(self, model: BaseEstimator):
        self.model = model
        self._predict_func = "predict"
        if isinstance(self.model, ClassifierMixin) and hasattr(self.model, "predict_proba"):
            self._predict_func = "predict_proba"

    def predict(self, X):
        if self._predict_func == "predict":
            y = self.model.predict(X)
        else:
            y = self.model.predict_proba(X)
        return y


class SklearnPredictorFileLoader(PredictorFileLoader):
    """Loads a predictor from a file for the Scikit-Learn backend.

    Args:
        path_predictor_file (str): the path to the predictor file.
    """

    def __init__(self, path_predictor_file: str):
        super().__init__(path_predictor_file)

    def load(self) -> SklearnPredictor:
        with open(self.path_predictor_file, "rb") as f:
            model = pickle.load(f)
        return SklearnPredictor(model)

    @staticmethod
    def find_predictor_files(path_directory: str, file_extension: str = "pkl") -> List[str]:
        """Finds the predictor files in a directory given a specific extension.

        Args:
            path_directory (str): the directory path.
            file_extension (str, optional): the file extension. Defaults to ``"pkl"``.

        Returns:
            List[str]: the list of predictor files found in the directory.
        """
        return PredictorFileLoader.find_predictor_files(path_directory, file_extension)
