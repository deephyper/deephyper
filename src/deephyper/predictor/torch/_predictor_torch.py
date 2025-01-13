from typing import List

import torch

from deephyper.predictor import Predictor, PredictorFileLoader


class TorchPredictor(Predictor):
    """Represents a frozen torch model that can only predict."""

    def __init__(self, module: torch.nn.Module):
        if not isinstance(module, torch.nn.Module):
            raise ValueError(
                f"The given module is of type {type(module)} when it should be of type "
                f"torch.nn.Module!"
            )

        self.module = module

    def pre_process_inputs(self, X):
        X = torch.from_numpy(X)
        return X

    def post_process_predictions(self, y):
        if isinstance(y, torch.Tensor):
            y = y.detach().numpy()
        elif isinstance(y, dict):
            y = {k: v.detach().numpy() for k, v in y.items()}
        elif isinstance(y, list):
            y = [yi.detach().numpy() for yi in y]
        return y

    def predict(self, X):
        X = self.pre_process_inputs(X)
        training = self.module.training
        if training:
            self.module.eval()

        if hasattr(self.module, "predict_proba"):
            y = self.module.predict_proba(X)
        else:
            y = self.module(X)

        self.module.train(training)
        y = self.post_process_predictions(y)
        return y


class TorchPredictorFileLoader(PredictorFileLoader):
    """Loads a predictor from a file for the TensorFlow Keras 2 backend.

    Args:
        path_predictor_file (str): the path to the predictor file.
    """

    def __init__(self, path_predictor_file: str):
        super().__init__(path_predictor_file)

    def load(self) -> TorchPredictor:
        model = torch.load(self.path_predictor_file, weights_only=False)
        return TorchPredictor(model)

    @staticmethod
    def find_predictor_files(path_directory: str, file_extension: str = "pt") -> List[str]:
        """Finds the predictor files in a directory given a specific extension.

        Args:
            path_directory (str): the directory path.
            file_extension (str, optional): the file extension. Defaults to ``"pt"``.

        Returns:
            List[str]: the list of predictor files found in the directory.
        """
        return PredictorFileLoader.find_predictor_files(path_directory, file_extension)
