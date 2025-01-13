from typing import List

import tensorflow as tf

# TODO: Check if this import could be removed. Currently, the following import
# is necessary to avoid a bug if checkpointed models that are loaded have tfp
# layers.
# import tensorflow_probability as tfp
import tf_keras as tfk

from deephyper.predictor import Predictor, PredictorFileLoader


class TFKeras2Predictor(Predictor):
    """Represents a frozen TensorFlow/Keras2 model that can only predict."""

    def __init__(self, model: tfk.Model):
        self.model = model

    def pre_process_inputs(self, X):
        return X

    def post_process_predictions(self, y):
        if isinstance(y, tf.Tensor):
            y = y.numpy()
        elif isinstance(y, dict):
            y = {k: v.numpy() for k, v in y.items()}
        elif isinstance(y, list):
            y = [yi.numpy() for yi in y]
        return y

    def predict(self, X):
        X = self.pre_process_inputs(X)
        y = self.model(X, training=False)
        y = self.post_process_predictions(y)
        return y


class TFKeras2PredictorFileLoader(PredictorFileLoader):
    """Loads a predictor from a file for the TensorFlow/Keras2 backend.

    Args:
        path_predictor_file (str): the path to the predictor file.
    """

    def __init__(self, path_predictor_file: str):
        super().__init__(path_predictor_file)

    def load(self) -> TFKeras2Predictor:
        model = tfk.models.load_model(self.path_predictor_file, compile=False, safe_mode=False)
        return TFKeras2Predictor(model)

    @staticmethod
    def find_predictor_files(path_directory: str, file_extension: str = "keras") -> List[str]:
        """Finds the predictor files in a directory given a specific extension.

        Args:
            path_directory (str): the directory path.
            file_extension (str, optional): the file extension. Defaults to ``"keras"``.

        Returns:
            List[str]: the list of predictor files found in the directory.
        """
        return PredictorFileLoader.find_predictor_files(path_directory, file_extension)
