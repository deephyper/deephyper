import os
import pathlib
import pickle as pkl

import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor


def wrap_and_predict(model):
    from deephyper.predictor.sklearn import SklearnPredictor

    x = np.zeros((16, 1), dtype=np.float32)
    y_target = np.zeros((16, 1), dtype=np.int32)
    y_target[8:, :] = 1
    model.fit(x, y_target)

    predictor = SklearnPredictor(model)
    y = predictor.predict(x)

    return y


def test_sklearn_predictor_with_single_output_regressor():
    y = wrap_and_predict(DummyRegressor())
    assert isinstance(y, np.ndarray)
    assert np.shape(y) == (16,)


def test_sklearn_predictor_with_single_output_classifier():
    y = wrap_and_predict(DummyClassifier())
    assert isinstance(y, np.ndarray)
    assert np.shape(y) == (16, 2)


def test_sklearn_predictor_from_pkl_file(tmp_path):
    from deephyper.predictor.sklearn import SklearnPredictorFileLoader

    x = np.zeros((16, 1), dtype=np.float32)
    y_target = np.zeros((16, 1), dtype=np.int32)
    y_target[8:, :] = 1

    model = DummyClassifier()
    model.fit(x, y_target)

    pathlib.Path(tmp_path).mkdir(parents=True, exist_ok=True)

    model_ckpt_path = os.path.join(tmp_path, "dummy_classifier.pkl")
    with open(model_ckpt_path, "wb") as fp:
        pkl.dump(model, fp)

    found_ckpt_files = SklearnPredictorFileLoader.find_predictor_files(tmp_path)
    assert len(found_ckpt_files)
    assert found_ckpt_files[0] == model_ckpt_path

    predictor_loader = SklearnPredictorFileLoader(model_ckpt_path)
    predictor = predictor_loader.load()
    y = predictor.predict(x)

    assert isinstance(y, np.ndarray)
    assert np.shape(y) == (16, 2)


if __name__ == "__main__":
    test_sklearn_predictor_with_single_output_regressor()
    test_sklearn_predictor_with_single_output_classifier()

    tmp_path = "/tmp/deephyper_test"
    test_sklearn_predictor_from_pkl_file(tmp_path)
