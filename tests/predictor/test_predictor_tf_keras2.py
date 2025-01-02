import numpy as np
import pytest


def wrap_and_predict(model):
    from deephyper.predictor.tf_keras2 import TFKeras2Predictor

    predictor = TFKeras2Predictor(model)
    x = np.zeros((16, 1), dtype=np.float32)
    y = predictor.predict(x)
    return y


@pytest.mark.tf_keras2
def test_tf_keras2_predictor_with_single_output():
    import tf_keras as tfk

    model = tfk.Sequential(
        [
            tfk.layers.Dense(10, activation="relu"),
            tfk.layers.Dense(1),
        ],
    )

    y = wrap_and_predict(model)
    assert isinstance(y, np.ndarray)
    assert np.shape(y) == (16, 1)


@pytest.mark.tf_keras2
def test_tf_keras2_predictor_with_list_output():
    import tf_keras as tfk

    input_tensor = tfk.layers.Input(shape=(1,))
    hidden_layer = tfk.layers.Dense(10, activation="relu")
    output_layer = tfk.layers.Dense(1)

    out_0 = output_layer(hidden_layer(input_tensor))
    out_1 = output_layer(hidden_layer(input_tensor))

    model = tfk.Model(inputs=input_tensor, outputs=[out_0, out_1])

    y = wrap_and_predict(model)

    assert type(y) is list
    assert len(y) == 2
    assert all(isinstance(y[i], np.ndarray) for i in range(2))
    assert all(np.shape(y[i]) == (16, 1) for i in range(2))


@pytest.mark.tf_keras2
def test_tf_keras2_predictor_with_dict_output():
    import tf_keras as tfk

    input_tensor = tfk.layers.Input(shape=(1,))
    hidden_layer = tfk.layers.Dense(10, activation="relu")
    output_layer = tfk.layers.Dense(1)

    out_0 = output_layer(hidden_layer(input_tensor))
    out_1 = output_layer(hidden_layer(input_tensor))

    model = tfk.Model(inputs=input_tensor, outputs={"out_0": out_0, "out_1": out_1})

    y = wrap_and_predict(model)

    assert type(y) is dict
    assert len(y) == 2
    assert all(isinstance(v, np.ndarray) for v in y.values())
    assert all(np.shape(v) == (16, 1) for v in y.values())


if __name__ == "__main__":
    test_tf_keras2_predictor_with_single_output()
    test_tf_keras2_predictor_with_list_output()
    test_tf_keras2_predictor_with_dict_output()
