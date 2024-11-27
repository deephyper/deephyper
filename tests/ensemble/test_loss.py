import numpy as np
import pytest


@pytest.mark.fast
def test_import():
    from deephyper.ensemble.loss import Loss  # noqa: F401


@pytest.mark.fast
def test_squared_loss():
    from deephyper.ensemble.loss import SquaredError

    y_true = np.full((2, 1), fill_value=2.0)
    y_pred = np.full((2, 1), fill_value=0.0)

    loss = SquaredError()

    value = loss(y_true, y_pred)
    assert np.shape(value) == (2, 1)
    assert value[0][0] == 4

    value = loss(y_true, {"loc": y_pred})
    assert np.shape(value) == (2, 1)
    assert value[0][0] == 4

    with pytest.raises(ValueError):
        value = loss(y_true, None)

    with pytest.raises(ValueError):
        value = loss(y_true, {"foo": y_pred})


@pytest.mark.fast
def test_absolute_loss():
    from deephyper.ensemble.loss import AbsoluteError

    y_true = np.full((2, 1), fill_value=2.0)
    y_pred = np.full((2, 1), fill_value=0.0)

    loss = AbsoluteError()

    value = loss(y_true, y_pred)
    assert np.shape(value) == (2, 1)
    assert value[0][0] == 2

    value = loss(y_true, {"loc": y_pred})
    assert np.shape(value) == (2, 1)
    assert value[0][0] == 2

    with pytest.raises(ValueError):
        value = loss(y_true, None)

    with pytest.raises(ValueError):
        value = loss(y_true, {"foo": y_pred})


@pytest.mark.fast
def test_normal_nll_loss():
    from deephyper.ensemble.loss import NormalNegLogLikelihood

    y_true = np.full((2, 1), fill_value=2.0)
    y_pred = np.full((2, 2, 1), fill_value=1.0)

    loss = NormalNegLogLikelihood()

    value = loss(y_true, y_pred)
    assert np.shape(value) == (2, 1)
    assert abs(value[0][0] - 1.4189) <= 1e-4

    value = loss(y_true, {"loc": y_pred[0], "scale": y_pred[1]})
    assert np.shape(value) == (2, 1)
    assert abs(value[0][0] - 1.4189) <= 1e-4

    with pytest.raises(ValueError):
        value = loss(y_true, None)

    with pytest.raises(ValueError):
        value = loss(y_true, {"foo": y_pred})


@pytest.mark.fast
def test_zero_one_loss():
    from deephyper.ensemble.loss import ZeroOneLoss

    y_true = np.full((2, 1), fill_value=2.0)
    y_pred = np.full((2, 1), fill_value=1.0)

    loss = ZeroOneLoss()

    value = loss(y_true, y_pred)
    assert np.shape(value) == (2, 1)
    assert value[0][0] == 1

    value = loss(y_true, {"loc": y_pred})
    assert np.shape(value) == (2, 1)
    assert value[0][0] == 1

    with pytest.raises(ValueError):
        value = loss(y_true, None)

    with pytest.raises(ValueError):
        value = loss(y_true, {"foo": y_pred})

    y_true = np.full((2, 1), fill_value=1.0)
    y_pred = np.full((2, 2), fill_value=0.0)
    y_pred[:, 0] = 1.0

    loss = ZeroOneLoss(predict_proba=True)

    value = loss(y_true, y_pred)
    assert np.shape(value) == (2, 1)
    assert value[0][0] == 1

    y_pred[:, 0] = 0.0
    y_pred[:, 1] = 1.0

    value = loss(y_true, y_pred)
    assert np.shape(value) == (2, 1)
    assert value[0][0] == 0

    y_true = np.full((2,), fill_value=1.0)
    value = loss(y_true, y_pred)
    assert np.shape(value) == (2,)
    assert value[0] == 0


@pytest.mark.fast
def test_categorical_crossentropy():
    from deephyper.ensemble.loss import CategoricalCrossEntropy

    y_true = np.full((2,), fill_value=1)
    y_pred = np.full((2, 2), fill_value=0.0)
    y_pred[0, 0] = 1.0
    y_pred[1, 1] = 1.0

    loss = CategoricalCrossEntropy()

    value = loss(y_true, y_pred)
    assert np.shape(value) == (2,)
    assert abs(value[0] - 36.04) <= 1e-2
    assert value[1] <= 1e-10

    value = loss(y_true, {"loc": y_pred})
    assert np.shape(value) == (2,)
    assert abs(value[0] - 36.04) <= 1e-2
    assert value[1] <= 1e-10

    with pytest.raises(ValueError):
        value = loss(y_true, None)

    with pytest.raises(ValueError):
        value = loss(y_true, {"foo": y_pred})


if __name__ == "__main__":
    # test_squared_loss()
    # test_absolute_loss()
    # test_normal_nll_loss()
    # test_zero_one_loss()
    test_categorical_crossentropy()
