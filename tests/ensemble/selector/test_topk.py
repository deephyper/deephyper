import numpy as np


def test_topk_selector_np_array():
    """Test Topk Selector with regular Numpy Arrays."""
    from deephyper.ensemble.loss import SquaredError
    from deephyper.ensemble.selector import TopKSelector

    y_true = np.array([[0, 0, 0, 0]])
    y_predictors = [
        np.array([[0, 0, 0, 0]]),
        np.array([[1, 0, 0, 0]]),
        np.array([[1, 1, 0, 0]]),
        np.array([[1, 1, 1, 0]]),
        np.array([[1, 1, 1, 1]]),
    ]

    selector = TopKSelector(loss_func=SquaredError(), k=3)
    indices, weights = selector.select(y_true, y_predictors)

    assert len(indices) == 3
    assert len(weights) == 3
    assert indices == [0, 1, 2]
    assert weights == [1.0, 1.0, 1.0]


def test_topk_selector_np_ma_array():
    """Test Topk Selector with Masked Numpy Arrays."""
    from deephyper.ensemble.loss import SquaredError
    from deephyper.ensemble.selector import TopKSelector

    y_true = np.array([[0, 0, 0, 0]])
    y_predictors = [
        np.ma.array([[0, 0, 0, 0]], mask=[[False, False, False, True]]),
        np.ma.array([[1, 0, 0, 0]], mask=[[False, False, True, False]]),
        np.ma.array([[1, 1, 0, 0]], mask=[[True, True, False, False]]),
        np.ma.array([[1, 1, 1, 0]], mask=[[False, False, False, True]]),
        np.ma.array([[1, 1, 1, 1]], mask=[[False, False, False, True]]),
    ]

    selector = TopKSelector(loss_func=SquaredError(), k=3)
    indices, weights = selector.select(y_true, y_predictors)

    assert len(indices) == 3
    assert len(weights) == 3
    assert indices == [0, 2, 1]
    assert weights == [1.0, 1.0, 1.0]
