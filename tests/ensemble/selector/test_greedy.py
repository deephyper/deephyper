import numpy as np


def test_greedy_selector_np_array():
    """Test greedy Selector with regular Numpy Arrays."""
    from deephyper.ensemble.loss import SquaredError
    from deephyper.ensemble.selector import GreedySelector
    from deephyper.ensemble.aggregator import MeanAggregator

    y_true = np.array([[0, 0, 0, 0]])
    y_predictors = [
        np.array([[1, 0, 0, 0]]),
        np.array([[1, 0.5, 0, 0]]),
        np.array([[-1, 0, 0, 0]]),
        np.array([[0, -0.5, 0, 0]]),
        np.array([[0, -1, 1, 0]]),
        np.array([[1, 1, 1, 1]]),
    ]

    selector = GreedySelector(loss_func=SquaredError(), aggregator=MeanAggregator(), k_init=1)
    indices, weights = selector.select(y_true, y_predictors)

    assert len(indices) == 1
    assert len(weights) == 1
    assert indices == [3]
    assert weights == [1.0]

    selector = GreedySelector(loss_func=SquaredError(), aggregator=MeanAggregator(), k_init=2)
    indices, weights = selector.select(y_true, y_predictors)
    assert len(indices) == 3
    assert len(weights) == 3
    assert indices == [0, 2, 3]
    assert abs(sum(weights) - 1) <= 1e-3

    selector = GreedySelector(
        loss_func=SquaredError(),
        aggregator=MeanAggregator(),
        k_init=1,
        early_stopping=False,
        k=5,
        with_replacement=False,
    )
    indices, weights = selector.select(y_true, y_predictors)
    assert len(indices) == 5
    assert len(weights) == 5
    assert indices == [0, 1, 2, 3, 4]
    assert abs(sum(weights) - 1) <= 1e-3


if __name__ == "__main__":
    test_greedy_selector_np_array()
