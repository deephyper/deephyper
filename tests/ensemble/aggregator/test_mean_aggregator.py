import numpy as np
import pytest

from deephyper.ensemble.aggregator import MeanAggregator


def test_import():
    from deephyper.ensemble.aggregator import Aggregator  # noqa: F401


def test_mean_aggregator_valid_input():
    """Test the MeanAggregator with valid input arrays."""
    y = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
    aggregator = MeanAggregator()
    result = aggregator.aggregate(y)
    expected = np.mean(np.stack(y, axis=0), axis=0)
    assert np.allclose(result, expected), "MeanAggregator failed with valid input."

    aggregator = MeanAggregator(with_scale=True)
    result = aggregator.aggregate(y)
    assert isinstance(result, dict)
    assert "loc" in result and "scale" in result
    expected_loc = np.mean(np.stack(y, axis=0), axis=0)
    expected_scale = np.std(np.stack(y, axis=0), axis=0)
    assert np.allclose(result["loc"], expected_loc), "MeanAggregator failed with valid input."
    assert np.allclose(result["scale"], expected_scale), "MeanAggregator failed with valid input."


def test_mean_aggregator_with_weights():
    """Test the MeanAggregator with weights."""
    y = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
    weights = [0.3, 0.7]
    aggregator = MeanAggregator()
    result = aggregator.aggregate(y, weights=weights)
    expected = np.average(np.stack(y, axis=0), axis=0, weights=weights)
    assert np.allclose(result, expected), "MeanAggregator failed with weights."


def test_mean_aggregator_with_masked_arrays():
    """Test the MeanAggregator with masked arrays."""
    y = [
        np.ma.array([[1, 2], [3, 4]], mask=[[False, True], [False, False]]),
        np.ma.array([[5, 6], [7, 8]], mask=[[False, False], [True, False]]),
    ]
    aggregator = MeanAggregator()
    result = aggregator.aggregate(y)
    expected = np.ma.average(np.ma.stack(y, axis=0), axis=0)
    print(f"{expected=}")
    assert np.ma.allclose(result, expected), "MeanAggregator failed with masked arrays."


def test_mean_aggregator_invalid_weights_length():
    """Test the MeanAggregator with mismatched weights length."""
    y = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
    weights = [0.5]  # Incorrect length
    aggregator = MeanAggregator()
    with pytest.raises(
        ValueError,
        match="The length of `weights` must match the number of predictors in `y`.",
    ):
        aggregator.aggregate(y, weights=weights)


def test_mean_aggregator_invalid_input_type():
    """Test the MeanAggregator with invalid input types."""
    y = [
        "invalid_array",
        np.array([[1, 2], [3, 4]]),
    ]  # One element is not a valid array
    aggregator = MeanAggregator()
    with pytest.raises(
        TypeError,
        match="All elements of `y` must be numpy.ndarray or numpy.ma.MaskedArray.",
    ):
        aggregator.aggregate(y)


def test_mean_aggregator_empty_input():
    """Test the MeanAggregator with an empty input list."""
    y = []
    aggregator = MeanAggregator()
    with pytest.raises(ValueError, match="need at least one array to stack"):
        aggregator.aggregate(y)


def test_mean_aggregator_single_input():
    """Test the MeanAggregator with a single input array."""
    y = [np.array([[1, 2], [3, 4]])]
    aggregator = MeanAggregator()
    result = aggregator.aggregate(y)
    expected = y[0]
    assert np.allclose(result, expected), "MeanAggregator failed with a single input array."


if __name__ == "__main__":
    test_mean_aggregator_valid_input()
    # test_mean_aggregator_with_masked_arrays()
