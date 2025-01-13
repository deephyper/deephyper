import numpy as np

from deephyper.ensemble.aggregator import ModeAggregator


def test_mode_aggregator_valid_input():
    """Test the ModeAggregator with valid input arrays."""
    y = [
        np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]),
        np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]),
        np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]),
    ]
    aggregator = ModeAggregator()
    result = aggregator.aggregate(y)
    expected = np.array([0, 1, 1])
    assert np.allclose(result, expected), "ModeAggregator failed with valid input."

    aggregator = ModeAggregator(with_uncertainty=True)
    result = aggregator.aggregate(y)
    assert isinstance(result, dict)
    assert "loc" in result and "uncertainty" in result
    expected_loc = np.array([0, 1, 1])
    expected_uncertainty = np.array([1 / 3, 0, 0])
    assert np.allclose(result["loc"], expected_loc), "ModeAggregator failed with valid input."
    assert np.allclose(result["uncertainty"], expected_uncertainty), (
        "ModeAggregator failed with valid input."
    )


def test_mode_aggregator_valid_input_with_weights():
    """Test the ModeAggregator with valid input arrays."""
    y = [
        np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]),
        np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]),
        np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]),
    ]
    weights = [1, 0, 0]

    aggregator = ModeAggregator()
    result = aggregator.aggregate(y, weights=weights)
    expected = np.array([1, 1, 1])
    assert np.allclose(result, expected), "ModeAggregator failed with valid input."

    aggregator = ModeAggregator(with_uncertainty=True)
    result = aggregator.aggregate(y, weights=weights)
    assert isinstance(result, dict)
    assert "loc" in result and "uncertainty" in result
    expected_loc = np.array([1, 1, 1])
    expected_uncertainty = np.array([0, 0, 0])
    assert np.allclose(result["loc"], expected_loc), "ModeAggregator failed with valid input."
    print(result["uncertainty"])
    assert np.allclose(result["uncertainty"], expected_uncertainty), (
        "ModeAggregator failed with valid input."
    )


def test_mode_aggregator_valid_input_with_masked_array():
    """Test the ModeAggregator with valid input arrays."""
    y = [
        np.ma.array(
            [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            mask=[[True, True], [True, True], [False, False]],
        ),
        np.ma.array(
            [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
            mask=[[False, False], [True, True], [False, False]],
        ),
        np.ma.array(
            [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
            mask=[[False, False], [True, True], [False, False]],
        ),
    ]
    aggregator = ModeAggregator()
    result = aggregator.aggregate(y)
    assert isinstance(result, np.ma.MaskedArray)
    expected = np.ma.array([0, 0, 1], mask=[False, True, False])
    assert np.ma.allclose(result, expected), "ModeAggregator failed with valid input."

    aggregator = ModeAggregator(with_uncertainty=True)
    result = aggregator.aggregate(y)
    assert isinstance(result, dict)
    assert "loc" in result and "uncertainty" in result
    assert isinstance(result["loc"], np.ma.MaskedArray)
    assert isinstance(result["uncertainty"], np.ma.MaskedArray)
    expected_loc = np.ma.array([0, 0, 1], mask=[False, True, False])
    expected_uncertainty = np.ma.array([0, 0, 0], mask=[False, True, False])
    assert np.allclose(result["loc"], expected_loc), "ModeAggregator failed with valid input."
    assert np.allclose(result["uncertainty"], expected_uncertainty), (
        "ModeAggregator failed with valid input."
    )


if __name__ == "__main__":
    # test_mode_aggregator_valid_input()
    # test_mode_aggregator_valid_input_with_weights()
    test_mode_aggregator_valid_input_with_masked_array()
