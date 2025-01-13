import numpy as np
import pytest

from deephyper.ensemble.aggregator import MixedNormalAggregator


def test_mixed_normal_aggregator_valid_input():
    """Test the MixedNormalAggregator with valid input arrays."""
    y = [
        {
            "loc": np.array([[1, 2], [3, 4]]),
            "scale": np.array([[0.1, 0.2], [0.3, 0.4]]),
        },
        {
            "loc": np.array([[5, 6], [7, 8]]),
            "scale": np.array([[0.5, 0.6], [0.7, 0.8]]),
        },
    ]

    aggregator = MixedNormalAggregator()
    result = aggregator.aggregate(y)
    assert "loc" in result and "scale" in result
    expected = np.mean(np.stack([y[0]["loc"], y[1]["loc"]], axis=0), axis=0)
    assert np.allclose(result["loc"], expected), "MixedNormalAggregator failed with valid input."
    assert np.shape(result["loc"]) == np.shape(result["scale"]), (
        "MixedNormalAggregator failed with valid input."
    )

    aggregator = MixedNormalAggregator(decomposed_scale=True)
    result = aggregator.aggregate(y)
    assert "loc" in result and "scale_aleatoric" in result and "scale_epistemic" in result
    assert np.allclose(result["loc"], expected), "MixedNormalAggregator failed with valid input."
    assert np.shape(result["loc"]) == np.shape(result["scale_aleatoric"]), (
        "MixedNormalAggregator failed with valid input."
    )
    assert np.shape(result["loc"]) == np.shape(result["scale_epistemic"]), (
        "MixedNormalAggregator failed with valid input."
    )


def test_mean_aggregator_with_weights():
    """Test the MixedNormalAggregator with weights."""
    y = [
        {
            "loc": np.array([[1, 2], [3, 4]]),
            "scale": np.array([[0.1, 0.2], [0.3, 0.4]]),
        },
        {
            "loc": np.array([[5, 6], [7, 8]]),
            "scale": np.array([[0.5, 0.6], [0.7, 0.8]]),
        },
    ]
    weights = [0.3, 0.7]

    aggregator = MixedNormalAggregator()

    with pytest.raises(
        ValueError,
        match="The length of `weights` must match the number of predictors in `y`.",
    ):
        aggregator.aggregate(y, weights=weights[1:])

    result = aggregator.aggregate(y, weights=weights)
    assert "loc" in result and "scale" in result
    expected = np.sum(np.stack([0.3 * y[0]["loc"], 0.7 * y[1]["loc"]], axis=0), axis=0)
    assert np.allclose(result["loc"], expected), "MixedNormalAggregator failed with valid input."
    assert np.shape(result["loc"]) == np.shape(result["scale"]), (
        "MixedNormalAggregator failed with valid input."
    )

    aggregator = MixedNormalAggregator(decomposed_scale=True)
    result = aggregator.aggregate(y, weights=weights)
    assert "loc" in result and "scale_aleatoric" in result and "scale_epistemic" in result
    assert np.allclose(result["loc"], expected), "MixedNormalAggregator failed with valid input."
    assert np.shape(result["loc"]) == np.shape(result["scale_aleatoric"]), (
        "MixedNormalAggregator failed with valid input."
    )
    assert np.shape(result["loc"]) == np.shape(result["scale_epistemic"]), (
        "MixedNormalAggregator failed with valid input."
    )


def test_mean_aggregator_with_masked_arrays():
    """Test the MixedNormalAggregator with masked arrays."""
    y = [
        {
            "loc": np.ma.array([[1, 2], [3, 4]], mask=[[False, True], [False, False]]),
            "scale": np.ma.array([[0.1, 0.2], [0.3, 0.4]], mask=[[False, True], [False, False]]),
        },
        {
            "loc": np.ma.array([[5, 6], [7, 8]], mask=[[False, False], [True, False]]),
            "scale": np.ma.array([[0.5, 0.6], [0.7, 0.8]], mask=[[False, False], [True, False]]),
        },
        {
            "loc": np.ma.array([[9, 10], [11, 12]], mask=[[False, False], [False, False]]),
            "scale": np.ma.array([[0.9, 0.1], [0.11, 0.12]], mask=[[False, False], [False, False]]),
        },
    ]
    aggregator = MixedNormalAggregator()
    result = aggregator.aggregate(y)
    assert "loc" in result and "scale" in result
    expected = np.ma.average(np.ma.stack([y[0]["loc"], y[1]["loc"], y[2]["loc"]], axis=0), axis=0)
    assert np.ma.allclose(result["loc"], expected), (
        "MixedNormalAggregator failed with masked arrays."
    )


def test_mixed_normal_aggregator_invalid_input():
    """Test the MixedNormalAggregator with invalid input arrays."""
    y = [
        {
            "scale": np.array([[0.1, 0.2], [0.3, 0.4]]),
        },
        {
            "loc": np.array([[5, 6], [7, 8]]),
            "scale": np.array([[0.5, 0.6], [0.7, 0.8]]),
        },
    ]

    aggregator = MixedNormalAggregator()
    with pytest.raises(
        ValueError,
        match="All elements of 'y' must have a 'loc' key.",
    ):
        aggregator.aggregate(y)

    y = [
        {
            "loc": np.array([[1, 2], [3, 4]]),
        },
        {
            "loc": np.array([[5, 6], [7, 8]]),
            "scale": np.array([[0.5, 0.6], [0.7, 0.8]]),
        },
    ]

    aggregator = MixedNormalAggregator()
    with pytest.raises(
        ValueError,
        match="All elements of 'y' must have a 'scale' key.",
    ):
        aggregator.aggregate(y)
