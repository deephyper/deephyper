import numpy as np
import scipy.stats as ss
import pytest

from deephyper.ensemble.aggregator import MixedCategoricalAggregator


def test_mixed_categorical_aggregator_confidence():
    """Test the MixedCategoricalAggregator with confidence uncertainty."""
    y = [np.array([[0.6, 0.4], [0.7, 0.3]]), np.array([[0.5, 0.5], [0.8, 0.2]])]
    aggregator = MixedCategoricalAggregator(uncertainty_method="confidence")
    result = aggregator.aggregate(y)

    # Expected probabilities
    expected_probs = np.mean(np.stack(y, axis=0), axis=0)
    assert np.allclose(result["loc"], expected_probs), "Aggregated probabilities are incorrect."

    # Expected uncertainty
    expected_uncertainty = 1 - np.max(expected_probs, axis=-1)
    assert np.allclose(result["uncertainty"], expected_uncertainty), (
        "Uncertainty calculation is incorrect."
    )


def test_mixed_categorical_aggregator_entropy():
    """Test the MixedCategoricalAggregator with entropy uncertainty."""
    y = [np.array([[0.6, 0.4], [0.7, 0.3]]), np.array([[0.5, 0.5], [0.8, 0.2]])]
    aggregator = MixedCategoricalAggregator(uncertainty_method="entropy")
    result = aggregator.aggregate(y)

    # Expected probabilities
    expected_probs = np.mean(np.stack(y, axis=0), axis=0)
    assert np.allclose(result["loc"], expected_probs), "Aggregated probabilities are incorrect."

    # Expected uncertainty
    expected_uncertainty = ss.entropy(expected_probs, axis=-1)

    assert np.allclose(result["uncertainty"], expected_uncertainty), (
        "Uncertainty calculation is incorrect."
    )


def test_mixed_categorical_aggregator_with_weights():
    """Test the MixedCategoricalAggregator with weighted predictions."""
    y = [np.array([[0.6, 0.4], [0.7, 0.3]]), np.array([[0.5, 0.5], [0.8, 0.2]])]
    weights = [0.3, 0.7]
    aggregator = MixedCategoricalAggregator(uncertainty_method="confidence")
    result = aggregator.aggregate(y, weights=weights)

    # Expected probabilities
    weighted_probs = np.average(np.stack(y, axis=0), axis=0, weights=weights)
    assert np.allclose(result["loc"], weighted_probs), "Weighted aggregation failed."

    # Expected uncertainty
    expected_uncertainty = 1 - np.max(weighted_probs, axis=-1)
    assert np.allclose(result["uncertainty"], expected_uncertainty), (
        "Uncertainty calculation is incorrect with weights."
    )


def test_mixed_categorical_aggregator_decomposed_uncertainty():
    """Test MixedCategoricalAggregator with decomposed uncertainty."""
    y = [np.array([[0.6, 0.4], [0.7, 0.3]]), np.array([[0.5, 0.5], [0.8, 0.2]])]
    aggregator = MixedCategoricalAggregator(
        uncertainty_method="confidence", decomposed_uncertainty=True
    )
    result = aggregator.aggregate(y)

    # Expected probabilities
    expected_probs = np.mean(np.stack(y, axis=0), axis=0)
    assert np.allclose(result["loc"], expected_probs), "Aggregated probabilities are incorrect."

    # Expected uncertainty
    expected_uncertainty = 1 - np.max(expected_probs, axis=-1)
    aleatoric_uncertainty = np.mean([1 - np.max(p, axis=-1) for p in y], axis=0)
    epistemic_uncertainty = np.maximum(0, expected_uncertainty - aleatoric_uncertainty)

    assert np.allclose(result["uncertainty_aleatoric"], aleatoric_uncertainty), (
        "Aleatoric uncertainty is incorrect."
    )
    assert np.allclose(result["uncertainty_epistemic"], epistemic_uncertainty), (
        "Epistemic uncertainty is incorrect."
    )


def test_mixed_categorical_aggregator_invalid_uncertainty_method():
    """Test the MixedCategoricalAggregator with an invalid uncertainty method."""
    with pytest.raises(ValueError, match="Invalid uncertainty_method 'invalid'."):
        MixedCategoricalAggregator(uncertainty_method="invalid")


def test_mixed_categorical_aggregator_invalid_y():
    """Test the MixedCategoricalAggregator with invalid input for y."""
    y = ["not_an_array", np.array([[0.5, 0.5], [0.8, 0.2]])]
    aggregator = MixedCategoricalAggregator()
    with pytest.raises(TypeError, match="Input `y` must be a list of numpy.ndarray."):
        aggregator.aggregate(y)


def test_mixed_categorical_aggregator_invalid_weights():
    """Test the MixedCategoricalAggregator with mismatched weights."""
    y = [np.array([[0.6, 0.4]]), np.array([[0.5, 0.5]])]
    weights = [0.5]  # Incorrect length
    aggregator = MixedCategoricalAggregator()
    with pytest.raises(
        ValueError,
        match="The length of `weights` must match the number of predictors in `y`.",
    ):
        aggregator.aggregate(y, weights=weights)


def test_mixed_categorical_aggregator_empty_y():
    """Test the MixedCategoricalAggregator with an empty input list."""
    y = []
    aggregator = MixedCategoricalAggregator()
    with pytest.raises(ValueError, match="need at least one array to stack"):
        aggregator.aggregate(y)


def test_mixed_categorical_aggregator_single_predictor():
    """Test the MixedCategoricalAggregator with a single predictor."""
    y = [np.array([[0.6, 0.4], [0.7, 0.3]])]
    aggregator = MixedCategoricalAggregator()
    result = aggregator.aggregate(y)

    # If there's only one predictor, the output should match it
    assert np.allclose(result["loc"], y[0]), "Failed to handle single predictor case."


def test_mixed_categorical_aggregator_confidence_with_masked_arrays():
    """Test the MixedCategoricalAggregator with confidence uncertainty."""
    y = [
        np.ma.array([[0.6, 0.4], [0.7, 0.3]], mask=[[True, True], [True, True]]),
        np.ma.array([[0.5, 0.5], [0.8, 0.2]], mask=[[True, True], [False, False]]),
    ]
    aggregator = MixedCategoricalAggregator(uncertainty_method="confidence")
    result = aggregator.aggregate(y)

    # Expected probabilities
    expected_probs = np.ma.mean(np.ma.stack(y, axis=0), axis=0)
    assert isinstance(expected_probs, np.ma.MaskedArray)
    assert np.ma.allclose(result["loc"], expected_probs), "Aggregated probabilities are incorrect."

    # Expected uncertainty
    expected_uncertainty = 1 - np.ma.max(expected_probs, axis=-1)
    assert isinstance(result["uncertainty"], np.ma.MaskedArray)
    assert np.ma.allclose(result["uncertainty"], expected_uncertainty), (
        "Uncertainty calculation is incorrect."
    )


def test_mixed_categorical_aggregator_entropy_with_masked_arrays():
    """Test the MixedCategoricalAggregator with entropy uncertainty."""
    y = [
        np.ma.array([[0.6, 0.4], [0.7, 0.3]], mask=[[True, True], [True, True]]),
        np.ma.array([[0.5, 0.5], [0.8, 0.2]], mask=[[True, True], [False, False]]),
    ]
    aggregator = MixedCategoricalAggregator(uncertainty_method="entropy")
    result = aggregator.aggregate(y)

    # Expected probabilities
    expected_probs = np.ma.mean(np.ma.stack(y, axis=0), axis=0)
    assert isinstance(expected_probs, np.ma.MaskedArray)
    assert np.ma.allclose(result["loc"], expected_probs), "Aggregated probabilities are incorrect."

    # Expected uncertainty
    expected_uncertainty = ss.entropy(expected_probs, axis=-1)
    assert isinstance(result["uncertainty"], np.ma.MaskedArray)
    assert np.ma.allclose(result["uncertainty"], expected_uncertainty), (
        "Uncertainty calculation is incorrect."
    )


if __name__ == "__main__":
    test_mixed_categorical_aggregator_entropy_with_masked_arrays()
