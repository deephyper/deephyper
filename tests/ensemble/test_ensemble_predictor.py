import pytest
import numpy as np

from deephyper.predictor import Predictor
from deephyper.ensemble.aggregator import MeanAggregator
from deephyper.ensemble import EnsemblePredictor


class MockPredictor(Predictor):
    """Mock Predictor for testing."""

    def __init__(self, output):
        self.output = output

    def predict(self, X):
        return self.output


@pytest.fixture
def setup():
    """Set up mock predictors and aggregator."""
    predictors = [MockPredictor(np.array([i])) for i in range(3)]
    aggregator = MeanAggregator()
    return predictors, aggregator


def test_initialization_valid(setup):
    """Test valid initialization of EnsemblePredictor."""
    predictors, aggregator = setup
    predictor = EnsemblePredictor(predictors=predictors, aggregator=aggregator)
    assert predictor.predictors == predictors
    assert predictor.aggregator == aggregator
    assert predictor.evaluator_method == "thread"
    assert predictor._evaluator is not None

    predictors, aggregator = setup
    predictor = EnsemblePredictor(predictors=predictors, aggregator=aggregator, evaluator="process")
    assert predictor.predictors == predictors
    assert predictor.aggregator == aggregator
    assert predictor.evaluator_method == "process"
    assert predictor._evaluator is not None

    predictors, aggregator = setup
    predictor = EnsemblePredictor(
        predictors=predictors,
        aggregator=aggregator,
        evaluator={"method": "process", "method_kwargs": {"num_workers": 4}},
    )
    assert predictor.predictors == predictors
    assert predictor.aggregator == aggregator
    assert predictor.evaluator_method == "process"
    assert predictor.evaluator_method_kwargs == {"num_workers": 4}
    assert predictor._evaluator is not None


def test_initialization_invalid_evaluator(setup):
    """Test invalid evaluator argument."""
    predictors, aggregator = setup
    with pytest.raises(ValueError, match="evaluator must be either None or str or dict"):
        EnsemblePredictor(predictors=predictors, aggregator=aggregator, evaluator=42)


def test_predict(setup):
    """Test the predict method of EnsemblePredictor."""
    predictors, aggregator = setup
    predictor = EnsemblePredictor(predictors=predictors, aggregator=aggregator)

    X = np.array([[1, 2], [3, 4]])  # Mock input, not used in mock predictors
    result = predictor.predict(X)
    expected_output = np.mean([0, 1, 2])  # Outputs of mock predictors
    assert np.isclose(result, expected_output)


def test_predictions_from_predictors(setup):
    """Test predictions_from_predictors method."""
    predictors, aggregator = setup

    predictor = EnsemblePredictor(predictors=predictors, aggregator=aggregator)
    y = predictor.predictions_from_predictors(X=np.array([[1, 2]]), predictors=predictors)
    expected_output = [np.array([0]), np.array([1]), np.array([2])]  # Mock outputs
    assert len(y) == 3  # 3 mock predictors
    for r, e in zip(y, expected_output):
        assert np.array_equal(r, e)

    predictor = EnsemblePredictor(predictors=predictors, aggregator=aggregator, evaluator="process")
    y = predictor.predictions_from_predictors(X=np.array([[1, 2]]), predictors=predictors)
    expected_output = [np.array([0]), np.array([1]), np.array([2])]  # Mock outputs
    assert len(y) == 3  # 3 mock predictors
    for r, e in zip(y, expected_output):
        assert np.array_equal(r, e)


def test_predict_error_handling(setup):
    """Test error handling during prediction."""

    class FailingPredictor(MockPredictor):
        def predict(self, X):
            raise RuntimeError("Intentional failure")

    failing_predictor = FailingPredictor(None)
    predictors, aggregator = setup
    predictors.append(failing_predictor)
    predictor = EnsemblePredictor(predictors=predictors, aggregator=aggregator)

    X = np.array([[1, 2], [3, 4]])  # Mock input
    with pytest.raises(RuntimeError, match="Failed to call .predict"):
        predictor.predict(X)


def test_weights(setup):
    """Test aggregation with weights."""
    predictors, _ = setup
    predictor = EnsemblePredictor(
        predictors=predictors, aggregator=MeanAggregator(), weights=[0.1, 0.3, 0.6]
    )

    X = np.array([[1, 2], [3, 4]])  # Mock input
    result = predictor.predict(X)
    expected_output = np.average([0, 1, 2], weights=[0.1, 0.3, 0.6])  # Outputs of mock predictors
    assert np.isclose(result, expected_output)
