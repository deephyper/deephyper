import pytest
import numpy as np
from deephyper.ensemble.aggregator.utils import average


def test_average_with_ndarray():
    arr = np.array([1, 2, 3, 4])
    result = average(arr)
    expected = np.average(arr)
    assert np.isclose(result, expected), "Failed on plain ndarray without weights"


def test_average_with_ndarray_weights():
    arr = np.array([1, 2, 3, 4])
    weights = [0.1, 0.2, 0.3, 0.4]
    result = average(arr, weights=weights)
    expected = np.average(arr, weights=weights)
    assert np.isclose(result, expected), "Failed on plain ndarray with weights"


def test_average_with_ndarray_axis():
    arr = np.array([[1, 2], [3, 4]])
    result = average(arr, axis=0)
    expected = np.average(arr, axis=0)
    assert np.allclose(result, expected), "Failed on plain ndarray with axis"


def test_average_with_maskedarray():
    arr = np.ma.array([1, 2, 3, 4], mask=[False, True, False, False])
    result = average(arr)
    expected = np.ma.average(arr)
    assert np.isclose(result, expected), "Failed on masked array without weights"


def test_average_with_maskedarray_weights():
    arr = np.ma.array([1, 2, 3, 4], mask=[False, True, False, False])
    weights = [0.1, 0.2, 0.3, 0.4]
    result = average(arr, weights=weights)
    expected = np.ma.average(arr, weights=weights)
    assert np.isclose(result, expected), "Failed on masked array with weights"


def test_average_with_maskedarray_axis():
    arr = np.ma.array([[1, 2], [3, 4]], mask=[[False, True], [False, False]])
    result = average(arr, axis=0)
    expected = np.ma.average(arr, axis=0)
    assert np.allclose(result, expected), "Failed on masked array with axis"


def test_average_invalid_input():
    with pytest.raises(TypeError):
        average("invalid input")
