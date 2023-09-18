import pytest
import numpy as np
from python_forecast_utils.sqrt_transformer import SqrtTransformer

# Test with positive integers


def test_positive_integers():
    transformer = SqrtTransformer()
    X = np.array([1, 4, 9, 16])
    expected_output = np.array([1, 2, 3, 4])
    assert np.array_equal(transformer.transform(X), expected_output)

# Test with decimal numbers


def test_decimal_numbers():
    transformer = SqrtTransformer()
    X = np.array([1.5, 2.25, 3.75])
    expected_output = np.array([1.22474487, 1.5, 1.93649167])
    assert np.allclose(transformer.transform(X), expected_output)

# Test with zeros


def test_zeros():
    transformer = SqrtTransformer()
    X = np.array([0, 0, 0])
    expected_output = np.array([0, 0, 0])
    assert np.array_equal(transformer.transform(X), expected_output)

# Test with negative numbers


def test_negative_numbers():
    transformer = SqrtTransformer()
    X = np.array([-1, -4, -9, -16])
    assert np.isnan(transformer.transform(X)).all()

# Test with empty array


def test_empty_array():
    transformer = SqrtTransformer()
    X = np.array([])
    expected_output = np.array([])
    assert np.array_equal(transformer.transform(X), expected_output)

# Test with non-numeric input


def test_non_numeric_input():
    transformer = SqrtTransformer()
    X = np.array(['a', 'b', 'c'])
    with pytest.raises(TypeError):
        transformer.transform(X)
