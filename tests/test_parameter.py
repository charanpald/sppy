"""Validation helpers reject invalid inputs rather than silently passing tests."""

import numpy as np
import pytest

from sppy.lib.Parameter import Parameter


@pytest.mark.parametrize(
    "value,lower,upper",
    [
        (2, 0, 5),
        (0, 0, 5),
        (5, 0, 5),
        (2, 2, 2),
        (2, 0, np.inf),
        (2, -np.inf, 5),
        (np.int32(2), np.int32(0), np.int32(5)),
    ],
)
def test_check_int_accepts_valid_range(value, lower, upper):
    # Arrange
    validate = Parameter.checkInt

    # Act
    result = validate(value, lower, upper)

    # Assert
    assert result is True


@pytest.mark.parametrize(
    "value,lower,upper",
    [
        (2, 5, 0),
        (2, 0.0, 5),
        (2, 0, 5.0),
        (-1, 0, 5),
        (6, 0, 5),
        (np.array([1]), 0, 5),
        (2.0, 0, 5),
    ],
)
def test_check_int_rejects_invalid_range(value, lower, upper):
    # Arrange
    validate = Parameter.checkInt

    # Act / Assert
    with pytest.raises(ValueError):
        validate(value, lower, upper)


@pytest.mark.parametrize(
    "value,lower,upper",
    [
        (2.0, 0.0, 5.0),
        (0.0, 0.0, 5.0),
        (5.0, 0.0, 5.0),
        (2.0, 2.0, 2.0),
        (2.0, 0.0, np.inf),
        (2.0, -np.inf, 5.0),
        (np.float64(2), np.float64(0), np.float64(5)),
    ],
)
def test_check_float_accepts_valid_range(value, lower, upper):
    # Arrange
    validate = Parameter.checkFloat

    # Act
    result = validate(value, lower, upper)

    # Assert
    assert result is True


@pytest.mark.parametrize(
    "value,lower,upper",
    [
        (2.0, 5.0, 0.0),
        (2.0, 0, 5.0),
        (2.0, 0.0, 5),
        (2, 0.0, 5.0),
        (-1.0, 0.0, 5.0),
        (6.0, 0.0, 5.0),
    ],
)
def test_check_float_rejects_invalid_range(value, lower, upper):
    # Arrange
    validate = Parameter.checkFloat

    # Act / Assert
    with pytest.raises(ValueError):
        validate(value, lower, upper)


@pytest.mark.parametrize("value", ["a", "b", "c"])
def test_check_string_accepts_member(value):
    # Arrange
    allowed = ["a", "b", "c"]

    # Act
    result = Parameter.checkString(value, allowed)

    # Assert
    assert result is True


@pytest.mark.parametrize(
    "value,allowed", [("d", ["a", "b"]), (5, ["a", "b"]), ("a", "a")]
)
def test_check_string_rejects_invalid_input(value, allowed):
    # Arrange
    validate = Parameter.checkString

    # Act / Assert
    with pytest.raises(ValueError):
        validate(value, allowed)


@pytest.mark.parametrize(
    "values,validate,bounds",
    [
        ([1, 2, 3, 2], Parameter.checkInt, [1, 3]),
        ([0.1, 0.6, 1.4], Parameter.checkFloat, [0.1, 3.0]),
        (np.array([0.1, 0.6, 1.4]), Parameter.checkFloat, [0.1, 3.0]),
    ],
)
def test_check_list_accepts_valid_values(values, validate, bounds):
    # Arrange
    check = Parameter.checkList

    # Act
    result = check(values, validate, bounds)

    # Assert
    assert result is True


@pytest.mark.parametrize(
    "values", [[1, 2, 4], [1, 2, 0], [1, 2, 1.2], "a", np.array([[1, 2]])]
)
def test_check_list_rejects_invalid_values(values):
    # Arrange
    bounds = [1, 3]

    # Act / Assert
    with pytest.raises(ValueError):
        Parameter.checkList(values, Parameter.checkInt, bounds)


@pytest.mark.parametrize("value", [True, False])
def test_check_boolean_accepts_boolean(value):
    # Arrange
    validate = Parameter.checkBoolean

    # Act
    result = validate(value)

    # Assert
    assert result is True


@pytest.mark.parametrize("value", [0, 1, "s"])
def test_check_boolean_rejects_non_boolean(value):
    # Arrange
    validate = Parameter.checkBoolean

    # Act / Assert
    with pytest.raises(ValueError):
        validate(value)


@pytest.mark.parametrize("value,expected_type", [(2, int), (True, bool), ([], list)])
def test_check_class_accepts_expected_type(value, expected_type):
    # Arrange
    validate = Parameter.checkClass

    # Act
    result = validate(value, expected_type)

    # Assert
    assert result is True


@pytest.mark.parametrize("value,expected_type", [(2, list), ([], dict)])
def test_check_class_rejects_unexpected_type(value, expected_type):
    # Arrange
    validate = Parameter.checkClass

    # Act / Assert
    with pytest.raises(ValueError):
        validate(value, expected_type)


@pytest.mark.parametrize("tolerance", [1e-6, 0.2])
def test_check_symmetric_accepts_matrix(tolerance):
    # Arrange
    matrix = np.array([[1.0, 2.0], [2.0, 3.0]])

    # Act
    result = Parameter.checkSymmetric(matrix, tolerance)

    # Assert
    assert result is True


@pytest.mark.parametrize("tolerance", [1e-6, 0.1])
def test_check_symmetric_rejects_asymmetry(tolerance):
    # Arrange
    matrix = np.array([[1.0, 2.1], [2.0, 3.0]])

    # Act / Assert
    with pytest.raises(ValueError):
        Parameter.checkSymmetric(matrix, tolerance)


def test_check_symmetric_honors_tolerance():
    # Arrange
    matrix = np.array([[1.0, 2.1], [2.0, 3.0]])

    # Act
    result = Parameter.checkSymmetric(matrix, 0.2)

    # Assert
    assert result is True
