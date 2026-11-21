"""Regression coverage for numerical and validation defects found during review."""

from typing import Any

import numpy as np
import pytest
from numpy.typing import DTypeLike

import sppy
from sppy.linalg import GeneralLinearOperator, norm


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64])
@pytest.mark.parametrize("order", ["C", "F"])
def test_minimum_integer_is_not_zero(dtype: DTypeLike, order: str) -> None:
    """Signed magnitude overflow must not classify a negative integer as zero."""
    value = np.iinfo(dtype).min
    array = sppy.asarray(np.array([0, value], dtype=dtype), order=order)

    assert not array.is_zero()
    assert not array.is_zero(tolerance=1)
    assert sppy.zeros(2, dtype=dtype, order=order).is_zero()


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("ord", [1, np.inf])
def test_integer_matrix_norm_avoids_magnitude_overflow(
    dtype: DTypeLike, order: str, ord: int | float
) -> None:
    """Matrix norms must promote integers before taking magnitudes or sums."""
    dense = np.array([[np.iinfo(dtype).min, 0], [0, 1]], dtype=dtype)
    np.testing.assert_allclose(
        norm(sppy.asarray(dense, order=order), ord=ord), np.linalg.norm(dense, ord=ord)
    )


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64])
@pytest.mark.parametrize("order", ["C", "F"])
def test_integer_approximation_avoids_subtraction_overflow(
    dtype: DTypeLike, order: str
) -> None:
    """Signed integer endpoints must not appear close through modular subtraction."""
    left = sppy.asarray(np.array([np.iinfo(dtype).min], dtype=dtype), order=order)
    right = sppy.asarray(np.array([np.iinfo(dtype).max], dtype=dtype), order=order)

    assert not left.is_approx(right, rtol=0, atol=1)
    assert not right.is_approx(left, rtol=0, atol=1)
    assert left.is_approx(left, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(3,), (2, 3)])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("scalar", [np.inf, -np.inf, np.nan, complex(np.inf, 1)])
def test_nonfinite_scaling_includes_implicit_zeros(
    shape: tuple[int, ...], order: str, scalar: float | complex
) -> None:
    """Nonfinite scalar multiplication must transform implicit zeros as NumPy does."""
    dense = np.zeros(shape)
    dense.flat[0] = 2
    array = sppy.asarray(dense, order=order)
    with np.errstate(invalid="ignore"):
        expected = dense * scalar
        actual = array * scalar
        reverse = scalar * array

    np.testing.assert_allclose(actual.toarray(), expected, equal_nan=True)
    np.testing.assert_allclose(reverse.toarray(), expected, equal_nan=True)


@pytest.mark.parametrize(
    "operation", ["sum", "mean", "var", "std", "min", "max", "count_nonzero"]
)
@pytest.mark.parametrize("axis", [(-1, -2), (0, -1), (-2, 1)])
def test_reductions_normalize_all_axes(operation: str, axis: tuple[int, ...]) -> None:
    """Mixed positive and negative axis tuples must match dense reductions."""
    dense = np.array([[0.0, 2.0, 3.0], [-1.0, 0.0, 4.0]])
    actual = getattr(sppy.asarray(dense), operation)(axis=axis)
    expected = getattr(np, operation)(dense, axis=axis)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("axis", [(0, 0), (-1, 1), (0, None), ((0,), 1), (0, 2)])
def test_reductions_reject_invalid_axis_tuples(axis: Any) -> None:
    """Duplicate, nested, missing, and out-of-range axes must be rejected."""
    with pytest.raises(ValueError):
        sppy.eye(2).sum(axis=axis)


@pytest.mark.parametrize("axes", [(0, 0), (0, -1), (0, 0, 0)])
def test_vector_transpose_rejects_repeated_axes(axes: tuple[int, ...]) -> None:
    """Vector transposition must validate the full axis list."""
    with pytest.raises(ValueError):
        sppy.ones(3).transpose(*axes)


@pytest.mark.parametrize("axis1,axis2", [(2, 1), (0, 3), (-3, 0), (0, 0)])
def test_diagonal_rejects_invalid_axes(axis1: int, axis2: int) -> None:
    """Diagonal extraction must reject invalid axes before modulo normalization."""
    with pytest.raises(ValueError):
        sppy.eye(2).diagonal(axis1=axis1, axis2=axis2)


@pytest.mark.parametrize("operation", ["min", "max"])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("axis", [0, 1])
def test_complex_extrema_with_infinite_real_parts(
    operation: str, order: str, axis: int
) -> None:
    """Complex reduction identities must bound both real and imaginary components."""
    value = complex(np.inf, 3) if operation == "min" else complex(-np.inf, -3)
    dense = np.full((2, 3), value)
    actual = getattr(sppy.asarray(dense, order=order), operation)(axis=axis)
    np.testing.assert_array_equal(actual, getattr(np, operation)(dense, axis=axis))


@pytest.mark.parametrize("operation", ["matvec", "rmatvec", "matmat", "rmatmat"])
def test_parallel_dense_linear_operator(operation: str) -> None:
    """Dense operators must support parallel mode and Hermitian reverse products."""
    dense = np.array([[1 + 2j, 0, 3], [0, 4j, 5]])
    operator = GeneralLinearOperator.as_linear_operator(dense, parallel=True)
    matrix = dense.conj().T if operation.startswith("r") else dense
    operand = np.ones(
        (matrix.shape[1], 2) if operation.endswith("mat") else matrix.shape[1]
    )
    np.testing.assert_allclose(getattr(operator, operation)(operand), matrix @ operand)


def test_rand_accepts_seed() -> None:
    """Integer RNG seeds must produce repeatable sparse samples."""
    first = sppy.rand((5, 6), 0.5, rng=21)
    second = sppy.rand((5, 6), 0.5, rng=21)
    np.testing.assert_array_equal(first.toarray(), second.toarray())


@pytest.mark.parametrize("offset", [0.5, 100.5])
@pytest.mark.parametrize("operation", ["eye", "diag", "diagonal", "set_diagonal"])
def test_diagonal_offsets_require_integers(offset: float, operation: str) -> None:
    """Fractional diagonal offsets must fail even when the diagonal is empty."""
    with pytest.raises(TypeError):
        if operation == "eye":
            sppy.eye(2, k=offset)
        elif operation == "diag":
            sppy.diag([1, 2], k=offset)
        elif operation == "diagonal":
            sppy.eye(2).diagonal(offset)
        else:
            sppy.eye(2).set_diagonal(1, offset)
