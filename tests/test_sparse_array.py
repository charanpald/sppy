"""Sparse array contracts, expressed as Arrange–Act–Assert pytest cases."""

from __future__ import annotations

import operator
import pickle
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import sparse

import sppy


@pytest.mark.parametrize("shape", [(5, 7), (1, 1), (1, 0), (0, 0), (5,), (0,), 5, 0])
def test_empty_constructor_metadata(
    shape: tuple[int, ...], dtype: np.dtype[Any], order: str
) -> None:
    # Arrange
    """Verify empty constructor metadata."""
    expected_shape = (shape,) if isinstance(shape, int) else shape

    # Act
    result = sppy.SparseArray(shape, dtype=dtype, order=order)

    # Assert
    assert result.shape == expected_shape
    assert result.ndim == len(expected_shape)
    assert result.size == np.prod(expected_shape)
    assert result.dtype == dtype
    assert result.order == order
    assert result.nnz == 0


def test_dense_constructor_preserves_values(
    dense: NDArray[Any], dtype: np.dtype[Any], order: str
) -> None:
    # Arrange
    """Verify dense constructor preserves values."""
    expected = dense.astype(dtype)

    # Act
    result = sppy.SparseArray(expected, order=order)

    # Assert
    np.testing.assert_array_equal(result.toarray(), expected)
    assert result.toarray().dtype == dtype
    assert result.nnz == np.count_nonzero(expected)


@pytest.mark.parametrize(
    "source_format",
    [sparse.csr_matrix, sparse.csc_matrix, sparse.coo_matrix, sparse.csr_array],
)
def test_scipy_constructor(source_format: str, order: str) -> None:
    # Arrange
    """Verify scipy constructor."""
    expected = np.array([[0.0, 2.0], [-3.0, 0.0]])
    source = source_format(expected)

    # Act
    result = sppy.SparseArray(source, order=order)

    # Assert
    np.testing.assert_array_equal(result.toarray(), expected)
    assert result.nnz == source.nnz


@pytest.mark.parametrize(
    "source", [(0, 1, 2), (-1, 2), (2**31, 2), (1.5, 2), "invalid"]
)
def test_constructor_rejects_invalid_shapes(source: Any) -> None:
    # Arrange
    """Verify constructor rejects invalid shapes."""
    constructor = sppy.SparseArray

    # Act / Assert
    with pytest.raises(ValueError):
        constructor(source)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"order": "X"},
        {"dtype": np.uint64},
    ],
)
def test_constructor_rejects_invalid_options(kwargs: Any) -> None:
    # Arrange
    """Verify constructor rejects invalid options."""
    shape = (2, 3)

    # Act / Assert
    with pytest.raises(ValueError):
        sppy.SparseArray(shape, **kwargs)


def test_sparse_constructor_casts_and_copies(dense: NDArray[Any], order: str) -> None:
    # Arrange
    """Verify sparse constructor casts and copies."""
    source = sppy.asarray(dense)
    expected = dense.astype(np.int64)

    # Act
    result = sppy.SparseArray(source, dtype=np.int64, order=order)

    # Assert
    assert result is not source
    assert result.dtype == np.dtype(np.int64)
    np.testing.assert_array_equal(result.toarray(), expected)


@pytest.mark.parametrize(
    "operation",
    [operator.abs, operator.neg, lambda a: a * 2, lambda a: 0.5 * a, lambda a: a**2],
)
def test_elementwise_scalar_operations(
    dense: NDArray[Any], order: str, operation: str
) -> None:
    # Arrange
    """Verify elementwise scalar operations."""
    array = sppy.asarray(dense, order=order)
    expected = operation(dense)

    # Act
    result = operation(array)

    # Assert
    np.testing.assert_allclose(result.toarray(), expected)
    np.testing.assert_array_equal(array.toarray(), dense)


@pytest.mark.parametrize("operation", [operator.add, operator.sub, operator.mul])
def test_binary_operations_promote_and_match_numpy(
    dense: NDArray[Any], order: str, operation: str
) -> None:
    # Arrange
    """Verify binary operations promote and match numpy."""
    left = sppy.asarray(dense.astype(np.int64), order=order)
    right = sppy.asarray(dense.astype(np.float32), order="F" if order == "C" else "C")
    expected = operation(dense.astype(np.int64), dense.astype(np.float32))

    # Act
    result = operation(left, right)

    # Assert
    assert result.dtype == expected.dtype
    np.testing.assert_allclose(result.toarray(), expected)


@pytest.mark.parametrize(
    "operation", [operator.add, operator.sub, operator.mul, operator.matmul]
)
def test_binary_operations_reject_incompatible_shapes(operation: str) -> None:
    # Arrange
    """Verify binary operations reject incompatible shapes."""
    left = sppy.zeros((2, 3))
    right = sppy.zeros((4, 5))

    # Act / Assert
    with pytest.raises(ValueError):
        operation(left, right)


@pytest.mark.parametrize("shape", [(3, 4), (5,)])
def test_copy_is_independent(shape: tuple[int, ...], order: str) -> None:
    # Arrange
    """Verify copy is independent."""
    source = sppy.ones(shape, order=order)
    copied = source.copy()

    # Act
    source[...] = 3

    # Assert
    np.testing.assert_array_equal(copied.toarray(), np.ones(shape))
    np.testing.assert_array_equal(source.toarray(), np.full(shape, 3))
    assert copied.order == order


@pytest.mark.parametrize("method", ["reserve", "compress", "eliminate_zeros"])
def test_storage_operations_preserve_values(method: str, order: str) -> None:
    # Arrange
    """Verify storage operations preserve values."""
    expected = np.array([[0.0, 2.0], [-3.0, 0.0]])
    array = sppy.asarray(expected, order=order)
    arguments = (20,) if method == "reserve" else ()

    # Act
    getattr(array, method)(*arguments)

    # Assert
    np.testing.assert_array_equal(array.toarray(), expected)


def test_reserve_rejects_negative_capacity() -> None:
    # Arrange
    """Verify reserve rejects negative capacity."""
    array = sppy.zeros((2, 2))

    # Act / Assert
    with pytest.raises(ValueError):
        array.reserve(-1)


def test_nonzero_coordinates_match_stored_values(
    dense: NDArray[Any], order: str
) -> None:
    # Arrange
    """Verify nonzero coordinates match stored values."""
    array = sppy.asarray(dense, order=order)
    expected = np.zeros_like(dense)

    # Act
    coordinates = array.nonzero()
    values = array.values()
    expected[coordinates] = values

    # Assert
    np.testing.assert_array_equal(expected, dense)
    assert len(values) == array.nnz
    assert len(coordinates) == dense.ndim


@pytest.mark.parametrize("axis", [0, 1])
def test_row_and_column_indices(axis: int | None, order: str) -> None:
    # Arrange
    """Verify row and column indices."""
    dense = np.array([[0, 2, 0], [3, 0, -1]])
    array = sppy.asarray(dense, order=order)
    expected = np.flatnonzero(dense[1, :] if axis == 0 else dense[:, 1])
    method = array.row_indices if axis == 0 else array.column_indices

    # Act
    result = method(1)

    # Assert
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("method", ["row_indices", "column_indices"])
def test_axis_indices_reject_out_of_bounds(method: str) -> None:
    # Arrange
    """Verify axis indices reject out of bounds."""
    array = sppy.zeros((2, 3))

    # Act / Assert
    with pytest.raises(ValueError):
        getattr(array, method)(10)


def test_nonzero_rows_list(order: str) -> None:
    # Arrange
    """Verify nonzero rows list."""
    dense = np.array([[0.0, 2.0, -1.0], [3.0, 0.0, 0.0]])
    array = sppy.asarray(dense, order=order)
    expected = [np.flatnonzero(row) for row in dense]

    # Act
    result = array.nonzero_rows_list()

    # Assert
    assert len(result) == len(expected)
    for actual, indices in zip(result, expected, strict=True):
        np.testing.assert_array_equal(actual, indices)


def test_nonzero_rows_ptr(order: str) -> None:
    # Arrange
    """Verify nonzero rows ptr."""
    dense = np.array([[0.0, 2.0, -1.0], [3.0, 0.0, 0.0]])
    array = sppy.asarray(dense, order=order)
    expected = sparse.csr_matrix(dense)

    # Act
    indptr, columns = array.nonzero_rows_ptr()

    # Assert
    np.testing.assert_array_equal(indptr, expected.indptr)
    np.testing.assert_array_equal(columns, expected.indices)


@pytest.mark.parametrize("init", [False, True])
@pytest.mark.parametrize("values", [2.5, np.array([1.0, -2.0, 3.0])])
def test_put_assigns_values(init: bool, values: Any, order: str) -> None:
    # Arrange
    """Verify put assigns values."""
    array = sppy.zeros((4, 5), order=order)
    rows = np.array([0, 1, 3], dtype=np.int32)
    columns = np.array([1, 3, 2], dtype=np.int32)
    expected = np.zeros(array.shape)
    expected[rows, columns] = values

    # Act
    array.put(values, rows, columns, init=init)

    # Assert
    np.testing.assert_array_equal(array.toarray(), expected)
    assert array.nnz == 3


def test_initial_put_sums_duplicate_triplets(order: str) -> None:
    # Arrange
    """Verify initial put sums duplicate triplets."""
    array = sppy.zeros((2, 2), order=order)
    rows = np.array([0, 0, 1], dtype=np.int32)
    columns = np.array([1, 1, 0], dtype=np.int32)
    values = np.array([2.0, 3.0, -1.0])

    # Act
    array.put(values, rows, columns, init=True)

    # Assert
    np.testing.assert_array_equal(array.toarray(), [[0, 5], [-1, 0]])


def test_put_supports_large_sparse_dimensions(order: str) -> None:
    # Arrange
    """Verify put supports large sparse dimensions."""
    array = sppy.zeros((1_000_000, 1_000_000), order=order)
    rows = np.array([1, 900_000], dtype=np.int32)
    columns = np.array([999_999, 3], dtype=np.int32)
    values = np.array([1.5, -2.0])

    # Act
    array.put(values, rows, columns, init=True)

    # Assert
    np.testing.assert_array_equal(array[rows, columns], values)
    assert array.nnz == 2


@pytest.mark.parametrize("method", ["to_scipy_csr", "to_scipy_csc"])
def test_scipy_export(method: str, dtype: np.dtype[Any], order: str) -> None:
    # Arrange
    """Verify scipy export."""
    dense = np.array([[0, 2], [-3, 0]], dtype=dtype)
    array = sppy.asarray(dense, order=order)

    # Act
    result = getattr(array, method)()

    # Assert
    np.testing.assert_array_equal(result.toarray(), dense)
    assert result.dtype == dtype
    assert result.format == method.rsplit("_", 1)[1]


def test_pickle_roundtrip(
    dense: NDArray[Any], dtype: np.dtype[Any], order: str
) -> None:
    # Arrange
    """Verify pickle roundtrip."""
    dense = dense.astype(dtype)
    array = sppy.asarray(dense, order=order)

    # Act
    result = pickle.loads(pickle.dumps(array))

    # Assert
    np.testing.assert_array_equal(result.toarray(), dense)
    assert result.dtype == dtype
    assert result.order == order


def test_string_describes_array() -> None:
    # Arrange
    """Verify string describes array."""
    array = sppy.asarray([[0.0, 2.0], [3.0, 0.0]])

    # Act
    result = str(array)
    representation = repr(array)

    # Assert
    assert "shape:(2, 2)" in result
    assert "non-zeros:2" in result
    assert "SparseArray" in representation


def test_prune_removes_exact_stored_zero(order: str) -> None:
    # Arrange
    """Verify prune removes exact stored zero."""
    array = sppy.asarray([[1.0, 2.0], [0.0, 3.0]], order=order)
    array[0, 1] = 0
    expected = np.diag([1.0, 3.0])

    # Act
    array.prune()

    # Assert
    np.testing.assert_array_equal(array.toarray(), expected)
    assert array.nnz == 2


@pytest.mark.parametrize("method", ["submatrix", "subarray"])
def test_submatrix_selection(method: str, order: str) -> None:
    # Arrange
    """Verify submatrix selection."""
    dense = np.arange(20.0).reshape(4, 5)
    array = sppy.asarray(dense, order=order)
    arguments = (
        (1, 2, 2, 3)
        if method == "submatrix"
        else (np.array([1, 2]), np.array([2, 3, 4]))
    )
    expected = dense[1:3, 2:5]

    # Act
    result = getattr(array, method)(*arguments)

    # Assert
    np.testing.assert_array_equal(result.toarray(), expected)


@pytest.mark.parametrize(
    "arguments", [(-1, 0, 2, 2), (0, -1, 2, 2), (0, 0, 20, 2), (0, 0, 2, 20)]
)
def test_submatrix_rejects_invalid_block(arguments: Any) -> None:
    # Arrange
    """Verify submatrix rejects invalid block."""
    array = sppy.zeros((3, 4))

    # Act / Assert
    with pytest.raises(ValueError):
        array.submatrix(*arguments)


def test_transpose_matches_numpy(dense: NDArray[Any], order: str) -> None:
    # Arrange
    """Verify transpose matches numpy."""
    array = sppy.asarray(dense, order=order)
    expected = dense.T

    # Act
    result = array.T

    # Assert
    np.testing.assert_array_equal(result.toarray(), expected)


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("operand_shape", [(4,), (4, 3)])
def test_dense_dot_matches_numpy(
    parallel: bool, operand_shape: Any, order: str, rng: np.random.Generator
) -> None:
    # Arrange
    """Verify dense dot matches numpy."""
    dense = np.array([[0.0, 1.0, 0.0, -2.0], [3.0, 0.0, 4.0, 0.0]])
    array = sppy.asarray(dense, order=order)
    operand = rng.normal(size=operand_shape)
    expected = dense @ operand
    if parallel and operand.ndim == 2:
        array = array.copy(order="C")
    operation = array.parallel_dot if parallel else array.dot

    # Act
    result = operation(operand)

    # Assert
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("vector", [False, True])
def test_sparse_dot_matches_numpy(vector: bool, order: str) -> None:
    # Arrange
    """Verify sparse dot matches numpy."""
    dense = (
        np.array([0.0, 2.0, -3.0]) if vector else np.array([[0.0, 2.0], [-3.0, 1.0]])
    )
    array = sppy.asarray(dense, order=order)
    expected = dense @ dense

    # Act
    result = array @ array

    # Assert
    np.testing.assert_allclose(np.asarray(result), expected)


@pytest.mark.parametrize("shape", [(2, 3), (4,)])
def test_fill_ones(shape: tuple[int, ...]) -> None:
    # Arrange
    """Verify fill ones."""
    array = sppy.zeros(shape)
    expected = np.ones(shape)

    # Act
    array.ones()

    # Assert
    np.testing.assert_array_equal(array.toarray(), expected)
