"""Storage ownership and NumPy semantics at optimized native boundaries."""

import numpy as np
import pytest

import sppy
from sppy.linalg import norm
from profiling._common import numerical_threads


MATRIX_KEYS = [
    (slice(None), slice(None)),
    (slice(None, None, 2), slice(None, None, 2)),
    (slice(None, None, -1), slice(None, None, -2)),
    (slice(4, 0, -2), slice(5, 0, -2)),
    (slice(-10, 20, 3), slice(-10, 20, 2)),
    (slice(0, 0), slice(None)),
    (slice(None), slice(4, 2)),
    (slice(None, None, 10**100), slice(None, None, -(10**100))),
    (1, slice(None)),
    (-1, slice(None, None, -1)),
    (slice(None), 2),
    (slice(None, None, -2), -1),
    (slice(0, 0), 0),
    (0, slice(0, 0)),
]


@pytest.mark.parametrize("key", MATRIX_KEYS)
def test_basic_slices_match_numpy_and_keep_stored_zeros(key, dtype, order):
    dense = np.arange(30).reshape(5, 6).astype(dtype)
    a = sppy.asarray(dense, order=order)
    a[0, 0] = 0  # Every position is now explicitly stored.
    expected = dense[key]
    result = a[key]
    np.testing.assert_array_equal(result.toarray(), expected)
    assert result.shape == expected.shape
    assert result.dtype == dtype
    assert result.order == order
    assert result.nnz == expected.size
    if result.ndim == 1:
        assert result.outer_size == 1
    a.fill(7)
    np.testing.assert_array_equal(result.toarray(), expected)


@pytest.mark.parametrize(
    "key",
    [
        slice(None),
        slice(None, None, -2),
        slice(1, 5, 2),
        slice(4, 2),
        slice(None, None, 10**100),
    ],
)
def test_vector_slices_are_owned_and_keep_order(key, dtype, order):
    dense = np.arange(6).astype(dtype)
    a = sppy.asarray(dense, order=order)
    a[0] = 0
    actual = a[key]
    np.testing.assert_array_equal(actual.toarray(), dense[key])
    assert actual.nnz == dense[key].size
    assert actual.outer_size == 1
    assert actual.order == order


@pytest.mark.parametrize(
    "shape,key",
    [
        ((0, 4), (slice(None), slice(None, None, -1))),
        ((3, 0), (slice(None, None, -1), slice(None))),
        ((0,), slice(None, None, -1)),
    ],
)
def test_empty_slices_preserve_shape(shape, key, order):
    a = sppy.zeros(shape, order=order)
    result = a[key]
    assert result.shape == np.zeros(shape)[key].shape
    assert result.nnz == 0


def test_basic_slices_of_huge_sparse_axes_do_not_allocate_index_grids():
    n = 2**31 - 1
    vector = sppy.zeros(n)
    vector[-1] = 3
    result = vector[::-2]
    assert result.shape == (len(range(n - 1, -1, -2)),)
    assert result.outer_size == result.nnz == 1
    assert result[0] == 3
    row = sppy.zeros((1, n), order="C")
    row[0, -1] = 4
    selected = row[0, ::-1]
    assert selected.shape == (n,)
    assert selected.outer_size == selected.nnz == 1
    assert selected[0] == 4
    column = sppy.zeros((n, 1), order="F")
    column[-1, 0] = 5
    selected = column[::-1, 0]
    assert selected.outer_size == selected.nnz == 1
    assert selected[0] == 5


@pytest.mark.parametrize("compressed", [False, True])
def test_copy_and_storage_conversion_preserve_entries_without_aliasing(
    compressed, dtype, order
):
    a = sppy.asarray(np.array([[0, 2, 0], [3, 0, 4]], dtype=dtype), order=order)
    a[0, 0] = 0
    if compressed:
        a.compress()
    else:
        a.reserve(np.full(a.outer_size, 4, dtype=np.int32))
    expected = a.toarray()
    for target_order in ("C", "F"):
        copied = a.copy(order=target_order)
        constructed = sppy.SparseArray(a, order=target_order)
        for result in (copied, constructed):
            assert result.order == target_order
            assert result.nnz == a.nnz
            np.testing.assert_array_equal(result.toarray(), expected)
            result[0, 1] = 9
            np.testing.assert_array_equal(a.toarray(), expected)
    values = a.values()
    values[:] = 0
    np.testing.assert_array_equal(a.toarray(), expected)
    np.testing.assert_allclose(norm(a), np.linalg.norm(expected), rtol=1e-6)


@pytest.mark.parametrize("canonical", [False, True])
def test_compressed_import_copies_and_sums_duplicate_indices(canonical, dtype, order):
    # Compressed outer vectors represent rows for C and columns for F.
    indices = np.array([0, 1, 2, 3] if canonical else [2, 0, 2, 1], dtype=np.int32)
    values = np.array([0, 2, 3, 4], dtype=dtype)
    indptr = np.array([0, 4, 4], dtype=np.int32)
    shape = (2, 4) if order == "C" else (4, 2)
    expected = np.zeros(shape, dtype=dtype)
    for index, value in zip(indices, values):
        if order == "C":
            expected[0, index] += value
        else:
            expected[index, 0] += value
    a = sppy.SparseArray.from_compressed(shape, values, indices, indptr, order=order)
    np.testing.assert_array_equal(a.toarray(), expected)
    assert a.nnz == len(np.unique(indices))
    values[:] = 8
    indices[:] = 0
    indptr[:] = 0
    np.testing.assert_array_equal(a.toarray(), expected)


@pytest.mark.parametrize("indptr", [[0, -1, 2], [0, 3, 2], [1, 1, 2]])
def test_compressed_import_rejects_invalid_int32_pointers(indptr):
    with pytest.raises(ValueError, match="pointers"):
        sppy.SparseArray.from_compressed(
            (2, 2), [1.0, 2.0], np.array([0, 1], np.int32), np.array(indptr, np.int32)
        )


def test_wide_invalid_indices_are_rejected_before_narrowing():
    with pytest.raises(ValueError, match="indices"):
        sppy.SparseArray.from_compressed((1, 2), [1.0], [2**32 + 1], [0, 1])


@pytest.mark.parametrize("columns", [0, 1, 8, 17])
def test_dense_products_use_typed_owned_outputs(columns, dtype, order):
    dense = (np.arange(35).reshape(5, 7) % 3).astype(dtype)
    operand = (np.arange(7 * columns).reshape(7, columns) % 3).astype(dtype)
    if dtype.kind == "c":
        dense += 1j * dense
    a = sppy.asarray(dense, order=order)
    # Reverse both arrays along the contraction axis to exercise noncontiguous input.
    operand = operand[::-1]
    expected = dense @ operand
    actual = a @ operand
    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(actual, expected, atol=1e-5)
    operand[:] = 0
    np.testing.assert_allclose(actual, expected, atol=1e-5)


@pytest.mark.parametrize("order_right", ["C", "F"])
def test_rectangular_sparse_products_keep_left_storage(dtype, order, order_right):
    left = (np.arange(35).reshape(5, 7) % 3).astype(dtype)
    right = (np.arange(28).reshape(7, 4) % 3).astype(dtype)
    actual = sppy.asarray(left, order=order) @ sppy.asarray(right, order=order_right)
    assert actual.order == order
    assert actual.dtype == dtype
    np.testing.assert_array_equal(actual.toarray(), left @ right)
    _, indices, indptr = actual.compressed_arrays()
    for begin, end in zip(indptr[:-1], indptr[1:]):
        assert np.all(np.diff(indices[begin:end]) > 0)


@pytest.mark.parametrize("value_dtype", [np.float64, np.complex128])
def test_parallel_row_products_match_serial(value_dtype):
    rng = np.random.default_rng(21)
    dense = rng.normal(size=(512, 12)).astype(value_dtype)
    operand = rng.normal(size=(12, 8)).astype(value_dtype)
    if np.dtype(value_dtype).kind == "c":
        dense += 1j * rng.normal(size=dense.shape)
        operand += 1j * rng.normal(size=operand.shape)
    a = sppy.asarray(dense)
    with numerical_threads(2):
        actual = a @ operand
    np.testing.assert_allclose(actual, dense @ operand, atol=1e-12)


@pytest.mark.native
@pytest.mark.parametrize(
    "descriptor", [(2**63 - 1, 1, 2), (0, 0, 1), (0, -(2**63), 1), (0, 1, -1)]
)
def test_native_slice_rejects_malformed_descriptors(descriptor):
    a = sppy.eye(3)
    with pytest.raises(ValueError):
        a._array.slice_basic(*descriptor, 0, 1, 3, 0)


@pytest.mark.parametrize("shape", [(0, 3), (3, 0), (0, 0), (3, 5)])
@pytest.mark.parametrize("columns", [None, 0, 1, 8])
@pytest.mark.parametrize("value_dtype", [np.int64, np.float64, np.complex128])
def test_empty_sparse_dense_products_initialize_all_outputs(
    shape, columns, value_dtype, order
):
    a = sppy.zeros(shape, dtype=value_dtype, order=order)
    rhs_shape = (shape[1],) if columns is None else (shape[1], columns)
    rhs = np.ones(rhs_shape, dtype=value_dtype)
    expected = np.zeros(shape, dtype=value_dtype) @ rhs
    actual = a @ rhs
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)
    assert a.nnz == 0


@pytest.mark.parametrize("order", ["C", "F"])
def test_compressed_import_accepts_readonly_buffers(order):
    values = np.array([0.0, 2.0, 3.0])
    indices = np.array([0, 1, 1], np.int32)
    indptr = np.array([0, 2, 3], np.int32)
    for array in (values, indices, indptr):
        array.flags.writeable = False
    result = sppy.SparseArray.from_compressed(
        (2, 2), values, indices, indptr, order=order
    )
    expected = np.array([[0.0, 2.0], [0.0, 3.0]])
    np.testing.assert_array_equal(
        result.toarray(), expected if order == "C" else expected.T
    )
    assert result.nnz == 3


@pytest.mark.parametrize("uncompressed", [False, True])
def test_sparse_product_retains_cancellation_and_stored_zeros(
    uncompressed, dtype, order
):
    left = sppy.asarray(np.array([[1, 1, 0], [0, 0, 2]], dtype=dtype), order=order)
    right = sppy.asarray(np.array([[3, 0], [-3, 0], [0, 4]], dtype=dtype), order=order)
    left[0, 2] = 0
    right[0, 1] = 0
    if uncompressed:
        left.reserve(np.full(left.outer_size, 4, dtype=np.int32))
        right.reserve(np.full(right.outer_size, 4, dtype=np.int32))
    expected_left, expected_right = left.toarray(), right.toarray()
    result = left @ right
    np.testing.assert_array_equal(result.toarray(), expected_left @ expected_right)
    assert result.nnz == 3  # Cancellation and an explicitly zero product survive.
    assert result.is_compressed
    values, indices, indptr = result.compressed_arrays()
    for begin, end in zip(indptr[:-1], indptr[1:]):
        assert np.all(np.diff(indices[begin:end]) > 0)
    result.fill(9)
    np.testing.assert_array_equal(left.toarray(), expected_left)
    np.testing.assert_array_equal(right.toarray(), expected_right)


@pytest.mark.parametrize(
    "dtype", [np.dtype(t) for t in (np.int8, np.int16, np.int32, np.int64)]
)
def test_sparse_product_integer_precision_and_overflow(dtype, order):
    maximum = np.iinfo(dtype).max
    left = np.array([[maximum, maximum], [maximum, 2]], dtype=dtype)
    right = np.array([[maximum, 2], [2, maximum]], dtype=dtype)
    result = sppy.asarray(left, order=order) @ sppy.asarray(right, order=order)
    np.testing.assert_array_equal(result.toarray(), left @ right)


def test_sparse_product_complex_values_are_not_conjugated(order):
    left = np.array([[1 + 2j, 3j], [0, -2j]])
    right = np.array([[3j, 0], [1 - 1j, 2j]])
    result = sppy.asarray(left, order=order) @ sppy.asarray(right, order=order)
    np.testing.assert_array_equal(result.toarray(), left @ right)


def test_sparse_product_huge_inner_axis_uses_sparse_workspace(order):
    n = 2**31 - 1
    if order == "C":
        left = sppy.asarray([[2.0, 2.0]], order=order)
        right = sppy.zeros((2, n), order=order)
        right[0, n - 1] = 3
        right[1, 1] = 4
    else:
        left = sppy.zeros((n, 2), order=order)
        left[n - 1, 0] = 3
        left[1, 1] = 4
        right = sppy.asarray([[2.0], [2.0]], order=order)
    result = left @ right
    assert result.shape == ((1, n) if order == "C" else (n, 1))
    assert result.nnz == 2
    position = (0, n - 1) if order == "C" else (n - 1, 0)
    assert result[position] == 6
    _, indices, _ = result.compressed_arrays()
    np.testing.assert_array_equal(indices, [1, n - 1])


@pytest.mark.parametrize("shape,columns", [((0, 3), 4), ((3, 0), 4), ((3, 2), 0)])
def test_sparse_product_empty_dimensions(shape, columns, dtype, order):
    left = sppy.zeros(shape, dtype=dtype, order=order)
    right = sppy.zeros((shape[1], columns), dtype=dtype, order=order)
    result = left @ right
    assert result.shape == (shape[0], columns)
    assert result.nnz == 0
    np.testing.assert_array_equal(result.toarray(), np.zeros(result.shape, dtype=dtype))
