"""NumPy interface regressions; each scenario has a single explicit action."""

import numpy as np
import pytest

import sppy


@pytest.mark.parametrize("operation", ["sum", "mean", "var", "std", "min", "max"])
@pytest.mark.parametrize("axis", [None, 0, 1, -1])
@pytest.mark.parametrize("keepdims", [False, True])
@pytest.mark.parametrize("use_out", [False, True])
def test_numpy_reductions(operation, axis, keepdims, use_out):
    # Arrange
    dense = np.array([[0.0, 2.0, -3.0], [1.0, 0.0, 0.0]])
    array = sppy.asarray(dense)
    expected = getattr(np, operation)(dense, axis=axis, keepdims=keepdims)
    output = np.empty_like(expected) if use_out else None

    # Act
    result = getattr(np, operation)(array, axis=axis, keepdims=keepdims, out=output)

    # Assert
    np.testing.assert_allclose(result, expected)
    if use_out:
        assert result is output


@pytest.mark.parametrize("operation", ["sum", "mean", "var", "std"])
def test_reduction_dtype(operation):
    # Arrange
    dense = np.array([[0, 2], [-3, 0]], dtype=np.int8)
    array = sppy.asarray(dense)
    expected = getattr(np, operation)(dense, dtype=np.float32)

    # Act
    result = getattr(array, operation)(dtype=np.float32)

    # Assert
    assert result.dtype == expected.dtype
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("operation", ["var", "std"])
@pytest.mark.parametrize("axis", [None, 0, 1])
def test_variance_degrees_of_freedom(operation, axis):
    # Arrange
    dense = np.array([[0.0, 2.0, -3.0], [1.0, 0.0, 0.0]])
    array = sppy.asarray(dense)
    expected = getattr(np, operation)(dense, axis=axis, ddof=1)

    # Act
    result = getattr(array, operation)(axis=axis, ddof=1)

    # Assert
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    "key",
    [
        1,
        (slice(None), 1),
        (slice(None, None, -1), slice(None, None, 2)),
        (slice(-2, None), slice(1, 3)),
        ([1, 0, 1], slice(None)),
        ([True, False], slice(None)),
        (np.array([[0], [1]]), np.array([0, 2])),
        Ellipsis,
        (slice(0, 0), slice(None)),
    ],
)
def test_numpy_indexing(key):
    # Arrange
    dense = np.array([[0.0, 2.0, -3.0], [1.0, 0.0, 0.0]])
    array = sppy.asarray(dense)
    expected = dense[key]

    # Act
    result = array[key]

    # Assert
    np.testing.assert_array_equal(np.asarray(result), expected)


@pytest.mark.parametrize(
    "key,value",
    [
        ((slice(None), 1), [4, 5]),
        ((slice(None, None, -1), slice(None, None, 2)), 3),
        ((-1, -1), 7),
        (([0, 1], [1, 2]), [8, 9]),
    ],
)
def test_numpy_assignment(key, value):
    # Arrange
    expected = np.zeros((2, 3))
    expected[key] = value
    array = sppy.zeros(expected.shape)

    # Act
    array[key] = value

    # Assert
    np.testing.assert_array_equal(array.toarray(), expected)


@pytest.mark.parametrize(
    "key",
    [
        (20, 1),
        (1, 20),
        (-3, 1),
        (1, -4),
        ("a", "b"),
        (np.array([0, 1]), np.array([0, 1, 0])),
        (np.array([0, 2]), np.array([1, 1])),
    ],
)
def test_invalid_index_rejected(key):
    # Arrange
    array = sppy.zeros((2, 3))

    # Act / Assert
    with pytest.raises(ValueError):
        array[key]


@pytest.mark.parametrize("function", ["ceil", "floor", "sin", "cos", "sign"])
@pytest.mark.parametrize("dtype", [np.int64, np.float32, np.float64])
def test_numpy_unary(function, dtype, order):
    # Arrange
    dense = np.array([[0, 2.3], [-1.2, 0]], dtype=dtype)
    array = sppy.asarray(dense, order=order)
    expected = getattr(np, function)(dense)

    # Act
    result = getattr(array, function)()

    # Assert
    assert result.dtype == expected.dtype
    np.testing.assert_allclose(result.toarray(), expected)


@pytest.mark.parametrize("bounds", [(0, 2), (1, 2), (-2, None), (None, 1)])
def test_numpy_clip(bounds):
    # Arrange
    dense = np.array([[0.0, 2.3], [-1.2, 0.0]])
    array = sppy.asarray(dense)
    expected = np.clip(dense, *bounds)

    # Act
    result = array.clip(*bounds)

    # Assert
    np.testing.assert_array_equal(result.toarray(), expected)


@pytest.mark.parametrize("offset", [-1, 0, 1, 10])
def test_numpy_trace_and_diagonal(offset):
    # Arrange
    dense = np.arange(12.0).reshape(3, 4)
    array = sppy.asarray(dense)
    expected_diagonal = np.diagonal(dense, offset=offset)
    expected_trace = np.trace(dense, offset=offset)

    # Act
    diagonal = array.diagonal(offset)
    trace = np.trace(array, offset=offset)

    # Assert
    np.testing.assert_array_equal(diagonal, expected_diagonal)
    assert trace == expected_trace


@pytest.mark.parametrize("shape", [(2,), (2, 2)])
def test_large_integer_roundtrip_and_product(shape):
    # Arrange
    dense = np.zeros(shape, dtype=np.int64)
    dense.flat[-1] = 2**60 + 1
    array = sppy.asarray(dense)
    vector = np.ones(shape[-1], dtype=np.int64)
    expected_product = dense @ vector

    # Act
    result = array.toarray()
    product = array @ vector

    # Assert
    assert result.dtype == np.dtype(np.int64)
    np.testing.assert_array_equal(result, dense)
    np.testing.assert_array_equal(product, expected_product)


@pytest.mark.parametrize("method", ["asarray", "astype"])
def test_copy_false_reuses_compatible_sparse_input(method):
    # Arrange
    array = sppy.eye(2)

    # Act
    result = (
        sppy.asarray(array, copy=False)
        if method == "asarray"
        else array.astype(array.dtype, copy=False)
    )

    # Assert
    assert result is array


def test_dense_conversion_preserves_dtype(dtype):
    # Arrange
    expected = np.array([[0, 2], [-3, 0]], dtype=dtype)
    array = sppy.asarray(expected)

    # Act
    result = np.asarray(array)

    # Assert
    assert result.dtype == dtype
    np.testing.assert_array_equal(result, expected)


def test_dense_conversion_rejects_copy_false():
    # Arrange
    array = sppy.eye(2)

    # Act / Assert
    with pytest.raises(ValueError):
        np.asarray(array, copy=False)


def test_astype_rejects_unsafe_cast():
    # Arrange
    array = sppy.asarray(np.array([[0, 2], [-3, 0]], dtype=np.int64))

    # Act / Assert
    with pytest.raises(TypeError):
        array.astype(np.int8, casting="safe")


def test_astype_changes_dtype_and_order(order):
    # Arrange
    dense = np.array([[0, 2], [-3, 0]], dtype=np.int64)
    array = sppy.asarray(dense)
    expected = dense.astype(np.float32)

    # Act
    result = array.astype(np.float32, order=order)

    # Assert
    assert result.dtype == expected.dtype
    assert result.order == order
    np.testing.assert_array_equal(result.toarray(), expected)


def test_missing_attribute_raises():
    # Arrange
    array = sppy.zeros((2, 2))

    # Act / Assert
    with pytest.raises(AttributeError):
        getattr(array, "missing_attribute")


def test_eliminate_zeros_updates_stored_count():
    # Arrange
    array = sppy.asarray([[1.0, 2.0], [0.0, 3.0]])
    array[0, 1] = 0
    expected = np.diag([1.0, 3.0])

    # Act
    array.eliminate_zeros()

    # Assert
    assert array.nnz == array.count_nonzero() == 2
    np.testing.assert_array_equal(array.toarray(), expected)


@pytest.mark.parametrize("init", [False, True])
@pytest.mark.parametrize(
    "values,rows,columns",
    [
        (np.ones(1), [0, 1], [0, 1]),
        (1.0, [-1], [0]),
        (1.0, [0], [2]),
        (1.0, [0], [0, 1]),
    ],
)
def test_put_rejects_invalid_inputs(init, values, rows, columns):
    # Arrange
    array = sppy.zeros((2, 2))
    rows = np.array(rows, dtype=np.int32)
    columns = np.array(columns, dtype=np.int32)

    # Act / Assert
    with pytest.raises(ValueError):
        array.put(values, rows, columns, init=init)


@pytest.mark.parametrize("init", [False, True])
def test_put_accepts_empty_inputs(init):
    # Arrange
    array = sppy.zeros((2, 2))
    indices = np.array([], dtype=np.int32)
    values = np.array([])

    # Act
    array.put(values, indices, indices, init=init)

    # Assert
    assert array.nnz == 0


@pytest.mark.parametrize("shape", [(0,), (0, 0)])
@pytest.mark.parametrize("operation", ["min", "max", "mean", "var", "std"])
def test_empty_reductions_retain_nan_contract(shape, operation):
    # Arrange
    array = sppy.zeros(shape)

    # Act
    result = getattr(array, operation)()

    # Assert
    assert np.isnan(result)


@pytest.mark.parametrize(
    "key",
    [
        0,
        -1,
        slice(None, None, -1),
        slice(1, 4, 2),
        np.array([0, 3, 4]),
        np.array([True, False, False, True, False]),
    ],
)
def test_vector_indexing(key):
    # Arrange
    dense = np.array([2.0, 0.0, -3.0, 4.0, 0.0])
    array = sppy.asarray(dense)
    expected = dense[key]

    # Act
    result = array[key]

    # Assert
    np.testing.assert_array_equal(np.asarray(result), expected)


@pytest.mark.parametrize(
    "key,value", [(-1, 2.0), (slice(1, 4), [1, 2, 3]), ([0, 3], -2)]
)
def test_vector_assignment(key, value):
    # Arrange
    expected = np.zeros(5)
    expected[key] = value
    array = sppy.zeros(5)

    # Act
    array[key] = value

    # Assert
    np.testing.assert_array_equal(array.toarray(), expected)


@pytest.mark.parametrize("operation", ["sum", "mean", "var", "std", "min", "max"])
@pytest.mark.parametrize(
    "values", [[0, 2, 3], [0, -2, -3], [1, 2, 3], [-1, -2, -3], [0, 0, 0]]
)
def test_vector_reductions_include_implicit_zeros(operation, values):
    # Arrange
    dense = np.array(values, dtype=np.float64)
    array = sppy.asarray(dense)
    expected = getattr(np, operation)(dense)

    # Act
    result = getattr(array, operation)()

    # Assert
    np.testing.assert_allclose(result, expected)
