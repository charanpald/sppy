"""Regression tests for modern NumPy, SciPy, and validation behavior."""

import io

import numpy as np
import pytest
from scipy import sparse

from sppy import csarray
from sppy.io import mmread, mmwrite


@pytest.mark.parametrize(
    "dtype", [np.int8, np.int16, np.int32, np.int64, np.float32, np.float64]
)
@pytest.mark.parametrize("storage", ["row", "col"])
def test_dtype_roundtrip(dtype, storage):
    source = np.array([[0, 2], [-3, 0]], dtype=dtype)
    array = csarray(source, dtype=np.dtype(dtype), storagetype=storage)
    assert array.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(array.toarray(), source)
    np.testing.assert_allclose(array.dot(np.ones(2)), source @ np.ones(2))


def test_large_integer_vector():
    array = csarray((2,), dtype=np.int64)
    array[1] = 2**60 + 1
    assert array[1] == 2**60 + 1
    assert array.toarray().dtype == np.dtype(np.int64)
    assert array.toarray()[1] == 2**60 + 1


@pytest.mark.parametrize("shape", [(-1, 2), (2**31, 2), (1.5, 2)])
def test_invalid_shape(shape):
    with pytest.raises(ValueError):
        csarray(shape)


def test_scipy_sparse_array():
    source = sparse.csr_array([[0.0, 2.0], [3.0, 0.0]])
    np.testing.assert_array_equal(csarray(source).toarray(), source.toarray())


def test_matrix_market_symmetric():
    source = io.BytesIO(
        b"%%MatrixMarket matrix coordinate real symmetric\n% comment\n2 2 2\n1 1 2\n2 1 3\n"
    )
    np.testing.assert_array_equal(mmread(source).toarray(), [[2, 3], [3, 0]])


def test_matrix_market_roundtrip():
    source = csarray(np.array([[0.0, np.pi], [1e-15, 0.0]]))
    stream = io.BytesIO()
    mmwrite(stream, source)
    stream.seek(0)
    np.testing.assert_array_equal(mmread(stream).toarray(), source.toarray())


def test_paired_indices_validate_lengths():
    array = csarray((2, 2))
    with pytest.raises(ValueError):
        array[np.array([0, 1]), np.array([0, 1, 0])]


@pytest.mark.parametrize("init", [False, True])
def test_put_validates_inputs(init):
    array = csarray((2, 2))
    with pytest.raises(ValueError):
        array.put(
            np.ones(1),
            np.array([0, 1], dtype=np.int32),
            np.array([0, 1], dtype=np.int32),
            init=init,
        )
    with pytest.raises(ValueError):
        array.put(
            1.0,
            np.array([-1], dtype=np.int32),
            np.array([0], dtype=np.int32),
            init=init,
        )
    array.put(
        np.array([]),
        np.array([], dtype=np.int32),
        np.array([], dtype=np.int32),
        init=init,
    )
    assert array.nnz == 0


def test_missing_attribute_raises():
    array = csarray((2, 2))
    with pytest.raises(AttributeError):
        getattr(array, "missing_attribute")


def test_integer_solver_rejected():
    from sppy.linalg import biCGSTAB

    array = csarray(np.eye(2, dtype=np.int64), dtype=np.int64)
    with pytest.raises(ValueError):
        biCGSTAB(array, np.ones(2, dtype=np.int64))


@pytest.mark.parametrize("order", ["C", "F"])
def test_numpy_operators_and_names(order):
    import sppy

    dense = np.array([[0, 2], [3, 0]], dtype=np.int64)
    array = sppy.asarray(dense, order=order)
    assert isinstance(array, sppy.SparseArray)
    assert sppy.csarray is sppy.SparseArray
    assert array.order == order
    np.testing.assert_array_equal(np.asarray(array), dense)
    np.testing.assert_array_equal((array @ array).toarray(), dense @ dense)
    np.testing.assert_array_equal((array * array).toarray(), dense * dense)
    np.testing.assert_array_equal((0.5 * array).toarray(), 0.5 * dense)
    np.testing.assert_array_equal(
        array @ np.ones(2, dtype=np.int64), dense @ np.ones(2, dtype=np.int64)
    )
    np.testing.assert_array_equal(np.ones(2) @ array, np.ones(2) @ dense)
    np.testing.assert_array_equal(array.to_scipy_csr().toarray(), dense)
    np.testing.assert_array_equal(array.to_scipy_csc().toarray(), dense)
    assert sppy.asarray(array, copy=False) is array
    assert array.astype(array.dtype, copy=False) is array
    assert array.astype(np.float32).dtype == np.dtype(np.float32)
    with pytest.raises(ValueError):
        np.asarray(array, copy=False)
    with pytest.raises(TypeError):
        array.astype(np.int8, casting="safe")


@pytest.mark.parametrize("operation", ["sum", "mean", "var", "std", "min", "max"])
@pytest.mark.parametrize("axis", [None, 0, 1, -1])
@pytest.mark.parametrize("keepdims", [False, True])
def test_numpy_reductions(operation, axis, keepdims):
    import sppy

    dense = np.array([[0.0, 2.0, -3.0], [1.0, 0.0, 0.0]])
    array = sppy.asarray(dense)
    expected = getattr(np, operation)(dense, axis=axis, keepdims=keepdims)
    result = getattr(np, operation)(array, axis=axis, keepdims=keepdims)
    np.testing.assert_allclose(result, expected)
    output = np.empty_like(expected)
    assert getattr(array, operation)(axis=axis, keepdims=keepdims, out=output) is output
    np.testing.assert_allclose(output, expected)


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
    ],
)
def test_numpy_indexing(key):
    import sppy

    dense = np.array([[0.0, 2.0, -3.0], [1.0, 0.0, 0.0]])
    array = sppy.asarray(dense)
    np.testing.assert_array_equal(np.asarray(array[key]), dense[key])


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
    import sppy

    dense = np.zeros((2, 3))
    array = sppy.zeros(dense.shape)
    dense[key] = value
    array[key] = value
    np.testing.assert_array_equal(array.toarray(), dense)


def test_factories_and_exact_integer_product():
    import sppy

    np.testing.assert_array_equal(sppy.eye(3, 5, k=1).toarray(), np.eye(3, 5, k=1))
    np.testing.assert_array_equal(
        sppy.diag([1, 2], k=-1).toarray(), np.diag([1, 2], k=-1)
    )
    dense = np.array([[2**60 + 1, 0], [0, -3]], dtype=np.int64)
    array = sppy.asarray(dense)
    np.testing.assert_array_equal(
        array @ np.ones(2, dtype=np.int64), dense @ np.ones(2, dtype=np.int64)
    )
    np.testing.assert_array_equal(array.diagonal(), dense.diagonal())
    np.testing.assert_array_equal(
        (array + array.astype(np.float64, order="F")).toarray(),
        dense + dense.astype(float),
    )
    assert sppy.zeros((2, 3)).order == "C"
    assert sppy.zeros((2, 3), order="F").order == "F"


@pytest.mark.parametrize("function", ["ceil", "floor", "sin", "cos", "sign"])
@pytest.mark.parametrize("dtype", [np.int64, np.float32, np.float64])
def test_numpy_unary(function, dtype):
    import sppy

    dense = np.array([[0, 2.3], [-1.2, 0]], dtype=dtype)
    expected = getattr(np, function)(dense)
    result = getattr(sppy.asarray(dense), function)()
    assert result.dtype == expected.dtype
    np.testing.assert_allclose(result.toarray(), expected)


def test_numpy_clip_and_trace():
    import sppy

    dense = np.array([[0.0, 2.3], [-1.2, 0.0]])
    array = sppy.asarray(dense)
    np.testing.assert_array_equal(array.clip(1, 2).toarray(), np.clip(dense, 1, 2))
    assert np.trace(array, offset=1) == np.trace(dense, offset=1)
    array[0, 0] = 0
    array.eliminate_zeros()
    assert array.nnz == array.count_nonzero() == np.count_nonzero(dense)


def test_snake_case_linear_operator():
    import sppy
    from sppy.linalg import GeneralLinearOperator, bicgstab

    array = sppy.eye(3)
    operator = GeneralLinearOperator.as_linear_operator(array)
    np.testing.assert_allclose(operator.matvec(np.ones(3)), np.ones(3))
    result, status = bicgstab(array, np.ones(3))
    assert status == 0
    np.testing.assert_allclose(result, np.ones(3))
