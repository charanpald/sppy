"""Array factories compared with NumPy and deterministic RNG expectations."""

import numpy as np
import pytest

import sppy


@pytest.mark.parametrize("factory", ["ones", "zeros"])
@pytest.mark.parametrize("shape", [(3, 4), (5,), (0,), (0, 0)])
def test_factory_values(factory, shape, dtype, order):
    # Arrange
    expected = getattr(np, factory)(shape, dtype=dtype)

    # Act
    result = getattr(sppy, factory)(shape, dtype=dtype, order=order)

    # Assert
    assert result.order == order
    assert result.dtype == dtype
    np.testing.assert_array_equal(result.toarray(), expected)


@pytest.mark.parametrize(
    "shape,offset", [((3, 3), 0), ((3, 5), 1), ((5, 3), -1), ((2, 3), 10)]
)
def test_eye(shape, offset, dtype, order):
    # Arrange
    expected = np.eye(*shape, k=offset, dtype=dtype)

    # Act
    result = sppy.eye(*shape, k=offset, dtype=dtype, order=order)

    # Assert
    assert result.dtype == dtype
    np.testing.assert_array_equal(result.toarray(), expected)


@pytest.mark.parametrize("offset", [-2, 0, 1])
def test_diag_construction(offset, dtype):
    # Arrange
    vector = np.array([1, 2, -3], dtype=dtype)
    expected = np.diag(vector, k=offset)

    # Act
    result = sppy.diag(vector, k=offset)

    # Assert
    np.testing.assert_array_equal(result.toarray(), expected)


@pytest.mark.parametrize("shape", [(4, 5), (10,), (0,), (0, 0)])
@pytest.mark.parametrize("density", [0.0, 0.25, 1.0])
def test_rand_is_reproducible(shape, density, order):
    # Arrange
    left_rng = np.random.default_rng(21)
    right_rng = np.random.default_rng(21)

    # Act
    first = sppy.rand(shape, density, order=order, rng=left_rng)
    second = sppy.rand(shape, density, order=order, rng=right_rng)

    # Assert
    assert first.shape == shape
    assert first.order == order
    np.testing.assert_array_equal(first.toarray(), second.toarray())
    assert first.nnz <= int(np.prod(shape) * density)


@pytest.mark.parametrize("density", [-0.1, 1.1])
def test_rand_rejects_invalid_density(density):
    # Arrange
    shape = (2, 3)

    # Act / Assert
    with pytest.raises(ValueError):
        sppy.rand(shape, density)


def test_legacy_constructor_alias():
    # Arrange
    constructor = sppy.csarray

    # Act
    result = constructor((2, 3))

    # Assert
    assert constructor is sppy.SparseArray
    assert isinstance(result, sppy.SparseArray)


@pytest.mark.parametrize("factory", ["ones", "zeros"])
def test_legacy_storage_keyword(factory):
    # Arrange
    constructor = getattr(sppy, factory)

    # Act
    result = constructor((2, 3), storageType="col")

    # Assert
    assert result.order == "F"


def test_legacy_eye_dtype_argument():
    # Arrange
    expected = np.eye(3, dtype=np.int64)

    # Act
    result = sppy.eye(3, np.int64)

    # Assert
    assert result.dtype == expected.dtype
    np.testing.assert_array_equal(result.toarray(), expected)
