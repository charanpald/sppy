"""Deterministic solver and decomposition tests with explicit numerical checks."""

import numpy as np
import pytest
from scipy.sparse.linalg import eigsh

import sppy
from sppy.linalg import GeneralLinearOperator, biCGSTAB, bicgstab, norm, rsvd


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("operation", ["matvec", "rmatvec", "matmat", "rmatmat"])
def test_linear_operator_products(operation, parallel, rng):
    # Arrange
    dense = rng.normal(size=(8, 5))
    array = sppy.asarray(dense, order="C")
    operator = GeneralLinearOperator.as_linear_operator(array, parallel=parallel)
    transposed = operation.startswith("r")
    operand_shape = (
        ((8 if transposed else 5), 3)
        if operation.endswith("mat")
        else ((8 if transposed else 5),)
    )
    operand = rng.normal(size=operand_shape)
    expected = (dense.T if transposed else dense) @ operand

    # Act
    result = getattr(operator, operation)(operand)

    # Assert
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


def test_linear_operator_supports_scipy_eigensolver(rng):
    # Arrange
    dense = rng.normal(size=(15, 15))
    dense = dense + dense.T
    array = sppy.asarray(dense)
    operator = GeneralLinearOperator.as_linear_operator(array)
    expected = np.linalg.eigvalsh(dense)[-5:]
    initial_vector = rng.normal(size=15)

    # Act
    values, vectors = eigsh(operator, k=5, which="LA", v0=initial_vector)

    # Assert
    np.testing.assert_allclose(values, expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        dense @ vectors, vectors * values, rtol=1e-10, atol=1e-10
    )


@pytest.mark.parametrize("operation", ["matvec", "rmatvec", "matmat", "rmatmat"])
def test_linear_operator_sum(operation, rng):
    # Arrange
    left = rng.normal(size=(8, 5))
    right = rng.normal(size=(8, 5))
    operator = GeneralLinearOperator.as_linear_operator_sum(
        GeneralLinearOperator.as_linear_operator(sppy.asarray(left)),
        GeneralLinearOperator.as_linear_operator(sppy.asarray(right)),
    )
    transposed = operation.startswith("r")
    operand_shape = (
        ((8 if transposed else 5), 3)
        if operation.endswith("mat")
        else ((8 if transposed else 5),)
    )
    operand = rng.normal(size=operand_shape)
    expected = ((left + right).T if transposed else left + right) @ operand

    # Act
    result = getattr(operator, operation)(operand)

    # Assert
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


def test_linear_operator_sum_rejects_shape_mismatch():
    # Arrange
    left = GeneralLinearOperator.as_linear_operator(sppy.zeros((2, 3)))
    right = GeneralLinearOperator.as_linear_operator(sppy.zeros((3, 2)))

    # Act / Assert
    with pytest.raises(ValueError, match="Shapes"):
        GeneralLinearOperator.as_linear_operator_sum(left, right)


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_bicgstab_solves_known_system(legacy, dtype, order):
    # Arrange
    dense = np.diag(np.array([2, 3, 4, 5], dtype=dtype))
    array = sppy.asarray(dense, order=order)
    expected = np.array([1, -2, 3, -4], dtype=dtype)
    rhs = dense @ expected
    solve = biCGSTAB if legacy else bicgstab
    options = (
        {"maxIter": 100, "tol": 1e-6} if legacy else {"maxiter": 100, "rtol": 1e-6}
    )

    # Act
    result, status = solve(array, rhs, **options)

    # Assert
    assert status == 0
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(dense @ result, rhs, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("shape,rhs", [((3, 2), np.ones(3)), ((3, 3), np.ones(4))])
def test_bicgstab_rejects_invalid_dimensions(shape, rhs):
    # Arrange
    array = sppy.ones(shape)

    # Act / Assert
    with pytest.raises(ValueError):
        bicgstab(array, rhs)


@pytest.mark.parametrize("options", [{"maxiter": 0}, {"rtol": 0}, {"maxiter": -1}])
def test_bicgstab_rejects_invalid_options(options):
    # Arrange
    array = sppy.eye(3)
    rhs = np.ones(3)

    # Act / Assert
    with pytest.raises(ValueError):
        bicgstab(array, rhs, **options)


@pytest.mark.parametrize("solver", [biCGSTAB, bicgstab])
def test_integer_solver_rejected(solver):
    # Arrange
    array = sppy.eye(2, dtype=np.int64)
    rhs = np.ones(2, dtype=np.int64)

    # Act / Assert
    with pytest.raises(ValueError):
        solver(array, rhs)


@pytest.mark.parametrize("storage", ["dense", "sparse", "operator"])
@pytest.mark.parametrize("rank", [3, 6, 10])
def test_randomized_svd_is_orthonormal_and_bounded(storage, rank, rng):
    # Arrange
    dense = rng.normal(size=(24, 20))
    array = sppy.asarray(dense)
    source = {
        "dense": dense,
        "sparse": array,
        "operator": GeneralLinearOperator.as_linear_operator(array),
    }[storage]
    omega = rng.normal(size=(20, rank + 4))
    optimal_values = np.linalg.svd(dense, compute_uv=False)
    optimal_error = np.linalg.norm(optimal_values[rank:])

    # Act
    left, values, right = rsvd(source, rank, p=4, q=2, omega=omega)

    # Assert
    assert values.shape == (rank,)
    np.testing.assert_allclose(left.T @ left, np.eye(rank), atol=1e-12)
    np.testing.assert_allclose(right.T @ right, np.eye(rank), atol=1e-12)
    error = np.linalg.norm(dense - (left * values) @ right.T)
    assert optimal_error - 1e-10 <= error <= np.linalg.norm(dense)


def test_randomized_svd_uses_warm_start(rng):
    # Arrange
    dense = rng.normal(size=(18, 12))
    rank = 4
    _, exact_values, exact_right = np.linalg.svd(dense, full_matrices=False)
    omega = exact_right[:rank].T
    optimal_error = np.linalg.norm(exact_values[rank:])

    # Act
    left, values, right = rsvd(dense, rank, p=0, q=0, omega=omega)

    # Assert
    np.testing.assert_allclose(values, exact_values[:rank], atol=1e-12)
    np.testing.assert_allclose(
        np.linalg.norm(dense - (left * values) @ right.T), optimal_error
    )


@pytest.mark.parametrize("options", [{"k": 0}, {"k": 2, "p": -1}, {"k": 2, "q": -1}])
def test_randomized_svd_rejects_invalid_options(options):
    # Arrange
    dense = np.eye(4)

    # Act / Assert
    with pytest.raises(ValueError):
        rsvd(dense, **options)


def test_sparse_norm_matches_numpy(order, rng):
    # Arrange
    dense = rng.normal(size=(8, 5))
    array = sppy.asarray(dense, order=order)
    expected = np.linalg.norm(dense)

    # Act
    result = norm(array)

    # Assert
    assert result == pytest.approx(expected)


def test_legacy_linear_operator_alias():
    # Arrange
    array = sppy.eye(3)
    vector = np.ones(3)
    constructor = GeneralLinearOperator.asLinearOperator

    # Act
    operator = constructor(array)
    result = operator.matvec(vector)

    # Assert
    np.testing.assert_array_equal(result, vector)
