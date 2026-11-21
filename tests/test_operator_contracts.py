"""Shape, fallback, laziness, and numerical contracts for callback-based operators."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.sparse.linalg import aslinearoperator

import sppy
from sppy.linalg import GeneralLinearOperator, rsvd


@pytest.mark.parametrize("shape", [(4, 3), (0, 3), (3, 0), (0, 0)])
@pytest.mark.parametrize("column", [False, True])
def test_vector_callbacks_supply_matrix_fallbacks(
    shape: tuple[int, int], column: bool
) -> None:
    """Forward and reverse matrix products must work from vector callbacks alone."""
    dense = np.arange(np.prod(shape), dtype=float).reshape(shape).astype(complex) * (
        1 + 2j
    )

    def forward(v: NDArray[Any]) -> NDArray[Any]:
        """Return either a flat vector or column vector from the callback."""
        result = dense @ v
        return result[:, None] if column else result

    operator = GeneralLinearOperator(
        shape, forward, lambda v: dense.conj().T @ v, dtype=dense.dtype
    )
    for count in (0, 1, 3):
        rhs = np.ones((shape[1], count))
        reverse_rhs = np.ones((shape[0], count))
        np.testing.assert_array_equal(operator.matmat(rhs), dense @ rhs)
        np.testing.assert_array_equal(
            operator.rmatmat(reverse_rhs), dense.conj().T @ reverse_rhs
        )
    np.testing.assert_array_equal(
        aslinearoperator(operator).matmat(np.ones((shape[1], 2))),
        dense @ np.ones((shape[1], 2)),
    )
    assert operator.matvec(np.ones((shape[1], 1))).shape == (shape[0], 1)


@pytest.mark.parametrize("operation", ["rmatvec", "rmatmat", "add"])
def test_missing_operations_report_unsupported_capabilities(operation: str) -> None:
    """Missing reverse products and addition must raise explicit capability errors."""
    operator = GeneralLinearOperator((2, 2), lambda v: v, dtype=float)
    with pytest.raises(NotImplementedError):
        getattr(operator, operation)(
            np.ones((2, 1)) if operation.endswith("mat") else np.ones(2)
        )


@pytest.mark.parametrize("shape", [(2,), (2, 3, 4), (-1, 2), (2.5, 3), (True, 3)])
def test_operator_rejects_invalid_shape(shape: Any) -> None:
    """An operator must have two nonnegative integer dimensions."""
    with pytest.raises(ValueError):
        GeneralLinearOperator(shape, lambda v: v)


@pytest.mark.parametrize(
    "operation,operand",
    [
        ("matvec", np.ones(2)),
        ("rmatvec", np.ones(3)),
        ("matmat", np.ones((2, 2))),
        ("rmatmat", np.ones((3, 2))),
    ],
)
def test_operator_rejects_invalid_callback_results(
    operation: str, operand: NDArray[Any]
) -> None:
    """Callbacks must return exactly the dimensions of the represented product."""
    operator = GeneralLinearOperator(
        (3, 2),
        lambda v: np.ones((4, 2)),
        lambda v: np.ones((4, 2)),
        lambda v: np.ones((4, 2)),
        lambda v: np.ones((4, 2)),
    )
    with pytest.raises(ValueError, match="callback"):
        getattr(operator, operation)(operand)


def test_operator_validates_input_before_evaluating_callbacks() -> None:
    """Construction, empty matrix fallbacks, and invalid operands need no callback evaluation."""

    def unexpected(v: NDArray[Any]) -> NDArray[Any]:
        """Reject accidental callback evaluation."""
        raise AssertionError("Callback should not run")

    operator = GeneralLinearOperator((3, 2), unexpected, unexpected, dtype=np.complex64)
    assert operator.matmat(np.empty((2, 0), dtype=np.float32)).dtype == np.dtype(
        np.complex64
    )
    for operand in (np.ones(3), np.ones((2, 2)), np.ones((1, 2)), np.ones((2, 1, 1))):
        with pytest.raises(ValueError, match="operand"):
            operator.matvec(operand)
    with pytest.raises(ValueError, match="operand"):
        operator.matmat(np.ones((3, 2)))


def test_reverse_matrix_callback_supplies_reverse_vector() -> None:
    """A supplied reverse matrix callback must also support vector products."""
    dense = np.arange(6).reshape(3, 2)
    operator = GeneralLinearOperator(
        dense.shape, lambda v: dense @ v, rmatmat=lambda v: dense.T @ v
    )
    np.testing.assert_array_equal(operator.rmatvec(np.ones(3)), dense.T @ np.ones(3))


@pytest.mark.parametrize("storage", ["dense", "sparse"])
def test_operator_observes_source_mutations(storage: str) -> None:
    """Reverse products must observe mutations rather than reuse a stale adjoint."""
    dense = np.eye(3, dtype=complex)
    source = dense if storage == "dense" else sppy.asarray(dense)
    operator = GeneralLinearOperator.as_linear_operator(source)
    operator.rmatvec(np.ones(3))
    source[0, 1] = 2j
    expected = np.asarray(source).conj().T @ np.ones(3)
    np.testing.assert_array_equal(operator.rmatvec(np.ones(3)), expected)
    assert GeneralLinearOperator.as_linear_operator(operator) is operator


@pytest.mark.parametrize("shape", [(12, 6), (6, 12)])
@pytest.mark.parametrize("power", [0, 2])
def test_randomized_svd_accepts_vector_only_operators(
    shape: tuple[int, int], power: int
) -> None:
    """SVD must recover a low-rank complex matrix through vector-only callbacks."""
    rng = np.random.default_rng(21)
    left = rng.standard_normal((shape[0], 2)) + 1j * rng.standard_normal((shape[0], 2))
    right = rng.standard_normal((2, shape[1])) + 1j * rng.standard_normal((2, shape[1]))
    dense = left @ right
    operator = GeneralLinearOperator(
        shape, lambda v: dense @ v, lambda v: dense.conj().T @ v, dtype=dense.dtype
    )
    u, s, v = rsvd(operator, 2, p=2, q=power, rng=21)
    np.testing.assert_allclose((u * s) @ v.conj().T, dense, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("value", [2**53, 2**60, 2**63 - 2, -(2**63)])
@pytest.mark.parametrize("order", ["C", "F"])
def test_large_adjacent_integers_compare_without_float_rounding(
    value: int, order: str
) -> None:
    """Adjacent signed int64 values remain distinct at zero tolerance."""
    left = sppy.asarray(np.array([[value, 0]], dtype=np.int64), order=order)
    right = sppy.asarray(np.array([[value + 1, 0]], dtype=np.int64), order=order)
    right.put(0, [0], [1])
    assert not left.is_approx(right, rtol=0, atol=0)
    assert left.is_approx(right, rtol=0, atol=1)
    assert left.is_approx(left, rtol=0, atol=0)
