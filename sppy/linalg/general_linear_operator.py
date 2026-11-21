"""Lazy matrix products and sums with Hermitian reverse operations."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray


class GeneralLinearOperator:
    """
    A more general form of LinearOperator in scipy.sparse.linalg. Can be used
    with many of the scipy functions.

    The new operation is rmatmat which is X.conj().T @ V.
    """

    def __init__(
        self,
        shape: tuple[int, int],
        matvec: Callable[[ArrayLike], NDArray[Any]],
        rmatvec: Callable[[ArrayLike], NDArray[Any]] | None = None,
        matmat: Callable[[ArrayLike], NDArray[Any]] | None = None,
        rmatmat: Callable[[ArrayLike], NDArray[Any]] | None = None,
        add: Callable[[Any], Any] | None = None,
        dtype: DTypeLike | None = None,
    ) -> None:
        """Store matrix metadata and callbacks without evaluating any products."""
        self.shape = shape
        self.matvec = matvec
        self.rmatvec = rmatvec
        self.matmat = matmat
        self.rmatmat = rmatmat
        self.add = add
        self.dtype = dtype

    @staticmethod
    def as_linear_operator(X: Any, parallel: bool = False) -> GeneralLinearOperator:
        """
        Make a general linear operator from SparseArray X.
        """

        def matvec(v: ArrayLike) -> NDArray[Any]:
            """Apply the forward product to a vector or multiple columns."""
            product = (
                X.parallel_dot if parallel and hasattr(X, "parallel_dot") else X.dot
            )
            return product(v)

        def rmatvec(v: ArrayLike) -> NDArray[Any]:
            """Apply the Hermitian reverse product to a vector or multiple columns."""
            adjoint = X.H if hasattr(X, "H") else X.conj().T
            product = (
                adjoint.parallel_dot
                if parallel and hasattr(adjoint, "parallel_dot")
                else adjoint.dot
            )
            return product(v)

        def add(V: Any) -> Any:
            """Add the represented matrix to the operand."""
            return X + V

        return GeneralLinearOperator(
            X.shape, matvec, rmatvec, matvec, rmatvec, add, X.dtype
        )

    @staticmethod
    def as_linear_operator_sum(
        X: Any, Y: GeneralLinearOperator
    ) -> GeneralLinearOperator:
        """
        Take two linear operators X and Y, and operate on their sum, using lazy
        evaluation.
        """
        if X.shape != Y.shape:
            raise ValueError(
                "Shapes of X and Y do not match: " + str(X.shape) + " " + str(Y.shape)
            )

        def matvec(v: ArrayLike) -> NDArray[Any]:
            """Apply the forward matrix product to a vector."""
            return X.matvec(v) + Y.matvec(v)

        def rmatvec(v: ArrayLike) -> NDArray[Any]:
            """Apply the Hermitian reverse product to a vector."""
            return X.rmatvec(v) + Y.rmatvec(v)

        def matmat(V: ArrayLike) -> NDArray[Any]:
            """Apply the forward matrix product to multiple columns."""
            return X.matmat(V) + Y.matmat(V)

        def rmatmat(V: ArrayLike) -> NDArray[Any]:
            """Apply the Hermitian reverse product to multiple columns."""
            return X.rmatmat(V) + Y.rmatmat(V)

        def add(V: ArrayLike) -> Any:
            """Add the represented matrix to the operand."""
            return Y.add(X.add(V))

        return GeneralLinearOperator(
            X.shape,
            matvec,
            rmatvec,
            matmat,
            rmatmat,
            add,
            np.result_type(X.dtype, Y.dtype),
        )
