"""Validated lazy matrix products and sums with Hermitian reverse operations."""

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray

Product = Callable[[NDArray[Any]], ArrayLike]


class GeneralLinearOperator:
    """Represent a matrix through forward and optional Hermitian reverse products.

    Vector callbacks accept a one-dimensional array and may return a vector or
    a column vector. Missing matrix callbacks are evaluated column by column.
    Unsupported adjoint products and addition raise NotImplementedError.
    Callbacks can promote the declared dtype to accommodate their operands.
    """

    def __init__(
        self,
        shape: tuple[int, int],
        matvec: Product,
        rmatvec: Product | None = None,
        matmat: Product | None = None,
        rmatmat: Product | None = None,
        add: Callable[[Any], Any] | None = None,
        dtype: DTypeLike | None = None,
    ) -> None:
        """Validate metadata and store callbacks without evaluating any products."""
        if len(shape) != 2:
            raise ValueError("An operator requires two dimensions")
        try:
            dimensions = tuple(operator.index(n) for n in shape)
        except TypeError as error:
            raise ValueError(
                "Operator dimensions must be nonnegative integers"
            ) from error
        if any(isinstance(n, (bool, np.bool_)) for n in shape) or min(dimensions) < 0:
            raise ValueError("Operator dimensions must be nonnegative integers")
        for name, callback in (
            ("matvec", matvec),
            ("rmatvec", rmatvec),
            ("matmat", matmat),
            ("rmatmat", rmatmat),
            ("add", add),
        ):
            if (callback is None and name == "matvec") or (
                callback is not None and not callable(callback)
            ):
                raise TypeError(f"{name} must be callable")
        self.shape = dimensions[0], dimensions[1]
        self.dtype = None if dtype is None else np.dtype(dtype)
        self._matvec, self._rmatvec = matvec, rmatvec
        self._matmat, self._rmatmat, self._add = matmat, rmatmat, add

    def _vector(self, operand: ArrayLike, *, reverse: bool) -> NDArray[Any]:
        """Validate one vector product and preserve vector/column-vector shape."""
        rows, columns = self.shape[::-1] if reverse else self.shape
        vector = np.asarray(operand)
        if vector.shape not in {(columns,), (columns, 1)}:
            raise ValueError("Vector operand has incompatible dimensions")
        callback = self._rmatvec if reverse else self._matvec
        if callback is None:
            if reverse and self._rmatmat is not None:
                return self.rmatmat(vector.reshape(columns, 1)).reshape(
                    (rows, 1) if vector.ndim == 2 else (rows,)
                )
            raise NotImplementedError("Hermitian reverse products are not available")
        result = np.asarray(callback(vector.reshape(columns)))
        if result.shape not in {(rows,), (rows, 1)}:
            raise ValueError("Vector callback returned an incompatible shape")
        return result.reshape((rows, 1) if vector.ndim == 2 else (rows,))

    def _matrix(self, operand: ArrayLike, *, reverse: bool) -> NDArray[Any]:
        """Validate a matrix product or evaluate its vector callbacks by columns."""
        rows, columns = self.shape[::-1] if reverse else self.shape
        matrix = np.asarray(operand)
        if matrix.ndim != 2 or matrix.shape[0] != columns:
            raise ValueError("Matrix operand has incompatible dimensions")
        callback = self._rmatmat if reverse else self._matmat
        if callback is not None:
            result = np.asarray(callback(matrix))
            if result.shape != (rows, matrix.shape[1]):
                raise ValueError("Matrix callback returned an incompatible shape")
            return result
        if reverse and self._rmatvec is None:
            raise NotImplementedError("Hermitian reverse products are not available")
        if not matrix.shape[1]:
            return np.empty(
                (rows, 0),
                dtype=np.result_type(self.dtype, matrix.dtype)
                if self.dtype is not None
                else matrix.dtype,
            )
        product = self.rmatvec if reverse else self.matvec
        return np.column_stack([product(column) for column in matrix.T])

    def matvec(self, v: ArrayLike) -> NDArray[Any]:
        """Apply the forward product to a vector or a single column."""
        return self._vector(v, reverse=False)

    def rmatvec(self, v: ArrayLike) -> NDArray[Any]:
        """Apply the Hermitian reverse product to a vector or a single column."""
        return self._vector(v, reverse=True)

    def matmat(self, v: ArrayLike) -> NDArray[Any]:
        """Apply the forward product to a matrix, using vector fallbacks if needed."""
        return self._matrix(v, reverse=False)

    def rmatmat(self, v: ArrayLike) -> NDArray[Any]:
        """Apply the Hermitian reverse product to a matrix of right-hand sides."""
        return self._matrix(v, reverse=True)

    def add(self, v: Any) -> Any:
        """Add the represented matrix to an operand if addition is supported."""
        if self._add is None:
            raise NotImplementedError("Matrix addition is not available")
        return self._add(v)

    @staticmethod
    def as_linear_operator(X: Any, parallel: bool = False) -> GeneralLinearOperator:
        """Wrap a dense or sparse matrix, retaining live access to its values."""
        if isinstance(X, GeneralLinearOperator):
            return X

        def forward(v: NDArray[Any]) -> ArrayLike:
            """Apply the matrix's forward product to vectors or matrices."""
            product = (
                X.parallel_dot if parallel and hasattr(X, "parallel_dot") else X.dot
            )
            return product(v)

        def reverse(v: NDArray[Any]) -> ArrayLike:
            """Apply the current adjoint without caching mutable matrix values."""
            adjoint = getattr(X, "H", None)
            if adjoint is None:
                adjoint = X.conj().T
            product = (
                adjoint.parallel_dot
                if parallel and hasattr(adjoint, "parallel_dot")
                else adjoint.dot
            )
            return product(v)

        def add(v: Any) -> Any:
            """Add the current matrix to the operand."""
            return X + v

        return GeneralLinearOperator(
            X.shape, forward, reverse, forward, reverse, add, X.dtype
        )

    @staticmethod
    def as_linear_operator_sum(
        X: GeneralLinearOperator, Y: GeneralLinearOperator
    ) -> GeneralLinearOperator:
        """Lazily sum equal-shaped operators while preserving dtype promotion."""
        if X.shape != Y.shape:
            raise ValueError(f"Shapes of X and Y do not match: {X.shape} {Y.shape}")

        def forward(v: NDArray[Any]) -> ArrayLike:
            """Sum forward vector products."""
            return X.matvec(v) + Y.matvec(v)

        def reverse(v: NDArray[Any]) -> ArrayLike:
            """Sum Hermitian reverse vector products."""
            return X.rmatvec(v) + Y.rmatvec(v)

        def matrices(v: NDArray[Any]) -> ArrayLike:
            """Sum forward matrix products."""
            return X.matmat(v) + Y.matmat(v)

        def reverse_matrices(v: NDArray[Any]) -> ArrayLike:
            """Sum Hermitian reverse matrix products."""
            return X.rmatmat(v) + Y.rmatmat(v)

        def add(v: Any) -> Any:
            """Add both represented matrices to the operand."""
            return Y.add(X.add(v))

        dtype = (
            np.result_type(X.dtype, Y.dtype)
            if X.dtype is not None and Y.dtype is not None
            else None
        )
        return GeneralLinearOperator(
            X.shape, forward, reverse, matrices, reverse_matrices, add, dtype
        )
