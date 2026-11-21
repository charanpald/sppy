"""Sparse norms and randomized singular-value decomposition."""

from __future__ import annotations

import gc
from typing import Any

import numpy
from numpy.typing import ArrayLike, NDArray

from sppy.linalg.general_linear_operator import GeneralLinearOperator
from sppy.sparse_array import SparseArray


def norm(a: SparseArray, ord: int | float | str | None = None) -> float:
    """Vector p-norm or matrix Frobenius/1/infinity norm, without densifying."""
    if a.ndim == 1:
        values = a.values()
        if (
            isinstance(ord, (int, float, numpy.integer, numpy.floating))
            and ord < 0
            and len(values) < a.size
        ):
            # One representative implicit zero suffices for negative norms,
            # while letting NumPy preserve NaNs and complex absolute values.
            values = numpy.append(values, values.dtype.type(0))
        return numpy.linalg.norm(values, ord=ord)
    if ord in (None, "fro"):
        return a._array.norm()
    if a.dtype.kind == "i":
        # Matrix norms accumulate magnitudes in floating point, as NumPy does.
        # Taking abs first would overflow at the minimum signed integer.
        a = a.astype(numpy.float64)
    if ord == 1:
        return abs(a).sum(axis=0).max(initial=0)
    if ord == numpy.inf:
        return abs(a).sum(axis=1).max(initial=0)
    raise ValueError("Supported matrix norms are fro, 1, and infinity")


def rsvd(
    A: SparseArray | NDArray[Any] | GeneralLinearOperator,
    k: int,
    p: int = 10,
    q: int = 2,
    omega: ArrayLike | None = None,
    *,
    rng: numpy.random.Generator | int | None = None,
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """
    Compute the randomised SVD using the algorithm on page 9 of Halko et al.,
    Finding Structure with randomness: stochastic algorithms for constructing
    approximate matrix decompositions, 2009.

    Finds the partial SVD of a sparse or dense matrix A, resolving the largest k
    singular vectors/values, using exponent q and k+p projections. Returns the
    left and right singular vectors, and the singular values. The resulting matrix
    can be approximated using A ~ (U * s) @ V.conj().T. To improve the approximation quality
    for a fixed k, increase p or q.

    :param A: A sparse or dense matrix or GeneralLinearOperator

    :param k: The number of singular values and random projections

    :param p: The oversampling parameter

    :param q: The exponent for the projections.

    :param omega: An initial matrix to perform random projections onto with at least k columns

    :param rng: Optional NumPy Generator or seed for reproducible projections.

    :return U: The left singular vectors

    :return s: The singular values

    :return V: The right singular vectors
    """
    for name, value, minimum in (("k", k, 1), ("p", p, 0), ("q", q, 0)):
        if not isinstance(value, (int, numpy.integer)) or value < minimum:
            raise ValueError(f"Invalid {name}")

    if isinstance(A, GeneralLinearOperator):
        L = A
    else:
        L = GeneralLinearOperator.as_linear_operator(A)

    n = L.shape[1]
    if k > min(L.shape):
        raise ValueError("k exceeds matrix dimensions")
    rng = numpy.random.default_rng(rng)
    if omega is None:
        omega = rng.standard_normal((n, min(n, k + p)))
    else:
        omega = numpy.asarray(omega)
        if (
            omega.ndim != 2
            or omega.shape[0] != n
            or not k <= omega.shape[1] <= min(n, p + k)
        ):
            raise ValueError("Invalid projection matrix shape")
        missing = min(n, p + k) - omega.shape[1]
        omega = numpy.c_[omega, rng.standard_normal((n, missing))]

    Y = L.matmat(omega)
    Q, R = numpy.linalg.qr(Y)
    del omega

    for i in range(q):
        Y = L.rmatmat(Q)
        Q, R = numpy.linalg.qr(Y)
        gc.collect()

        Y = L.matmat(Q)
        Q, R = numpy.linalg.qr(Y)
        gc.collect()

    del Y
    del R
    gc.collect()

    B = L.rmatmat(Q).conj().T
    U, s, V = numpy.linalg.svd(B, full_matrices=False)
    del B
    V = V.conj().T
    U = Q.dot(U)

    U = U[:, 0:k]
    s = s[0:k]
    V = V[:, 0:k]

    return U, s, V
