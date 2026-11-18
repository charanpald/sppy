"""Eigen's built-in sparse direct and iterative solvers.

No SciPy dependency is needed. Native solvers preserve float32/64 and
complex64/128 precision and return owned arrays and factors.
"""

from dataclasses import dataclass
from enum import IntEnum
from functools import wraps
from threading import RLock
import operator

import numpy as np

from sppy import _native
from sppy.sparse_array import SparseArray


class ComputationInfo(IntEnum):
    """Eigen computation status."""

    SUCCESS = 0
    NUMERICAL_ISSUE = 1
    NO_CONVERGENCE = 2
    INVALID_INPUT = 3


def _matrix(a):
    if not isinstance(a, SparseArray) or a.ndim != 2:
        raise ValueError("Expected a two-dimensional SparseArray")
    if a.dtype.kind not in "fc":
        raise ValueError("Solvers require floating-point or complex storage")
    if not all(a.shape):
        raise ValueError("Solvers require nonempty matrix dimensions")
    if not np.isfinite(a.values()).all():
        raise ValueError("Matrix must contain finite values")
    return a


def _arguments(a):
    rows, columns = a.stored_indices()
    return *a.shape, rows, columns, a.values()


def _rhs(b, dtype, rows):
    b = np.asarray(b)
    if b.ndim not in {1, 2} or b.shape[0] != rows:
        raise ValueError("Right-hand side has incompatible dimensions")
    if b.dtype.kind == "c" and dtype.kind != "c":
        raise TypeError("A real solver cannot discard a complex right-hand side")
    b = np.asarray(b, dtype=dtype)
    if not np.isfinite(b).all():
        raise ValueError("Right-hand side must contain finite values")
    vector = b.ndim == 1
    return np.ascontiguousarray(b[:, None] if vector else b), vector


def _locked(method):
    @wraps(method)
    def synchronized(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return synchronized


class SparseFactorization:
    """Reusable Eigen factorization, with separate symbolic and numeric steps.

    Parameters:
        a: Optional matrix to factor immediately.
        method: One of llt, ldlt, lu, or qr.
        ordering: natural, amd, or colamd. Defaults to amd for Cholesky,
            colamd for LU/QR. AMD requires square matrices.
        lower: Read the lower triangle for Cholesky; False reads the upper.
        shift_offset: Cholesky diagonal offset.
        shift_scale: Cholesky diagonal scale, applied before the offset.
        pivot_threshold: Optional LU or QR pivot threshold.
    """

    def __init__(
        self,
        a=None,
        *,
        method="lu",
        ordering=None,
        lower=True,
        shift_offset=0.0,
        shift_scale=1.0,
        pivot_threshold=None,
    ):
        if method not in {"llt", "ldlt", "lu", "qr"}:
            raise ValueError("Unknown direct method")
        ordering = ordering or ("amd" if method in {"llt", "ldlt"} else "colamd")
        if ordering not in {"natural", "amd", "colamd"}:
            raise ValueError("Unknown ordering")
        if not np.isfinite([shift_offset, shift_scale]).all():
            raise ValueError("Shift parameters must be finite")
        if pivot_threshold is not None and (
            not np.isfinite(pivot_threshold) or pivot_threshold < 0
        ):
            raise ValueError("Pivot threshold must be nonnegative and finite")
        self.method, self.ordering, self.lower = method, ordering, lower
        self.shift_offset, self.shift_scale = shift_offset, shift_scale
        self.pivot_threshold = pivot_threshold
        self._lock = RLock()
        self.info = ComputationInfo.INVALID_INPUT
        self._native = None
        self._shape = None
        self._dtype = None
        if a is not None:
            self.compute(a)

    def _prepare(self, a):
        a = _matrix(a)
        if self.method != "qr" and a.shape[0] != a.shape[1]:
            raise ValueError("This factorization requires a square matrix")
        if self.ordering == "amd" and a.shape[0] != a.shape[1]:
            raise ValueError("AMD requires a square matrix")
        return a.selfadjoint(lower=self.lower) if self.method in {"llt", "ldlt"} else a

    @_locked
    def analyze_pattern(self, a):
        """Analyze structure without numerical factorization; invalidates old factors."""
        self.info = ComputationInfo.INVALID_INPUT
        self._native = None
        a = self._prepare(a)
        cls = getattr(_native, f"Factor_{a.dtype.name}")
        self._native = cls(
            self.method,
            self.ordering,
            self.shift_offset,
            self.shift_scale,
            -1.0 if self.pivot_threshold is None else self.pivot_threshold,
        )
        self._shape, self._dtype = a.shape, a.dtype
        self._native.analyze(*_arguments(a))
        return self

    @_locked
    def factorize(self, a):
        """Numerically factor a matrix with the analyzed sparsity pattern.

        Raises numpy.linalg.LinAlgError on numerical failure. Shape, dtype, and
        stored coordinates must match the analyzed matrix.
        """
        self.info = ComputationInfo.INVALID_INPUT
        if self._native is None:
            raise ValueError("Call analyze_pattern before factorize")
        a = self._prepare(a)
        if a.dtype != self._dtype or a.shape != self._shape:
            raise ValueError("Shape and dtype must match the analyzed matrix")
        self.info = ComputationInfo(self._native.factorize(*_arguments(a)))
        if self.info != ComputationInfo.SUCCESS:
            raise np.linalg.LinAlgError(
                f"{self.method} factorization failed: {self.info.name}"
            )
        return self

    @_locked
    def compute(self, a):
        """Analyze and factor a matrix."""
        return self.analyze_pattern(a).factorize(a)

    def _require_ready(self):
        if self._native is None or self.info != ComputationInfo.SUCCESS:
            raise ValueError("No successful factorization is available")

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @_locked
    def solve(self, b):
        """Solve for dense vectors/matrices or a sparse RHS matrix."""
        self._require_ready()
        if isinstance(b, SparseArray):
            if b.ndim != 2 or b.shape[0] != self.shape[0]:
                raise ValueError("Sparse RHS must be a matrix with compatible rows")
            if b.dtype.kind == "c" and self.dtype.kind != "c":
                raise TypeError("A real solver cannot discard a complex RHS")
            if not np.isfinite(b.values()).all():
                raise ValueError("Right-hand side must be finite")
            b = b.astype(self.dtype, copy=False)
            rows, columns, ri, ci, values = self._native.solve_sparse(*_arguments(b))
            return SparseArray.from_triplets((rows, columns), ri, ci, values, order="F")
        b, vector = _rhs(b, self.dtype, self.shape[0])
        result = self._native.solve(b)
        return result[:, 0] if vector else result

    @_locked
    def factor(self, name):
        """Copy L/U for Cholesky/LU, D for LDLT, R for QR, or permutation P/Q.

        Permutations use Eigen's convention: P[indices[j], j] = 1. LU exposes
        row permutation P and column permutation Q; QR exposes column P.
        """
        self._require_ready()
        value = self._native.factor(name)
        if isinstance(value, tuple):
            rows, columns, ri, ci, values = value
            return SparseArray.from_triplets((rows, columns), ri, ci, values, order="F")
        return value.ravel() if name == "D" else value

    @_locked
    def determinant(self):
        """Return the determinant for LU or Cholesky."""
        self._require_ready()
        return self._native.determinant()

    @_locked
    def slogdet(self):
        """Return (phase, log(abs(det))) for LU or Cholesky without overflow."""
        self._require_ready()
        return self._native.slogdet()

    @property
    @_locked
    def rank(self):
        """Numerical rank from rank-revealing QR."""
        self._require_ready()
        return self._native.rank

    @_locked
    def apply_q(self, b, *, adjoint=False):
        """Apply QR's implicit Q or Q.H without forming a dense Q."""
        self._require_ready()
        b, vector = _rhs(b, self.dtype, self.shape[0])
        result = self._native.apply_q(b, adjoint)
        return result[:, 0] if vector else result


class SimplicialLLT(SparseFactorization):
    """Sparse Cholesky LL.H for Hermitian positive definite matrices."""

    def __init__(self, a=None, **options):
        super().__init__(a, method="llt", **options)


class SimplicialLDLT(SparseFactorization):
    """Sparse Cholesky LD L.H for Hermitian positive definite matrices."""

    def __init__(self, a=None, **options):
        super().__init__(a, method="ldlt", **options)


class SparseLU(SparseFactorization):
    """Sparse LU for general square matrices."""

    def __init__(self, a=None, **options):
        super().__init__(a, method="lu", **options)


class SparseQR(SparseFactorization):
    """Rank-revealing sparse QR for rectangular least-squares problems."""

    def __init__(self, a=None, **options):
        super().__init__(a, method="qr", **options)


def factorize(a, *, method="lu", **options):
    """Construct a reusable native sparse factorization."""
    return SparseFactorization(a, method=method, **options)


def solve(a, b, *, method="lu", **options):
    """Factor and solve once; use factorize for repeated right-hand sides."""
    return factorize(a, method=method, **options).solve(b)


def ordering(a, method="amd"):
    """Return a gather permutation for AMD, COLAMD, or natural ordering.

    AMD expects a square matrix and operates on its self-adjoint structure.
    Use a.permuted(p, p) for symmetric ordering, or a.permuted(columns=p)
    for a column ordering.
    """
    if not isinstance(a, SparseArray) or a.ndim != 2:
        raise ValueError("Ordering requires a sparse matrix")
    if method not in {"natural", "amd", "colamd"}:
        raise ValueError("Unknown ordering")
    if method == "amd" and a.shape[0] != a.shape[1]:
        raise ValueError("AMD requires a square matrix")
    if not all(a.shape):
        return np.arange(a.shape[1])
    a = a if a.dtype.kind in "fc" else a.astype(np.float64)
    function = getattr(_native, f"ordering_{a.dtype.name}")
    return np.argsort(function(*_arguments(a), method))


def triangular_solve(a, b, *, lower=True, unit_diagonal=False):
    """Solve a sparse triangular system; optionally assume a unit diagonal."""
    a = _matrix(a)
    if a.shape[0] != a.shape[1]:
        raise ValueError("Triangular solve requires a square matrix")
    b, vector = _rhs(b, a.dtype, a.shape[0])
    result = getattr(_native, f"triangular_solve_{a.dtype.name}")(
        *_arguments(a),
        b,
        lower,
        unit_diagonal,
    )
    return result[:, 0] if vector else result


@dataclass(frozen=True)
class IterativeResult:
    """Solution, Eigen status, iteration count, and estimated relative error.

    Unpacking yields (solution, info). Inspect iterations/error on the object.
    For multiple right-hand sides, status, iterations, and error are the worst
    values across columns. Check info before using a nonconverged solution.
    """

    solution: np.ndarray
    info: ComputationInfo
    iterations: int
    error: float

    def __iter__(self):
        yield self.solution
        yield self.info


def _iterate(
    a,
    b,
    *,
    method,
    preconditioner,
    x0=None,
    maxiter=1000,
    rtol=1e-6,
    drop_tolerance=1e-4,
    fill_factor=10,
    initial_shift=1e-3,
):
    a = _matrix(a)
    b, vector = _rhs(b, a.dtype, a.shape[0])
    if x0 is None:
        guess = np.zeros((a.shape[1], b.shape[1]), dtype=a.dtype)
    else:
        guess, guess_vector = _rhs(x0, a.dtype, a.shape[1])
        if guess.shape[1] != b.shape[1] or guess_vector != vector:
            raise ValueError("Initial guess must match the solution shape")
    maxiter, fill_factor = operator.index(maxiter), operator.index(fill_factor)
    function = getattr(_native, f"iterate_{a.dtype.name}")
    x, info, iterations, error = function(
        *_arguments(a),
        b,
        guess,
        method,
        preconditioner,
        maxiter,
        rtol,
        drop_tolerance,
        fill_factor,
        initial_shift,
    )
    return IterativeResult(
        x[:, 0] if vector else x, ComputationInfo(info), iterations, error
    )


def cg(a, b, *, preconditioner="diagonal", **options):
    """Conjugate gradient for Hermitian positive definite matrices.

    Preconditioners: identity, diagonal, incomplete_cholesky. Both triangles
    are read. Options: x0, maxiter, rtol, initial_shift.
    """
    return _iterate(a, b, method="cg", preconditioner=preconditioner, **options)


def bicgstab(a, b, *, preconditioner="diagonal", **options):
    """BiCGSTAB for general square matrices.

    Preconditioners: identity, diagonal, incomplete_lut. Options: x0, maxiter,
    rtol, drop_tolerance, fill_factor.
    """
    return _iterate(a, b, method="bicgstab", preconditioner=preconditioner, **options)


def lscg(a, b, *, preconditioner="least_square_diagonal", **options):
    """Least-squares conjugate gradient for rectangular matrices.

    Preconditioners: identity, least_square_diagonal. Options: x0, maxiter, rtol.
    The estimated error is measured on the normal equations, as in Eigen.
    """
    return _iterate(a, b, method="lscg", preconditioner=preconditioner, **options)


class Preconditioner:
    """Reusable native inverse preconditioner for dense vectors or RHS matrices.

    Methods: identity, diagonal, least_square_diagonal, incomplete_lut, and
    incomplete_cholesky. Eigen's diagonal preconditioners use a unit inverse for
    zero diagonal entries. ILUT accepts drop_tolerance/fill_factor; incomplete
    Cholesky accepts initial_shift. The matrix is snapshotted during construction.
    """

    def __init__(
        self,
        a,
        method="diagonal",
        *,
        drop_tolerance=1e-4,
        fill_factor=10,
        initial_shift=1e-3,
    ):
        a = _matrix(a)
        self.dtype, self.shape = a.dtype, (a.shape[1], a.shape[1])
        cls = getattr(_native, f"Preconditioner_{a.dtype.name}")
        self._native = cls(
            *_arguments(a),
            method,
            drop_tolerance,
            operator.index(fill_factor),
            initial_shift,
        )

    def solve(self, b):
        """Apply the inverse preconditioner to dense right-hand sides."""
        b, vector = _rhs(b, self.dtype, self.shape[0])
        result = self._native.solve(b)
        return result[:, 0] if vector else result
