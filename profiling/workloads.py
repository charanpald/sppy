"""Matched public-API workloads shared by benchmarks and cProfile.

All preparation and validation happens outside measured regions. Sparse results
are compared as sparse differences, without allocating a dense square matrix.
"""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse
from scipy.sparse import linalg as sla

import sppy
from profiling._common import BACKENDS, SOLVER_OPERATIONS
from sppy import linalg


@dataclass
class Workload:
    """Prepared callable, numerical validator, and measured operation scope."""

    run: Callable[[], Any]
    check: Callable[[Any], dict[str, Any]]
    description: str


@dataclass
class SolveResult:
    """SciPy solution and convergence status with the shared solver interface."""

    solution: NDArray[Any]
    info: int


def _random_values(
    rng: np.random.Generator, count: int, dtype: np.dtype[Any]
) -> NDArray[Any]:
    """Generate contiguous real or complex normal samples."""
    values = rng.standard_normal(count)
    if dtype.kind == "c":
        values = values + 1j * rng.standard_normal(count)
    return np.ascontiguousarray(values, dtype=dtype)


def _canonical(matrix: sparse.sparray, format: str) -> sparse.sparray:
    """Copy, coalesce, prune, and sort SciPy sparse buffers with int32 indices."""
    result = matrix.asformat(format).copy()
    result.sum_duplicates()
    result.eliminate_zeros()
    result.sort_indices()
    # Use the same storage index width as SpPy on both SciPy backends.
    result.indices = result.indices.astype(np.int32)
    result.indptr = result.indptr.astype(np.int32)
    return result


class Fixture:
    """Deterministic canonical input, independent of the implementation timed."""

    def __init__(self, options: argparse.Namespace) -> None:
        """Generate reproducible canonical inputs and backend-independent references."""
        self.options = options
        self.dtype = np.dtype(options.dtype)
        self.rng = np.random.default_rng(options.seed)
        n = options.size
        if options.pattern == "random":
            self.rows = self.rng.integers(0, n, options.entries, dtype=np.int32)
            self.columns = self.rng.integers(0, n, options.entries, dtype=np.int32)
        else:
            rows, columns = [], []
            width = min(options.bandwidth, n - 1)
            for offset in range(-width, width + 1):
                r = np.arange(max(0, -offset), min(n, n - offset), dtype=np.int32)
                rows.append(r)
                columns.append(r + offset)
            self.rows, self.columns = np.concatenate(rows), np.concatenate(columns)
        self.values = _random_values(self.rng, len(self.rows), self.dtype)
        coo = sparse.coo_array((self.values, (self.rows, self.columns)), shape=(n, n))
        self.csr, self.csc = _canonical(coo, "csr"), _canonical(coo, "csc")
        self.arrays = self._backends(self.csr)
        canonical = self.csr.tocoo()
        self.sample_rows = canonical.row[:256].astype(np.int32)
        self.sample_columns = canonical.col[:256].astype(np.int32)
        self.vector = _random_values(self.rng, n, self.dtype)
        self.dense = _random_values(self.rng, n * options.columns, self.dtype).reshape(
            n, options.columns
        )
        self.rtol = (
            options.rtol
            if options.rtol is not None
            else (
                1e-5
                if self.dtype.itemsize // (2 if self.dtype.kind == "c" else 1) == 4
                else 1e-8
            )
        )
        self.atol = (
            3e-5
            if self.dtype in (np.dtype("float32"), np.dtype("complex64"))
            else 1e-10
        )
        self.check_rtol = max(self.rtol * 10, self.atol)
        self._system = None
        self._factors = None

    @staticmethod
    def _backends(csr: sparse.csr_array) -> dict[str, Any]:
        """Create independent inputs for each sparse backend."""
        return {
            "sppy_C": sppy.asarray(csr, order="C"),
            "sppy_F": sppy.asarray(csr, order="F"),
            "scipy_csr": _canonical(csr, "csr"),
            "scipy_csc": _canonical(csr, "csc"),
        }

    def fingerprint(self) -> str:
        """Hash canonical input buffers and dtype for reproducibility."""
        digest = hashlib.sha256()
        for array in (
            np.asarray(self.csr.shape, dtype=np.int64),
            self.csr.indptr,
            self.csr.indices,
            self.csr.data,
        ):
            digest.update(array.tobytes())
        digest.update(self.dtype.str.encode())
        return digest.hexdigest()

    def system(self) -> dict[str, Any]:
        """Lazily construct a strictly diagonally dominant Hermitian system."""
        if self._system is None:
            # Strict Hermitian diagonal dominance gives a well-conditioned SPD
            # system for both direct and iterative benchmarks.
            a = _canonical((self.csr + self.csr.conj().T) * 0.5, "csr")
            a.setdiag(0)
            a.eliminate_zeros()
            diagonal = np.asarray(abs(a).sum(axis=1)).ravel() + 1
            a = _canonical(a + sparse.diags_array(diagonal), "csr").astype(self.dtype)
            self._system = self._backends(a)
        return self._system

    def factors(self) -> dict[str, Any]:
        """Lazily factor the matched system for each backend."""
        if self._factors is None:
            self._factors = {name: _lu(name, a) for name, a in self.system().items()}
        return self._factors

    def check_equal(self, actual: Any, expected: Any) -> dict[str, bool]:
        """Validate result values, sparse shapes, and sparse dtypes."""
        if isinstance(expected, sppy.SparseArray):
            expected = sparse.csr_array(expected.to_scipy_csr())
        if isinstance(actual, sppy.SparseArray):
            actual = sparse.csr_array(actual.to_scipy_csr())
        if sparse.issparse(expected):
            if not sparse.issparse(actual) or actual.shape != expected.shape:
                raise AssertionError("Sparse result type or shape differs")
            if actual.dtype != expected.dtype:
                raise AssertionError(
                    f"Sparse result dtype differs: {actual.dtype} != {expected.dtype}"
                )
            difference = (sparse.csr_array(actual) - sparse.csr_array(expected)).data
            limit = self.atol + self.check_rtol * np.max(
                np.abs(expected.data), initial=0
            )
            if not np.isfinite(difference).all() or (np.abs(difference) > limit).any():
                raise AssertionError(
                    "Sparse results differ beyond the validation tolerance"
                )
        elif isinstance(expected, tuple):
            for actual_part, expected_part in zip(actual, expected, strict=True):
                self.check_equal(actual_part, expected_part)
        else:
            np.testing.assert_allclose(
                np.asarray(actual),
                np.asarray(expected),
                rtol=self.check_rtol,
                atol=self.atol,
            )
        return {"passed": True}

    def check_solution(
        self, actual: Any, a: Any, rhs: ArrayLike
    ) -> dict[str, bool | float]:
        """Validate convergence and the relative residual of a solution."""
        if isinstance(actual, SolveResult) or isinstance(
            actual, linalg.IterativeResult
        ):
            if actual.info != 0:
                raise AssertionError(f"Solver did not converge: status {actual.info}")
            actual = actual.solution
        residual = np.linalg.norm(a @ actual - rhs) / max(
            float(np.linalg.norm(rhs)), np.finfo(float).tiny
        )
        if not np.isfinite(residual) or residual > self.check_rtol:
            raise AssertionError(
                f"Relative residual {residual} exceeds {self.check_rtol}"
            )
        return {"passed": True, "relative_residual": float(residual)}

    def workloads(self, operation: str) -> dict[str, Workload]:
        """Prepare matched operations and numerical checks for each backend."""
        arrays = self.system() if operation in SOLVER_OPERATIONS else self.arrays
        reference = arrays["scipy_csr"]
        result = {}
        for name in BACKENDS:
            a = arrays[name]
            is_sppy = name.startswith("sppy_")
            order = "C" if name in ("sppy_C", "scipy_csr") else "F"
            scipy_type = sparse.csr_array if order == "C" else sparse.csc_array
            expected = None
            note = "Preparation and validation excluded; public API allocation/conversion included."
            if operation == "construct-triplets":
                run = (
                    (
                        lambda order=order: sppy.SparseArray.from_triplets(
                            reference.shape,
                            self.rows,
                            self.columns,
                            self.values,
                            order=order,
                        )
                    )
                    if is_sppy
                    else (
                        lambda scipy_type=scipy_type: scipy_type(
                            (self.values, (self.rows, self.columns)),
                            shape=reference.shape,
                        )
                    )
                )
                expected = self.csr
            elif operation == "construct-compressed":
                source = self.csr if order == "C" else self.csc
                data, indices, indptr = source.data, source.indices, source.indptr
                if is_sppy:
                    run = (
                        lambda data=data, indices=indices, indptr=indptr, order=order: (
                            sppy.SparseArray.from_compressed(
                                source.shape, data, indices, indptr, order=order
                            )
                        )
                    )
                else:
                    run = (
                        lambda data=data, indices=indices, indptr=indptr, scipy_type=scipy_type: (
                            scipy_type(
                                (data, indices, indptr), shape=source.shape, copy=True
                            )
                        )
                    )
                expected = self.csr
                note = (
                    "Copy from matching CSR/CSC buffers; buffer preparation excluded."
                )
            elif operation == "copy":
                run, expected = a.copy, reference
            elif operation == "copy-assign":

                def run(a: Any = a) -> Any:
                    """Run the prepared operation and return its result."""
                    out = a.copy()
                    out[self.sample_rows, self.sample_columns] = 1
                    return out

                expected = reference.copy()
                expected[self.sample_rows, self.sample_columns] = 1
                note = "Includes a copy plus assignment to existing entries; source stays unchanged."
            elif operation == "gather":
                run = lambda a=a: _gather(a, self.sample_rows, self.sample_columns)
                expected = _gather(reference, self.sample_rows, self.sample_columns)
            elif operation == "slice":
                run, expected = lambda a=a: a[::2, ::2], reference[::2, ::2]
            elif operation == "transpose":
                run = (
                    a.transpose
                    if is_sppy
                    else lambda a=a: a.T.asformat(a.format, copy=True)
                )
                expected = reference.T
                note = "Owned transpose in the input storage order, rather than a SciPy view."
            elif operation == "adjoint":
                run = (
                    a.adjoint
                    if is_sppy
                    else lambda a=a: a.conj().T.asformat(a.format, copy=True)
                )
                expected = reference.conj().T
                note = "Owned conjugate transpose in the input storage order."
            elif operation == "add":
                run, expected = lambda a=a: a + a, reference + reference
            elif operation == "multiply":
                run, expected = lambda a=a: a.multiply(a), reference.multiply(reference)
            elif operation == "matvec":
                run, expected = lambda a=a: a @ self.vector, reference @ self.vector
            elif operation == "matmat":
                run, expected = lambda a=a: a @ self.dense, reference @ self.dense
            elif operation == "sparse-matmul":
                right = a.H if is_sppy else a.conj().T.asformat(a.format)
                run, expected = (
                    lambda a=a, right=right: a @ right,
                    reference @ reference.conj().T,
                )
                note = "Sparse A @ A.H; adjoint preparation excluded."
            elif operation == "sum":
                run, expected = a.sum, reference.sum()
            elif operation == "column-sum":
                run, expected = lambda a=a: a.sum(axis=0), reference.sum(axis=0)
            elif operation == "mean":
                run, expected = a.mean, reference.mean()
            elif operation == "norm":
                run = (
                    (lambda a=a: linalg.norm(a))
                    if is_sppy
                    else (lambda a=a: sla.norm(a))
                )
                expected = sla.norm(reference)
            elif operation == "compressed-export":
                run = (
                    a.compressed_arrays
                    if is_sppy
                    else lambda a=a: (a.data.copy(), a.indices.copy(), a.indptr.copy())
                )
                source = self.csr if order == "C" else self.csc
                expected = source.data, source.indices, source.indptr
                note = (
                    "Independent value/index/pointer copies in matching CSR/CSC order."
                )
            elif operation == "copy-prune":
                source = a.copy()
                source[self.sample_rows, self.sample_columns] = 0

                def run(source: Any = source) -> Any:
                    """Run the prepared operation and return its result."""
                    out = source.copy()
                    out.eliminate_zeros()
                    return out

                expected = reference.copy()
                expected[self.sample_rows, self.sample_columns] = 0
                expected.eliminate_zeros()
                note = "Includes copy and exact-zero removal; explicit zeros prepared outside timing."
            elif operation == "lu-factor":
                run = lambda name=name, a=a: _lu(name, a)
                check = lambda factor: self.check_solution(
                    factor.solve(self.vector), reference, self.vector
                )
                result[name] = Workload(
                    run,
                    check,
                    "COLAMD, pivot threshold 1; SciPy CSR-to-CSC conversion is included.",
                )
                continue
            elif operation == "lu-solve":
                factor = self.factors()[name]
                rhs = self.dense
                run = lambda factor=factor, rhs=rhs: factor.solve(rhs)
                check = lambda x, rhs=rhs: self.check_solution(x, reference, rhs)
                result[name] = Workload(
                    run,
                    check,
                    "Reused LU factor, multiple dense RHS; factorization excluded.",
                )
                continue
            elif operation in ("cg", "bicgstab"):
                rhs = reference @ self.vector
                run = (
                    (
                        lambda a=a, operation=operation, rhs=rhs: getattr(
                            linalg, operation
                        )(
                            a,
                            rhs,
                            preconditioner="diagonal",
                            rtol=self.rtol,
                            maxiter=self.options.maxiter,
                        )
                    )
                    if is_sppy
                    else (
                        lambda a=a, operation=operation, rhs=rhs: _scipy_iterative(
                            a, rhs, operation, self.rtol, self.options.maxiter
                        )
                    )
                )
                check = lambda x, rhs=rhs: self.check_solution(x, reference, rhs)
                result[name] = Workload(
                    run,
                    check,
                    "Hermitian SPD input; includes diagonal preconditioner setup; zero initial guess, atol=0.",
                )
                continue
            elif operation == "triangular-solve":
                triangle = (
                    a.triangular() if is_sppy else sparse.tril(a, format=a.format)
                )
                ref_triangle = sparse.tril(reference, format="csr")
                rhs = ref_triangle @ self.vector
                run = (
                    (
                        lambda triangle=triangle, rhs=rhs: linalg.triangular_solve(
                            triangle, rhs
                        )
                    )
                    if is_sppy
                    else (
                        lambda triangle=triangle, rhs=rhs: sla.spsolve_triangular(
                            triangle, rhs, lower=True
                        )
                    )
                )
                check = lambda x, rhs=rhs: self.check_solution(x, ref_triangle, rhs)
                result[name] = Workload(
                    run,
                    check,
                    "Lower triangle prepared outside timing; public triangular solve included.",
                )
                continue
            else:
                raise ValueError(f"Unknown operation: {operation}")
            check = lambda actual, expected=expected: self.check_equal(actual, expected)
            result[name] = Workload(run, check, note)
        return result


def _lu(name: str, a: Any) -> Any:
    """Factor a backend matrix using matched LU ordering and pivot settings."""
    if name.startswith("sppy_"):
        return linalg.SparseLU(a, ordering="colamd", pivot_threshold=1.0)
    return sla.splu(a.tocsc(), permc_spec="COLAMD", diag_pivot_thresh=1.0)


def _scipy_iterative(
    a: Any, rhs: ArrayLike, method: str, rtol: float, maxiter: int
) -> SolveResult:
    """Solve with a SciPy iterative method and diagonal preconditioning."""
    inverse = 1 / a.diagonal()
    preconditioner = sla.LinearOperator(
        a.shape, matvec=lambda x: inverse * x, dtype=a.dtype
    )
    x, info = getattr(sla, method)(
        a, rhs, M=preconditioner, rtol=rtol, atol=0, maxiter=maxiter
    )
    return SolveResult(x, info)


def _gather(a: Any, rows: ArrayLike, columns: ArrayLike) -> NDArray[Any]:
    """Gather paired coordinates and normalize SciPy results to dense arrays."""
    result = a[rows, columns]
    # SciPy versions can return a sparse 1-D result for empty paired indices.
    return result.toarray() if sparse.issparse(result) else np.asarray(result)
