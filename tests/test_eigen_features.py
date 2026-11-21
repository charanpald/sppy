"""Numerical contracts for Eigen 5 sparse modules and the 1.0 API."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import DTypeLike, NDArray

import sppy
from sppy import SparseArray
from sppy.linalg import (
    ComputationInfo,
    Preconditioner,
    SimplicialLDLT,
    SimplicialLLT,
    SparseLU,
    SparseQR,
    bicgstab,
    cg,
    lscg,
    ordering,
    triangular_solve,
)

NUMERIC = [np.float32, np.float64, np.complex64, np.complex128]
DIRECT = [SimplicialLLT, SimplicialLDLT, SparseLU, SparseQR]


def system(dtype: DTypeLike, rectangular: bool = False) -> NDArray[Any]:
    """Construct a deterministic real or Hermitian dense test system."""
    values = np.array([[5, 1, 0], [1, 4, 1], [0, 1, 3]], dtype=dtype)
    if np.dtype(dtype).kind == "c":
        values[0, 1] += 0.5j
        values[1, 0] -= 0.5j
    if rectangular:
        values = np.vstack([values, [[1, 0, 2]]]).astype(dtype)
    return values


@pytest.mark.parametrize("dtype", NUMERIC)
@pytest.mark.parametrize("solver", DIRECT)
@pytest.mark.parametrize("multiple_rhs", [False, True])
def test_direct_solutions_and_precision(
    dtype: DTypeLike, solver: Any, multiple_rhs: bool, order: str
) -> None:
    """Verify direct solutions and precision."""
    a = system(dtype)
    expected = np.arange(1, 4, dtype=dtype)
    if multiple_rhs:
        expected = np.column_stack([expected, -expected])
    factor = solver(sppy.asarray(a, order=order))
    actual = factor.solve(a @ expected)
    assert actual.dtype == np.dtype(dtype)
    assert factor.info == ComputationInfo.SUCCESS
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
    if solver != SparseQR:
        np.testing.assert_allclose(
            factor.determinant(), np.linalg.det(a), rtol=2e-5, atol=2e-5
        )


@pytest.mark.parametrize("solver", DIRECT)
@pytest.mark.parametrize("method", ["natural", "amd", "colamd"])
def test_direct_orderings(solver: Any, method: str) -> None:
    """Verify direct orderings."""
    a = system(np.float64)
    factor = solver(sppy.asarray(a), ordering=method)
    np.testing.assert_allclose(a @ factor.solve(np.ones(3)), np.ones(3), atol=1e-12)


@pytest.mark.parametrize("dtype", NUMERIC)
def test_qr_rectangular_factorization_and_implicit_q(dtype: DTypeLike) -> None:
    """Verify qr rectangular factorization and implicit q."""
    a = system(dtype, rectangular=True)
    factor = SparseQR(sppy.asarray(a))
    b = np.arange(1, 5, dtype=dtype)
    expected = np.linalg.lstsq(a, b, rcond=None)[0]
    np.testing.assert_allclose(factor.solve(b), expected, rtol=3e-5, atol=3e-5)
    q = factor.apply_q(np.eye(4, dtype=dtype))
    p = np.eye(3, dtype=dtype)[:, factor.factor("P")]
    np.testing.assert_allclose(q.conj().T @ q, np.eye(4), atol=3e-5)
    np.testing.assert_allclose(a @ p, q @ factor.factor("R").toarray(), atol=3e-5)
    np.testing.assert_allclose(
        factor.apply_q(b, adjoint=True), q.conj().T @ b, atol=3e-5
    )
    assert factor.rank == 3


@pytest.mark.parametrize("solver", [SimplicialLLT, SimplicialLDLT])
@pytest.mark.parametrize("lower", [False, True])
@pytest.mark.parametrize("dtype", NUMERIC)
def test_cholesky_factor_reconstruction(
    solver: Any, lower: bool, dtype: DTypeLike
) -> None:
    """Verify cholesky factor reconstruction."""
    a = system(dtype)
    triangle = np.tril(a) if lower else np.triu(a)
    factor = solver(sppy.asarray(triangle), lower=lower)
    p = np.eye(3, dtype=dtype)[:, factor.factor("P")]
    # Eigen represents natural Cholesky ordering with an empty permutation.
    if not p.size:
        p = np.eye(3, dtype=dtype)
    left = factor.factor("L").toarray()
    middle = np.diag(factor.factor("D")) if solver == SimplicialLDLT else np.eye(3)
    np.testing.assert_allclose(
        p @ a @ p.conj().T, left @ middle @ left.conj().T, atol=3e-5
    )


@pytest.mark.parametrize("solver", DIRECT)
def test_symbolic_analysis_reuse_and_failed_pattern_invalidates_solve(
    solver: Any,
) -> None:
    """Verify symbolic analysis reuse and failed pattern invalidates solve."""
    a = sppy.asarray(system(np.float64))
    factor = solver().analyze_pattern(a).factorize(a)
    changed = a * 2
    factor.factorize(changed)
    np.testing.assert_allclose(
        changed @ factor.solve(np.ones(3)), np.ones(3), atol=1e-12
    )
    changed[2, 0] = 1
    with pytest.raises(ValueError, match="pattern"):
        factor.factorize(changed)
    with pytest.raises(ValueError, match="successful"):
        factor.solve(np.ones(3))


@pytest.mark.parametrize("solver", [SimplicialLLT, SimplicialLDLT, SparseLU])
def test_singular_direct_failure(solver: Any) -> None:
    """Verify singular direct failure."""
    with pytest.raises(np.linalg.LinAlgError):
        solver(sppy.zeros((3, 3)))


def test_qr_reports_rank_deficiency_and_solves_underdetermined_system() -> None:
    """Verify qr reports rank deficiency and solves underdetermined system."""
    a = sppy.asarray([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
    factor = SparseQR(a)
    assert factor.rank == 1
    x = factor.solve([1.0, 2.0])
    np.testing.assert_allclose(a @ x, [1.0, 2.0], atol=1e-12)


@pytest.mark.parametrize(
    "solver,preconditioner",
    [
        (cg, "identity"),
        (cg, "diagonal"),
        (cg, "incomplete_cholesky"),
        (bicgstab, "identity"),
        (bicgstab, "diagonal"),
        (bicgstab, "incomplete_lut"),
        (lscg, "identity"),
        (lscg, "least_square_diagonal"),
    ],
)
@pytest.mark.parametrize("dtype", NUMERIC)
def test_iterative_methods_and_preconditioners(
    solver: Any, preconditioner: str, dtype: DTypeLike
) -> None:
    """Verify iterative methods and preconditioners."""
    a = system(dtype, rectangular=solver == lscg)
    x = np.arange(1, 4, dtype=dtype)
    result = solver(
        sppy.asarray(a), a @ x, preconditioner=preconditioner, rtol=1e-5, maxiter=100
    )
    assert result.info == ComputationInfo.SUCCESS
    assert result.solution.dtype == np.dtype(dtype)
    np.testing.assert_allclose(result.solution, x, rtol=1e-4, atol=1e-4)
    assert 0 <= result.iterations <= 100
    assert np.isfinite(result.error)


@pytest.mark.parametrize("solver", [cg, bicgstab, lscg])
def test_iterative_initial_guess_and_matrix_rhs(solver: Any) -> None:
    """Verify iterative initial guess and matrix rhs."""
    a = system(np.float64, rectangular=solver == lscg)
    x = np.arange(6.0).reshape(3, 2)
    result = solver(sppy.asarray(a), a @ x, x0=x, rtol=1e-12)
    np.testing.assert_allclose(result.solution, x, atol=1e-12)
    assert result.info == ComputationInfo.SUCCESS


def test_iterative_reports_nonconvergence() -> None:
    """Verify iterative reports nonconvergence."""
    a = sppy.asarray(system(np.float64))
    result = cg(a, [1.0, 2.0, 3.0], maxiter=1, rtol=1e-15, preconditioner="identity")
    assert result.info == ComputationInfo.NO_CONVERGENCE
    assert result.error > 1e-15


@pytest.mark.parametrize("dtype", NUMERIC)
@pytest.mark.parametrize(
    "method",
    [
        "identity",
        "diagonal",
        "least_square_diagonal",
        "incomplete_lut",
        "incomplete_cholesky",
    ],
)
def test_standalone_preconditioner(dtype: DTypeLike, method: str) -> None:
    """Verify standalone preconditioner."""
    diagonal = np.array([2.0, 3.0, 4.0], dtype=dtype)
    a = sppy.diag(diagonal)
    inverse = Preconditioner(a, method)
    expected = np.ones(3, dtype=dtype)
    rhs = diagonal if method != "identity" else expected
    if method == "least_square_diagonal":
        rhs = diagonal**2
    np.testing.assert_allclose(inverse.solve(rhs), expected, rtol=2e-5, atol=2e-5)
    a.set_zero()
    np.testing.assert_allclose(inverse.solve(rhs), expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("dtype", NUMERIC)
@pytest.mark.parametrize("lower", [False, True])
@pytest.mark.parametrize("unit", [False, True])
def test_triangular_solve(
    dtype: DTypeLike, lower: bool, unit: bool, order: str
) -> None:
    """Verify triangular solve."""
    a = system(dtype)
    triangle = np.tril(a) if lower else np.triu(a)
    if unit:
        np.fill_diagonal(triangle, 1)
    rhs = np.column_stack([np.ones(3), np.arange(3)]).astype(dtype)
    x = triangular_solve(
        sppy.asarray(a, order=order), rhs, lower=lower, unit_diagonal=unit
    )
    np.testing.assert_allclose(triangle @ x, rhs, atol=3e-5)


def test_triangular_zero_diagonal_rejected() -> None:
    """Verify triangular zero diagonal rejected."""
    with pytest.raises(ValueError, match="Singular"):
        triangular_solve(sppy.zeros((3, 3)), np.ones(3))


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_complex_operations(dtype: DTypeLike, order: str) -> None:
    """Verify complex operations."""
    dense = np.array([[0, 1 + 2j], [3 - 4j, 0]], dtype=dtype)
    a = sppy.asarray(dense, order=order)
    for actual, expected in [
        (a.H, dense.conj().T),
        (abs(a), np.abs(dense)),
        (a.real, dense.real),
        (a.imag, dense.imag),
        (a.abs2(), np.abs(dense) ** 2),
    ]:
        np.testing.assert_allclose(actual.toarray(), expected, atol=1e-5)
        assert actual.dtype == expected.dtype
    np.testing.assert_allclose(a @ a, dense @ dense)
    np.testing.assert_allclose(
        a @ np.ones(2, dtype=dtype), dense @ np.ones(2), atol=1e-5
    )
    np.testing.assert_allclose(a.var(axis=0), dense.var(axis=0), atol=1e-5)
    np.testing.assert_allclose(a.mean(axis=1), dense.mean(axis=1), atol=1e-5)
    triangle = np.tril(dense)
    np.testing.assert_allclose(
        a.selfadjoint().toarray(), triangle + np.tril(dense, -1).conj().T
    )


@pytest.mark.parametrize("method", ["natural", "amd", "colamd"])
def test_ordering_and_permutation(method: str, order: str) -> None:
    """Verify ordering and permutation."""
    a = sppy.asarray(system(np.float64), order=order)
    p = ordering(a, method)
    np.testing.assert_array_equal(np.sort(p), np.arange(3))
    np.testing.assert_array_equal(a.permuted(p, p).toarray(), a.toarray()[np.ix_(p, p)])


@pytest.mark.parametrize("dtype", NUMERIC)
def test_compressed_roundtrip_and_storage_ownership(
    dtype: DTypeLike, order: str
) -> None:
    """Verify compressed roundtrip and storage ownership."""
    a = sppy.asarray(system(dtype), order=order)
    values, indices, indptr = a.compressed_arrays()
    b = SparseArray.from_compressed(a.shape, values, indices, indptr, order=order)
    np.testing.assert_array_equal(a.toarray(), b.toarray())
    values[:] = 0
    assert b.nnz == a.nnz
    assert not b.is_zero()
    a.uncompress()
    assert not a.is_compressed
    a.reserve(np.full(a.outer_size, 3, dtype=np.int32))
    a.squeeze()
    assert a.is_compressed


def test_invalid_compressed_pointers_rejected() -> None:
    """Verify invalid compressed pointers rejected."""
    with pytest.raises(ValueError, match="pointers"):
        SparseArray.from_compressed((2, 2), [1.0], [0], [0, 2, 1])


def test_stored_zeros_have_distinct_index_contract() -> None:
    """Verify stored zeros have distinct index contract."""
    a = sppy.eye(2)
    a[0, 1] = 0
    assert a.nnz == 3
    assert len(a.stored_indices()[0]) == 3
    assert len(a.nonzero()[0]) == 2
    a.eliminate_zeros()
    assert a.nnz == 2


def test_storage_resize_swap_diagonal_and_insert(order: str) -> None:
    """Verify storage resize swap diagonal and insert."""
    a = sppy.eye(3, order=order)
    a.resize((4, 5), preserve=True)
    np.testing.assert_array_equal(a.toarray()[:3, :3], np.eye(3))
    a.set_diagonal(2)
    a.insert_from_triplets([0, 0], [1, 1], [2.0, 3.0])
    assert a[0, 1] == 5
    b = sppy.zeros((2, 2), order=order)
    a.swap(b)
    assert a.shape == (2, 2) and a.is_zero()
    assert b.shape == (4, 5)
    b.resize((2, 3))
    assert b.is_zero()
    b.set_identity()
    np.testing.assert_array_equal(b.toarray(), np.eye(2, 3))


@pytest.mark.parametrize("method", ["minimum", "maximum"])
def test_coefficient_extrema_include_implicit_zeros(method: str) -> None:
    """Verify coefficient extrema include implicit zeros."""
    a = np.array([[0, 2], [-3, 0]], dtype=np.int64)
    b = np.array([[1, 0], [0, -4]], dtype=np.int64)
    result = getattr(sppy.asarray(a), method)(sppy.asarray(b))
    np.testing.assert_array_equal(result.toarray(), getattr(np, method)(a, b))
    assert result.dtype == a.dtype


def test_diagonal_scaling_and_sparse_norms() -> None:
    """Verify diagonal scaling and sparse norms."""
    dense = system(np.complex128)
    a = sppy.asarray(dense)
    scale = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(a.scale_rows(scale).toarray(), scale[:, None] * dense)
    np.testing.assert_array_equal(a.scale_columns(scale).toarray(), dense * scale)
    np.testing.assert_allclose(a.squared_norm(), np.linalg.norm(dense) ** 2)
    assert a.is_approx(a.copy())
    vector = sppy.asarray([3.0, 0.0, 4.0])
    np.testing.assert_allclose(vector.normalized().toarray(), [0.6, 0.0, 0.8])


@pytest.mark.parametrize(
    "legacy", ["csarray", "toScipyCsr", "pdot", "rowInds", "hadamard"]
)
def test_legacy_symbols_removed(legacy: str) -> None:
    """Verify legacy symbols removed."""
    target = sppy if legacy == "csarray" else sppy.eye(2)
    assert not hasattr(target, legacy)


@pytest.mark.parametrize("function", [cg, bicgstab, lscg])
@pytest.mark.parametrize(
    "options",
    [
        {"rtol": np.nan},
        {"rtol": 0},
        {"maxiter": -1},
        {"fill_factor": 0},
        {"preconditioner": "invalid"},
    ],
)
def test_iterative_rejects_invalid_controls(
    function: Any, options: dict[str, Any]
) -> None:
    """Verify iterative rejects invalid controls."""
    with pytest.raises((ValueError, TypeError)):
        function(sppy.eye(3), np.ones(3), **options)


def test_real_solver_rejects_complex_rhs() -> None:
    """Verify real solver rejects complex rhs."""
    with pytest.raises(TypeError, match="complex"):
        SparseLU(sppy.eye(3)).solve(np.ones(3, dtype=np.complex128))


def test_shifted_cholesky() -> None:
    """Verify shifted cholesky."""
    a = sppy.diag(np.array([-1.0, 2.0, 3.0]))
    factor = SimplicialLLT(a, shift_offset=2.0)
    np.testing.assert_allclose(
        (a.toarray() + 2 * np.eye(3)) @ factor.solve(np.ones(3)), np.ones(3)
    )


@pytest.mark.parametrize("solver", DIRECT)
@pytest.mark.parametrize("dtype", NUMERIC)
def test_sparse_rhs_solve(solver: Any, dtype: DTypeLike) -> None:
    """Verify sparse rhs solve."""
    a = system(dtype)
    rhs = sppy.eye(3, dtype=dtype)
    result = solver(sppy.asarray(a)).solve(rhs)
    assert isinstance(result, SparseArray)
    assert result.dtype == np.dtype(dtype)
    np.testing.assert_allclose(a @ result.toarray(), np.eye(3), atol=3e-5)


@pytest.mark.parametrize("dtype", NUMERIC)
def test_lu_factor_reconstruction(dtype: DTypeLike) -> None:
    """Verify lu factor reconstruction."""
    a = system(dtype)
    a[0, 0] = 0
    factor = SparseLU(sppy.asarray(a))
    p = np.eye(3)[:, factor.factor("P")]
    q = np.eye(3)[:, factor.factor("Q")]
    np.testing.assert_allclose(
        p @ a @ q.conj().T,
        factor.factor("L").toarray() @ factor.factor("U").toarray(),
        atol=3e-5,
    )


@pytest.mark.parametrize("solver", [SimplicialLLT, SimplicialLDLT, SparseLU])
@pytest.mark.parametrize("dtype", NUMERIC)
def test_stable_logdet(solver: Any, dtype: DTypeLike) -> None:
    """Verify stable logdet."""
    a = np.diag(np.array([1e20, 1e20, 1e20], dtype=dtype))
    factor = solver(sppy.asarray(a))
    phase, logabs = factor.slogdet()
    expected_phase, expected_logabs = np.linalg.slogdet(a)
    np.testing.assert_allclose(phase, expected_phase, atol=2e-5)
    np.testing.assert_allclose(logabs, expected_logabs, rtol=2e-5)


def test_eliminate_zeros_preserves_nan() -> None:
    """Verify eliminate zeros preserves nan."""
    a = sppy.asarray([np.nan, 1.0, 0.0])
    a[1] = 0
    a.eliminate_zeros()
    assert a.nnz == 1
    assert np.isnan(a[0])


def test_empty_list_index_and_out_of_range_transpose() -> None:
    """Verify empty list index and out of range transpose."""
    a = sppy.eye(3)
    assert a[[], :].shape == (0, 3)
    with pytest.raises(ValueError):
        a.transpose(3, 0)


def test_integer_prune_retains_minimum_and_exact_threshold() -> None:
    """Verify integer prune retains minimum and exact threshold."""
    a = sppy.asarray(
        np.array([np.iinfo(np.int64).min, 2**60 + 1, 2**60], dtype=np.int64)
    )
    a.prune(float(2**60))
    assert a.nnz == 2
    assert a[0] == np.iinfo(np.int64).min
    assert a[1] == 2**60 + 1


def test_integer_structural_ordering() -> None:
    """Verify integer structural ordering."""
    p = ordering(sppy.asarray(np.eye(3, dtype=np.int64)))
    np.testing.assert_array_equal(np.sort(p), np.arange(3))


def test_complex_randomized_svd_reconstruction_and_reproducibility() -> None:
    """Verify complex randomized svd reconstruction and reproducibility."""
    from sppy.linalg import rsvd

    rng = np.random.default_rng(42)
    dense = rng.normal(size=(8, 5)) + 1j * rng.normal(size=(8, 5))
    a = sppy.asarray(dense)
    u, s, v = rsvd(a, 5, p=0, rng=7)
    np.testing.assert_allclose((u * s) @ v.conj().T, dense, atol=1e-12)
    repeat = rsvd(a, 5, p=0, rng=7)
    for actual, expected in zip((u, s, v), repeat):
        np.testing.assert_array_equal(actual, expected)


def test_factor_allows_concurrent_solves() -> None:
    """Verify factor allows concurrent solves."""
    from concurrent.futures import ThreadPoolExecutor

    a = sppy.asarray(system(np.float64))
    factor = SparseLU(a)
    with ThreadPoolExecutor(max_workers=4) as executor:
        solutions = list(executor.map(factor.solve, [np.ones(3)] * 8))
    for x in solutions:
        np.testing.assert_allclose(a @ x, np.ones(3), atol=1e-12)


@pytest.mark.parametrize("dtype", NUMERIC)
def test_sparse_vector_inner_outer_and_dense_products(
    dtype: DTypeLike, order: str
) -> None:
    """Verify sparse vector inner outer and dense products."""
    x = np.array([0, 2, -3], dtype=dtype)
    if np.dtype(dtype).kind == "c":
        x[1] += 2j
    y = np.array([1, 0, 4], dtype=dtype)
    a, b = sppy.asarray(x, order=order), sppy.asarray(y, order=order)
    assert a.outer_size == 1
    np.testing.assert_allclose(a @ b, x @ y)
    np.testing.assert_allclose(a.vdot(b), np.vdot(x, y))
    np.testing.assert_allclose(a.outer(b).toarray(), np.outer(x, y))
    matrix = sppy.asarray(system(dtype), order=order)
    np.testing.assert_allclose(a @ matrix, x @ matrix.toarray(), atol=3e-5)
    np.testing.assert_allclose(matrix @ a, matrix.toarray() @ x, atol=3e-5)


def test_inplace_arithmetic_retains_identity_and_rejects_unsafe_casts() -> None:
    """Verify inplace arithmetic retains identity and rejects unsafe casts."""
    a = sppy.eye(3)
    same = a
    a += a
    a *= 3
    a /= 2
    assert a is same
    np.testing.assert_array_equal(a.toarray(), 3 * np.eye(3))
    integer = sppy.eye(2, dtype=np.int64)
    with pytest.raises(TypeError):
        integer /= 2


def test_huge_sparse_vector_keeps_constant_outer_storage() -> None:
    """Verify huge sparse vector keeps constant outer storage."""
    a = sppy.zeros(2**31 - 1)
    a[-1] = 3
    assert a.outer_size == 1
    assert a.nnz == 1
    assert a @ a == 9
    assert a.sum() == 3
