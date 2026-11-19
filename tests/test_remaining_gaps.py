"""Numerics for direct sorted products, SIMD panels, and relative tolerances."""

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.sparse import csr_array

import sppy
from sppy import _native
from sppy.linalg import ComputationInfo, bicgstab


@pytest.mark.parametrize(
    "width", [63, 64, 65, 4095, 4096, 4097, 262145, 1048576, 1048577]
)
def test_product_word_boundaries_reset_and_keep_structural_zeros(width, dtype, order):
    targets = sorted(
        {v for v in [0, 1, 63, 64, 4095, 4096, 262143, 262144, width - 1] if v < width}
    )
    ri = np.repeat(np.arange(3, dtype=np.int32), len(targets))
    ci = np.tile(np.array(targets, dtype=np.int32), 3)
    values = ((np.arange(ri.size) % 5) - 2).astype(dtype)
    dense = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=dtype)
    if dtype.kind == "c":
        values += 1j * values
        dense += 0.5j * dense
    reference = csr_array((values, (ri, ci)), shape=(3, width))
    if order == "C":
        left = sppy.asarray(dense, order=order)
        right = sppy.SparseArray.from_triplets((3, width), ri, ci, values, order=order)
        expected = csr_array(dense) @ reference
        pa, pb = csr_array(dense), reference.copy()
    else:
        left = sppy.SparseArray.from_triplets((width, 3), ci, ri, values, order=order)
        right = sppy.asarray(dense.T, order=order)
        expected = reference.T @ csr_array(dense.T)
        pa, pb = reference.T.copy(), csr_array(dense.T)
    result = left @ right
    difference = result.to_scipy_csr() - expected
    assert not np.any(difference.data)
    pa.data[:] = 1
    pb.data[:] = 1
    assert result.nnz == (pa @ pb).nnz
    assert result.dtype == dtype
    assert result.is_compressed
    _, indices, indptr = result.compressed_arrays()
    for begin, end in zip(indptr[:-1], indptr[1:]):
        assert np.all(np.diff(indices[begin:end]) > 0)


@pytest.mark.parametrize(
    "value_dtype", [np.float32, np.float64, np.complex64, np.complex128]
)
@pytest.mark.parametrize(
    "shape",
    [
        (0, 8, 4),
        (4, 8, 0),
        (1, 1, 1),
        (3, 5, 7),
        (4, 2, 4),
        (5, 3, 9),
        (8, 4, 17),
        (17, 8, 31),
        (65, 16, 128),
        (4, 33, 7),
    ],
)
@pytest.mark.parametrize("adjoint", [False, True])
@pytest.mark.parametrize("scaled", [False, True])
@pytest.mark.parametrize("padding", [0, 3])
def test_panel_products_and_fallbacks_preserve_accumulation(
    value_dtype, shape, adjoint, scaled, padding
):
    rng = np.random.default_rng(12)
    m, n, k = shape

    def values(shape):
        a = rng.normal(size=shape).astype(value_dtype)
        if np.dtype(value_dtype).kind == "c":
            a += 0.3j * a
        return a[::-1, ::-1]

    left = values((k, m) if adjoint else (m, k))
    right, initial = values((k, n)), values((m, n))
    alpha = value_dtype(2 if scaled else 1)
    if scaled and np.dtype(value_dtype).kind == "c":
        alpha += value_dtype(0.5j)
    expected = initial + alpha * ((left.conj().T if adjoint else left) @ right)
    result = getattr(_native, f"_gemm_{np.dtype(value_dtype).name}")(
        left, right, initial, alpha, adjoint, padding
    )
    np.testing.assert_allclose(result, expected, rtol=3e-5, atol=3e-5)
    assert result.dtype == expected.dtype


@pytest.mark.parametrize(
    "value_dtype", [np.float32, np.float64, np.complex64, np.complex128]
)
@pytest.mark.parametrize("scale", [1e-12, 1.0, 1e12])
def test_bicgstab_relative_tolerance_matches_rhs_scale(value_dtype, scale, order):
    rng = np.random.default_rng(91)
    dense = rng.normal(size=(31, 31)).astype(value_dtype)
    dense[rng.random(dense.shape) < 0.85] = 0
    dense += 10 * np.eye(31, dtype=value_dtype)
    rhs = rng.normal(size=31).astype(value_dtype) * value_dtype(scale)
    if np.dtype(value_dtype).kind == "c":
        dense += 0.1j * dense
        rhs += 0.3j * rhs
    tolerance = (
        1e-5
        if np.dtype(value_dtype).itemsize
        // (2 if np.dtype(value_dtype).kind == "c" else 1)
        == 4
        else 1e-8
    )
    result = bicgstab(sppy.asarray(dense, order=order), rhs, rtol=tolerance)
    assert result.info == ComputationInfo.SUCCESS
    assert result.error <= tolerance
    assert (
        np.linalg.norm(dense @ result.solution - rhs) / np.linalg.norm(rhs)
        < tolerance * 3
    )


def test_bicgstab_each_rhs_has_its_own_absolute_threshold(order):
    rng = np.random.default_rng(33)
    dense = rng.normal(size=(17, 17)) + 10 * np.eye(17)
    scales = np.array([1e-12, 1.0, 1e12])
    rhs = rng.normal(size=(17, 3)) * scales
    result = bicgstab(
        sppy.asarray(dense, order=order), rhs, x0=np.ones_like(rhs) * scales, rtol=1e-9
    )
    assert result.info == ComputationInfo.SUCCESS
    errors = np.linalg.norm(dense @ result.solution - rhs, axis=0) / np.linalg.norm(
        rhs, axis=0
    )
    assert np.all(errors < 3e-9)


def test_bicgstab_iteration_limit_uses_relative_status(order):
    rng = np.random.default_rng(31)
    dense = rng.normal(size=(17, 17)) + 10 * np.eye(17)
    rhs = rng.normal(size=17) * 1e12
    result = bicgstab(sppy.asarray(dense, order=order), rhs, maxiter=1, rtol=1e-9)
    assert result.info == ComputationInfo.NO_CONVERGENCE
    assert result.error > 1e-9


def test_bicgstab_zero_rhs_reports_zero_work(order):
    result = bicgstab(sppy.eye(5, order=order), np.zeros(5), x0=np.ones(5))
    np.testing.assert_array_equal(result.solution, np.zeros(5))
    assert result.info == ComputationInfo.SUCCESS
    assert result.iterations == result.error == 0


def test_runtime_simd_opt_out_uses_numerically_equivalent_fallback():
    script = """
import numpy as np
import sppy
assert sppy._native.runtime_simd == 'baseline'
rng = np.random.default_rng(51)
for dtype in [np.float32, np.float64, np.complex64, np.complex128]:
    a = rng.normal(size=(65, 31)).astype(dtype)
    b = rng.normal(size=(31, 8)).astype(dtype)
    c = rng.normal(size=(65, 8)).astype(dtype)
    if np.dtype(dtype).kind == 'c':
        a += 0.3j * a
        b += 0.7j * b
    actual = getattr(sppy._native, '_gemm_' + np.dtype(dtype).name)(a,b,c,dtype(1),False,3)
    np.testing.assert_allclose(actual,c+a@b,rtol=3e-5,atol=3e-5)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "SPPY_DISABLE_SIMD": "1"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
