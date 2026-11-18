# Linear algebra

These functions call Eigen directly and require no SciPy installation. Solvers
accept float32/64 and complex64/128 matrices. Integer matrices must be explicitly
cast before solving. See [Eigen coverage](eigen-coverage.md) for the module map.

## Direct solves and factor reuse

```python
import numpy as np
import sppy
from sppy.linalg import SparseLU, SparseQR, SimplicialLLT, cg, ComputationInfo

a = sppy.asarray([[4., 1.], [1., 3.]])
b = np.array([1., 2.])
factor = SparseLU(a)
x = factor.solve(b)
np.testing.assert_allclose(a @ x, b)

# Reuse the symbolic pattern when only coefficients change.
factor.factorize(a * 2)
np.testing.assert_allclose((a * 2) @ factor.solve(b), b)
```

Choose LLT/LDLT for Hermitian positive definite systems, LU for general square
systems, and QR for rectangular least squares. All support dense vector or matrix
right-hand sides and sparse RHS matrices. Sparse RHS solves return SparseArray. Cholesky reads the lower triangle by default; `lower=False`
uses the upper triangle. Its `shift_offset` and `shift_scale` transform the
diagonal as `scale * diagonal + offset`.

Direct factorization failures raise `numpy.linalg.LinAlgError`. Analyze and
factorize are separate operations; `compute` performs both. Numeric refactorization
requires an identical stored sparsity pattern, including explicitly stored zeros.
Changing the matrix afterward does not alter an existing factorization.

## Factors and implicit Q

Cholesky exposes `factor("L")`, `factor("U")`, and `factor("P")`; LDLT also
exposes `factor("D")`. LU exposes sparse `L`/`U`, row `P` and column `Q` permutations, a
determinant, and a stable `slogdet()`. QR exposes sparse `R`, column `P`, numerical `rank`, and
`apply_q(rhs, adjoint=False)` without allocating a full dense Q.

Permutation arrays follow Eigen's convention: `P[indices[j], j] = 1`.
The empty Cholesky permutation represents identity when no reordering is needed.

```python
rectangular = sppy.asarray([[1., 0.], [0., 1.], [1., 1.]])
qr = SparseQR(rectangular)
least_squares = qr.solve(np.array([1., 2., 3.]))
assert qr.rank == 2
np.testing.assert_allclose(rectangular @ least_squares, [1., 2., 3.])
```

## Iterative solves

```python
result = cg(a, b, preconditioner="incomplete_cholesky", rtol=1e-10)
assert result.info == ComputationInfo.SUCCESS
np.testing.assert_allclose(a @ result.solution, b)
x, status = result
```

| Function | Matrix | Available preconditioners |
| --- | --- | --- |
| `cg` | Hermitian positive definite; both triangles are read | identity, diagonal (default), incomplete_cholesky |
| `bicgstab` | General square | identity, diagonal (default), incomplete_lut |
| `lscg` | Rectangular least squares | identity, least_square_diagonal (default) |

Options include `x0`, `rtol`, and `maxiter`. ILUT adds `drop_tolerance` and
`fill_factor`; incomplete Cholesky adds `initial_shift`. Results expose
`solution`, `info`, `iterations`, and estimated relative `error`. For multiple
RHS columns, diagnostics report the worst column. LSCG's error is measured on
the normal equations. Inspect `info` before using a nonconverged result.

`Preconditioner(a, method=...)` builds an independently reusable inverse
preconditioner with a `.solve(rhs)` method. `triangular_solve` handles lower,
upper, and unit-diagonal systems. `ordering` computes AMD/COLAMD/natural gather
permutations; pass them to `a.permuted`.

## Norms and randomized SVD

`norm(a)` computes a Euclidean vector norm or Frobenius matrix norm. Matrix
1/infinity norms are supported. Randomized SVD returns `(u, s, v)` and
reconstructs as `(u * s) @ v.conj().T`. It uses local random sampling; pass
`omega` to supply projection directions or `rng` for reproducible sampling.

## Solver reference

::: sppy.linalg.solvers
    options:
      members: [ComputationInfo, IterativeResult, SparseFactorization, SimplicialLLT, SimplicialLDLT, SparseLU, SparseQR, factorize, solve, ordering, triangular_solve, cg, bicgstab, lscg, Preconditioner]
      inherited_members: true

## Other functions

::: sppy.linalg.core
    options:
      members: [norm, rsvd]

::: sppy.linalg.general_linear_operator.GeneralLinearOperator
    options:
      members: [__init__, as_linear_operator, as_linear_operator_sum]
