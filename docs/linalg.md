# Linear algebra

Import functions from `sppy.linalg`. The solver uses floating-point sparse
matrices; randomized SVD also accepts dense arrays and general linear operators.

## Solve a linear system

```python
import numpy as np
import sppy
from sppy.linalg import bicgstab, norm

a = sppy.asarray([[4., 1.], [1., 3.]])
b = np.array([1., 2.])
x, status = bicgstab(a, b, rtol=1e-10)
assert status == 0
np.testing.assert_allclose(a @ x, b)
frobenius = norm(a)
```

Always inspect `status`: 0 means success, 1 a numerical issue, 2 no convergence,
and 3 invalid input. These are SpPy's codes and differ from SciPy's convention.
`norm` computes only the Frobenius norm; its `ord` argument is currently ignored.

## Randomized SVD

```python
from sppy.linalg import rsvd

u, singular_values, v = rsvd(a, k=1, p=1, q=2)
approximation = (u * singular_values) @ v.T
assert approximation.shape == a.shape
```

`rsvd` returns right singular vectors as columns of `v`, rather than returning
`v.T`. Increase `p` (oversampling) or `q` (power iterations) to improve the
approximation. Sampling uses NumPy's global random state.

## Functions

::: sppy.linalg.core
    options:
      members: [bicgstab, norm, rsvd, biCGSTAB]

## General linear operators

Wrap a matrix with `GeneralLinearOperator.as_linear_operator(a)` to expose
`matvec`, `rmatvec`, `matmat`, and `rmatmat` callbacks. Combine operators lazily
with `as_linear_operator_sum(left, right)`.

::: sppy.linalg.general_linear_operator.GeneralLinearOperator
    options:
      members: [__init__, as_linear_operator, as_linear_operator_sum]
