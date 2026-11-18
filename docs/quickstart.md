# Quick start

After [installing SpPy](install.md), run these examples in Python.

## Create an array

Use `asarray` to convert data and `zeros` to create an empty sparse array.
The conversion infers the dtype; pass `dtype` explicitly when needed.

```python
import numpy as np
import sppy

a = sppy.asarray([[0., 2., 0.], [3., 0., 4.]])
assert a.shape == (2, 3)
assert a.ndim == 2
assert a.size == 6
assert a.nnz == 3

b = sppy.zeros((2, 3), dtype=np.float64)
b[0, 1] = 2.
b[1, [0, 2]] = [3., 4.]
np.testing.assert_array_equal(a.toarray(), b.toarray())
```

`size` counts all positions, while `nnz` counts stored entries. Explicitly
stored zeros can contribute to `nnz`; use `count_nonzero()` to count nonzero values.

## Select and update values

```python
row = a[0]                       # Sparse vector with shape (3,)
column = a[:, 1]                 # Sparse vector with shape (2,)
block = a[:, 1:]                 # Sparse matrix with shape (2, 2)
a[-1, 0] = 5.
assert a[1, 0] == 5.
```

Slices are independent copies. Assign through `a[...]` to change the original.

## Compute with sparse arrays

```python
squared = a * a
product = a @ a.T
vector_product = a @ np.ones(3)
row_totals = a.sum(axis=1)
means = a.mean(axis=0, keepdims=True)

assert product.shape == (2, 2)
np.testing.assert_array_equal(vector_product, [2., 9.])
np.testing.assert_array_equal(row_totals, [2., 9.])
```

`*` multiplies elementwise; `@` performs matrix multiplication. Reductions and
products with dense vectors return dense results. For larger workloads, read
[the sparse array guide](sparseops.md) before converting to dense arrays.
