# NumPy and SciPy interoperability

## NumPy

SpPy follows common NumPy naming and indexing conventions, but does not
implement the complete array-function or ufunc dispatch protocols. Prefer
methods such as `a.sum(axis=0)` and operators such as `a @ b` to keep the
conversion boundary explicit.

```python
import numpy as np
import sppy

a = sppy.asarray([[0, 2], [3, 0]], dtype=np.int64)
dense = np.asarray(a)
assert dense.dtype == np.int64
assert a.sum() == 5
```

Dense conversion preserves the stored dtype, including exact `int64` values.
`np.asarray(a)` always allocates dense storage. Other NumPy functions may also
coerce to dense arrays; do not assume they preserve sparsity.

Supported storage dtypes are signed 8/16/32/64-bit integers and 32/64-bit floats.
Complex64/128 storage is supported. Boolean and unsigned storage are unsupported. Arrays have one or two
dimensions. Individual dimensions and stored-entry indices use signed 32-bit
limits. General arithmetic broadcasting, whole-array boolean masks, new-axis
indexing, and the full set of NumPy reduction keywords are unsupported.

## SciPy sparse arrays and matrices

Install the `scipy` extra as described in [installation](install.md).
The default development environment already includes SciPy.

```python
from scipy import sparse
import sppy

source = sparse.csr_array([[0., 2.], [3., 0.]])
a = sppy.asarray(source)
csr = a.to_scipy_csr()
csc = a.to_scipy_csc()
```

Use the constructor or `asarray` for conversion into SpPy. There is no
`fromScipySparse` method. Conversion allocates new storage.

## Matrix Market files

```python
from tempfile import TemporaryDirectory
from pathlib import Path
import sppy
from sppy.io import mmread, mmwrite

with TemporaryDirectory() as directory:
    path = Path(directory) / "matrix.mtx"
    mmwrite(path, sppy.eye(3), comment="Example identity matrix")
    restored = mmread(path)
    assert restored.shape == (3, 3)
```

Matrix Market IO requires two-dimensional arrays and SciPy. The reader supports
comments, symmetry, and compressed input through SciPy. Integer files are loaded
as `int64`, real files as `float64`; complex files as `complex128`.
See the [IO reference](io.md) for arguments and storage selection.
