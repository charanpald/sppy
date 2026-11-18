# SpPy

SpPy provides one- and two-dimensional sparse arrays backed by Eigen's C++
sparse matrix library. It stores entries and their coordinates instead of a
value for every position, making arrays with many zeros more economical.

Use it for sparse vectors and matrices, slicing, arithmetic, reductions,
and linear algebra with a NumPy-style interface.

```python
import numpy as np
import sppy

a = sppy.asarray([[0., 2.], [3., 0.]])
product = a @ a                   # Matrix multiplication
squared = a * a                   # Elementwise multiplication
column_sums = a.sum(axis=0)       # Dense result: [3., 2.]
dense = np.asarray(a)             # Explicitly allocate a dense array
```

Start with [installation](install.md) and the [quick start](quickstart.md),
then explore the [sparse array guide](sparseops.md). Existing users should read
[the 1.0 migration guide](migration.md).

## What to expect

- Signed integer storage: `int8`, `int16`, `int32`, and `int64`.
- Floating-point storage: `float32` and `float64`; complex: `complex64` and `complex128`.
- Row-major (`order="C"`) and column-major (`order="F"`) matrices.
- Optional [SciPy conversion and Matrix Market IO](interoperability.md).
- [Linear solvers, Frobenius norms, and randomized SVD](linalg.md).

SpPy requires Python 3.11+, NumPy 2, Eigen 5.0.1+, and a C++17 compiler to build.
It supports vectors and matrices, rather than arbitrary-dimensional tensors.
Unsigned and boolean storage are unsupported. NumPy compatibility is
partial; see [interoperability and limits](interoperability.md).

See [Eigen coverage](eigen-coverage.md) for supported modules and boundaries.

## Project and support

Report bugs and request features in the
[GitHub issue tracker](https://github.com/charanpald/sppy/issues).
For build problems, include your operating system, Python/compiler versions,
Eigen location, command, and full error output.
See [contributing](contributing.md) for validation and documentation commands.
