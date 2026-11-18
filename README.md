# SpPy

NumPy-style sparse vectors and matrices backed by Eigen 5, with native real and
complex arithmetic, sparse factorizations, iterative solvers, and preconditioners.

```python
import numpy as np
import sppy
from sppy.linalg import SparseLU, cg

a = sppy.asarray([[4., 1.], [1., 3.]])
product = a @ a
squared = a * a
column_sums = a.sum(axis=0)
x = SparseLU(a).solve(np.array([1., 2.]))
result = cg(a, np.array([1., 2.]), preconditioner="incomplete_cholesky")
```

Version **1.0** removes legacy aliases and standardizes `order="C"/"F"`.
Read the [migration guide](docs/migration.md) before upgrading.

## Installation

Requires Python 3.11+, NumPy 2, a C++17 compiler, and
[Eigen 5.0.1+ headers](https://libeigen.gitlab.io/). Install
[uv](https://docs.astral.sh/uv/getting-started/installation/), then from a checkout:

```sh
export EIGEN_INCLUDE_DIR=/path/to/eigen-5.0.1
uv sync --locked
uv run --locked python -c "import sppy; print(sppy.eye(3).toarray())"
```

`EIGEN_INCLUDE_DIR` must contain `Eigen/Core`. On Windows use an MSVC developer
PowerShell session and `$env:EIGEN_INCLUDE_DIR = "C:\path\to\eigen-5.0.1"`.
`uv sync` creates an editable development environment using the committed lockfile.
Python 3.11–3.14 are covered by Linux CI; `.python-version` selects 3.14.

For a minimal installation use `uv sync --locked --no-dev`. Add `--extra scipy`
when you need SciPy conversion or Matrix Market IO. SciPy is optional for all
native Eigen solvers. A regular source installation also works with
`python -m pip install .` after configuring Eigen.

See [installation and troubleshooting](docs/install.md) for details.

## Features and documentation

- Signed int8/16/32/64, float32/64, and complex64/128 sparse arrays.
- C/F storage, triplet/CSR/CSC assembly, slicing, pruning, and storage controls.
- Sparse arithmetic, dense/sparse products, adjoints, triangular and Hermitian operations.
- LLT/LDLT, LU, QR, CG, BiCGSTAB, and least-squares CG.
- Symbolic factor reuse, native preconditioners, orderings, and solver diagnostics.

Read the [documentation home](docs/index.md), [quick start](docs/quickstart.md),
[array guide](docs/sparseops.md), [solver reference](docs/linalg.md), and
[Eigen coverage and boundaries](docs/eigen-coverage.md). History is in the
[changelog](docs/changelog.md).

Preview or build the documentation without compiling SpPy:

```sh
uv run --locked --only-group docs mkdocs serve
uv run --locked --only-group docs mkdocs build --strict
```

## Development

```sh
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check sppy tests profiling
uv run --locked clang-format --dry-run --Werror sppy/native/*.h sppy/native/*.cpp
uv build
```

Enable OpenMP with `SPPY_OPENMP=1` before syncing or building. Keep it set on
subsequent uv commands. The compiler and runtime must support OpenMP; kernels
run serially otherwise. `set_num_threads` and `get_num_threads` control Eigen's
thread setting; `openmp_enabled` reports build support.

After updating dependencies, run `uv lock`. After changing external Eigen
headers, rebuild with `uv sync --locked --reinstall-package sppy`.
See [contributing](docs/contributing.md) and [profiling](profiling/README.md).
