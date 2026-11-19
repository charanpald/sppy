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

## Performance versus SciPy sparse

Local synthetic benchmarks compare all **24 workloads** in C/CSR and F/CSC
storage. Selected **SpPy C versus SciPy CSR** medians are below; speedup is
SciPy time / SpPy time, so **above 1× favors SpPy**.

| Input | Operation | SpPy C | SciPy CSR | SpPy speedup |
| --- | --- | ---: | ---: | ---: |
| Random 1,000 float64 | gather | 5.38 µs | 12.66 µs | 2.35× |
| Random 10,000 float64 | construct-compressed | 30.32 µs | 23.11 µs | 0.76× |
| Random 10,000 float64 | column-sum | 59.45 µs | 86.04 µs | 1.45× |
| Random 10,000 float64 | sparse-matmul | 2.25 ms | 2.57 ms | 1.14× |
| Banded 10,000 float64 | sparse-matmul | 481.96 µs | 575.98 µs | 1.20× |
| Random 1,000 complex128 | matvec | 6.13 µs | 7.11 µs | 1.16× |
| Random 1,000 complex128 | matmat | 20.64 µs | 40.58 µs | 1.97× |
| Random 1,000 complex128 | lu-factor | 99.31 ms | 80.12 ms | 0.81× |
| Random 1,000 complex128 | lu-solve | 2.98 ms | 3.11 ms | 1.04× |
| Random 1,000 complex128 | bicgstab | 629.08 µs | 675.26 µs | 1.07× |

Measured on 19 November 2026 on a Ryzen 5 5600X, with one numerical thread,
seven samples, Python 3.14.3, NumPy 2.5.5, SciPy 1.18.1 and Eigen 5.0.1.
Runtime AVX2/FMA was enabled on this CPU; portable fallbacks remain available.
Array inputs have about five stored entries per row. Solver inputs are derived
Hermitian positive definite systems; LU solves reuse prepared factors.

SpPy still trails SciPy in some workloads, including validated large imports
and complex LU factorization. Sparse output contracts and algorithms differ,
and these timings are not portable performance guarantees. See the
[complete results, F/CSC comparisons and remaining gaps](profiling/results/remaining-gaps/README.md)
and [benchmark methodology/reproduction](profiling/README.md) for raw samples,
validation and commands to measure your own inputs.

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
