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

Local synthetic benchmarks cover **24 operations** in C/CSR and F/CSC storage.
The table below lists common sparse-matrix operations on the same random
**1,000 × 1,000 float64 matrix with 4,993 stored entries**, comparing
**SpPy C versus SciPy CSR**. Times are median **microseconds per call**;
speedup is SciPy time / SpPy time, so **above 1× favors SpPy**.

| Operation | SpPy C (µs) | SciPy CSR (µs) | SpPy speedup |
| --- | ---: | ---: | ---: |
| COO / triplet construction | 47.91 | 68.73 | 1.43× |
| CSR / CSC construction | 6.69 | 10.11 | 1.51× |
| Copy (`A.copy()`) | 2.16 | 11.07 | 5.12× |
| Paired-coordinate indexing | 5.38 | 12.66 | 2.35× |
| Sparse slicing (`A[::2, ::2]`) | 10.23 | 53.53 | 5.23× |
| Transpose (`A.T`, owned result) | 10.76 | 37.41 | 3.48× |
| Sparse addition (`A + A`) | 10.17 | 32.70 | 3.22× |
| Elementwise multiplication (`A * A`) | 10.65 | 36.15 | 3.4× |
| Sparse × dense vector (`A @ x`) | 3.99 | 4.43 | 1.11× |
| Sparse × dense matrix (`A @ X`) | 9.55 | 18.16 | 1.9× |
| Sparse × sparse matrix (`A @ A.H`) | 177.39 | 220.75 | 1.24× |
| Total sum (`A.sum()`) | 1.61 | 3.47 | 2.15× |
| Column sums (`A.sum(axis=0)`) | 3.43 | 28.17 | 8.2× |
| Mean (`A.mean()`) | 4.13 | 28.74 | 6.95× |
| Frobenius norm (`linalg.norm(A)`) | 0.67 | 2.10 | 3.11× |
| LU factorization | 41721.38 | 40146.00 | 0.962× |
| LU solve (reused factors) | 1413.42 | 1707.33 | 1.21× |
| Conjugate gradient (CG) | 135.42 | 415.49 | 3.07× |
| BiCGSTAB | 163.54 | 527.50 | 3.23× |
| Triangular solve | 18.67 | 343.83 | 18.4× |

Dense matrix products and reused-factor LU solves use eight RHS columns.
Transpose timings include an owned result in the input storage order; SciPy's
transpose view is materialized. Sparse matrix multiplication excludes preparation
of `A.H`. Iterative solves include diagonal-preconditioner setup; LU factorization
and reused-factor solves are timed separately. All results are checked numerically.

See the [full 1,000-row report](profiling/results/remaining-gaps/after-random-1000.md)
for all operations and F/CSC timings, and the
[10,000-row float64](profiling/results/remaining-gaps/after-random-10000.md),
[banded](profiling/results/remaining-gaps/after-banded-10000.md), and
[complex128](profiling/results/remaining-gaps/after-complex-1000.md) reports
for other sizes, patterns, and dtypes.

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
uv run --locked mypy
uv run --locked python -m mypy.stubtest sppy._native --allowlist tools/native-stubtest-allowlist.txt
uv run --locked ruff check .
uv run --locked ruff format --check sppy tests profiling tools
uv run --locked clang-format --dry-run --Werror sppy/native/*.h sppy/native/*.cpp
uv build
uv run --locked python tools/validate_wheel.py
```

Enable OpenMP with `SPPY_OPENMP=1` before syncing or building. Keep it set on
subsequent uv commands. The compiler and runtime must support OpenMP; kernels
run serially otherwise. `set_num_threads` and `get_num_threads` control Eigen's
thread setting; `openmp_enabled` reports build support.

After updating dependencies, run `uv lock`. After changing external Eigen
headers, rebuild with `uv sync --locked --reinstall-package sppy`.
See [contributing](docs/contributing.md) and [profiling](profiling/README.md).
