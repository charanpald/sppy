# SpPy

SpPy provides NumPy-style sparse arrays backed by Eigen's C++ sparse matrix
library. It supports sparse vectors and matrices, arithmetic, slicing, reductions,
and linear algebra operations.

## Quick start

```python
import numpy as np
import sppy

a = sppy.asarray([[0., 2.], [3., 0.]])
b = a.astype(np.float32)
product = a @ a
elementwise = a * a
column_sums = np.sum(a, axis=0)
dense = np.asarray(a)
```

See the [migration guide](docs/migration.md) for Pythonic names, compatibility
aliases, behavior changes, and NumPy compatibility limits.

## Installation with uv

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and a C++17
compiler (GCC, Clang, or MSVC). Download and extract
[Eigen 5.0.1 or newer](https://libeigen.gitlab.io/); it is a separate header-only
C++ dependency. `EIGEN_INCLUDE_DIR` must point to the directory containing
`Eigen/Core`.

From the repository root on Linux or macOS:

```sh
export EIGEN_INCLUDE_DIR=/path/to/eigen-5.0.1
uv sync --locked
uv run --locked python -c "import sppy; print(sppy.eye(3).toarray())"
```

`uv sync` creates `.venv`, installs this package in editable mode, and installs
NumPy, SciPy, pytest, and the development tools at the versions in `uv.lock`.
No manual environment activation is required. The `.python-version` file selects
Python 3.14; uv can download it if needed. Python 3.11–3.14 are covered by CI;
to use another supported interpreter, run `uv sync --locked --python 3.12`.

On Windows, run from a PowerShell session with the MSVC C++ build tools available:

```powershell
$env:EIGEN_INCLUDE_DIR = "C:\path\to\eigen-5.0.1"
uv sync --locked
uv run --locked python -c "import sppy; print(sppy.eye(3).toarray())"
```

Installed Eigen CMake packages, standard system include directories, and
`vendor/eigen` are also searched. macOS and Windows have not been validated
locally.

For a minimal environment without development dependencies:

```sh
uv sync --locked --no-dev
uv run --locked --no-dev python -c "import sppy; print(sppy.__version__)"
```

Add `--extra scipy` to both commands when you need SciPy interoperability or
Matrix Market IO. SciPy is included in the default development environment.
Python build dependencies (Cython, NumPy headers, and scikit-build-core) are
installed in an isolated build environment. CMake and Ninja are provisioned by
the backend if suitable tools are not already installed.

### Development and validation

```sh
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check sppy tests profiling
uv run --locked clang-format --dry-run --Werror sppy/include/*.h tests/native/*.cpp profiling/native/*.cpp
uv build
```

The native extensions build with CMake and scikit-build-core, configured by
`pyproject.toml` and `CMakeLists.txt`. Generated Cython sources stay in the build
directory. `uv build` creates a source distribution and wheel in `dist/`.

After changing dependency declarations, run `uv lock` and commit `uv.lock`.
To rebuild native code or pick up externally updated Eigen headers, run:

```sh
uv sync --locked --reinstall-package sppy
```

To enable OpenMP, the compiler and linker must support it and its runtime must
be installed. Set `SPPY_OPENMP=1` before syncing or building:

```sh
SPPY_OPENMP=1 uv sync --locked --reinstall-package sppy
SPPY_OPENMP=1 uv build
```

Keep the variable set for subsequent `uv run` commands to retain that build
configuration. Without OpenMP, parallel kernels run serially. The CMake option
`-Ccmake.define.SPPY_OPENMP=ON` is also available with `uv build`.

For a regular non-editable installation, pip remains supported:

```sh
python -m pip install .
```

Set `EIGEN_INCLUDE_DIR` first as above. See the
[installation guide](docs/install.md) for more details.

## Tests and profiling

The pytest suite is in `tests/`, with focused, parametrized tests following
Arrange–Act–Assert (AAA). Tests use local random generators and temporary files;
they do not change application logging or NumPy print settings. Native C++
contracts in `tests/native/` are compiled and run by pytest when GCC/Clang and
Eigen headers are available; otherwise those tests are reported as skipped.

```sh
uv run --locked pytest
uv run --locked pytest tests/test_numpy_compatibility.py
uv run --locked pytest -m native
```

`sppy.test()` remains a convenience wrapper for running checkout tests. Tests
and profiling tools ship in source distributions, not runtime wheels.

Opt-in profiling commands live in `profiling/` and run only when invoked:

```sh
uv run python -m profiling.sparse_array --operation dot --size 1000 --entries 5000
uv run python -m profiling.linear_operator --size 1000 --columns 8
uv run python -m profiling.matrix_market --size 1000 --entries 5000
uv run python -m profiling.benchmark --operation dot --size 1000
```

See [profiling/README.md](profiling/README.md) for saved profiles, JSON timings,
and the native Eigen benchmark. The scripts use standard-library profiling,
configurable workloads, and deterministic inputs without external research
project dependencies or import-time execution.

## Documentation

Read the [documentation home](docs/index.md), [quick start](docs/quickstart.md),
[sparse array guide](docs/sparseops.md), and [API reference](docs/csarray.md).
Release history is in the [changelog](docs/changelog.md).

Preview or build the MkDocs site without compiling SpPy:

```sh
uv sync --locked --only-group docs
uv run --locked --only-group docs mkdocs serve
uv run --locked --only-group docs mkdocs build --strict
```

See [contributing](docs/contributing.md) for documentation and development details.
