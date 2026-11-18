# Installation

SpPy requires Python 3.11 or newer, Eigen 5.0.1 or newer, and a C++17 compiler.
The native build uses scikit-build-core and CMake, with Cython 3.3.1+ and NumPy 2
headers installed automatically in an isolated build environment. CMake and Ninja
are provisioned by the backend when suitable tools are not installed.

Install uv using https://docs.astral.sh/uv/getting-started/installation/.
Download and extract Eigen from https://libeigen.gitlab.io/ separately.

## Creating the development environment

From a source checkout on Linux or macOS:

```sh
export EIGEN_INCLUDE_DIR=/path/to/eigen-5.0.1
uv sync --locked
uv run --locked pytest
```

`uv sync` creates `.venv` and installs the project in editable mode, along
with the default `dev` dependency group. No activation is needed for `uv run`.
The committed `uv.lock` records the resolved dependency versions;
`.python-version` selects Python 3.14. Override the interpreter with, for
example, `uv sync --locked --python 3.12`.

The Eigen include directory must contain `Eigen/Core`. Installed Eigen CMake
packages, standard system include directories, and `vendor/eigen` are searched
when `EIGEN_INCLUDE_DIR` is unset. On Windows, set the variable in PowerShell:

```powershell
$env:EIGEN_INCLUDE_DIR = "C:\path\to\eigen-5.0.1"
uv sync --locked
uv run --locked pytest
```

Use an MSVC developer shell with C++17 support. Linux is validated locally;
macOS and Windows have not been validated locally.

## Minimal environments and SciPy

To omit development dependencies:

```sh
uv sync --locked --no-dev
uv run --locked --no-dev python -c "import sppy; print(sppy.__version__)"
```

For SciPy interoperability and Matrix Market IO, add `--extra scipy` to sync
and run commands. The default development group already includes SciPy.

## Install with pip

uv is a development convenience, not a runtime dependency. From a source
checkout with Eigen configured, a regular installation is supported:

```sh
python -m pip install .
python -m pip install '.[scipy]'
```

The second command also installs SciPy. Source installations need the same
compiler and Eigen headers as development builds.

## Troubleshooting

- **Eigen not found:** point `EIGEN_INCLUDE_DIR` at the directory containing
  `Eigen/Core`, rather than the `Eigen` subdirectory itself. Use Eigen 5.0.1+.
- **Compiler errors:** check that a C++17 compiler is available. On Windows,
  use an MSVC developer shell.
- **SciPy import errors:** add `--extra scipy` to both `uv sync` and `uv run`
  when using `--no-dev`.
- **Stale native build:** run `uv sync --locked --reinstall-package sppy`
  after updating external Eigen headers.

For checks, OpenMP builds, release artifacts, and documentation previews,
see [contributing](contributing.md).
