# Contributing

Set up the [development environment](install.md) before running package checks.

## Build and preview the documentation

The documentation uses MkDocs and mkdocstrings. API pages are extracted from
Python source without importing native extensions, so documentation-only work
does not require Eigen, a compiler, or an installed SpPy package.

```sh
uv sync --locked --only-group docs
uv run --locked --only-group docs mkdocs serve
```

Open the local address printed by MkDocs (normally `http://127.0.0.1:8000`).
The preview reloads when documentation changes. To build the static site:

```sh
uv run --locked --only-group docs mkdocs build --strict
```

Output is written to `site/`. Strict builds fail on warnings, including broken
page links and anchors, and run in CI. Navigation lives in `mkdocs.yml`; content
lives in `docs/*.md`. API reference directives read the public Python source.
Use relative Markdown links between pages and runnable Python code blocks.

`--only-group docs` installs only documentation dependencies. Run
`uv sync --locked` again before working on native code or running tests.
To keep both environments installed, use `uv sync --locked --group docs` and
`uv run --locked --group docs mkdocs serve` with Eigen configured.

## Checks and release builds

With the default development environment:

```sh
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check sppy tests profiling
uv run --locked clang-format --dry-run --Werror sppy/include/*.h tests/native/*.cpp profiling/native/*.cpp
uv build
```

`uv build` writes the source distribution and wheel to `dist/`. Native
binding generation runs in the build directory. `setup.py` and `MANIFEST.in`
are no longer needed; package and archive settings live in `pyproject.toml`.

After editing dependency declarations, run `uv lock` and commit the lockfile.
After changing external Eigen headers, force a rebuild:

```sh
uv sync --locked --reinstall-package sppy
```

For optional OpenMP:

```sh
export SPPY_OPENMP=1
uv sync --locked --reinstall-package sppy
uv run --locked pytest
uv build
```

The compiler and linker must support OpenMP and its runtime must be available.
Keep `SPPY_OPENMP` set for subsequent commands. Alternatively, release builds
accept `uv build -Ccmake.define.SPPY_OPENMP=ON`. Without OpenMP, parallel kernels
run serially.

## Tests and profiling layout

The pytest suite lives in the repository's top-level `tests/` directory.
Tests use Arrange–Act–Assert sections, parametrized cases, isolated temporary
files, and local random generators. Run a focused subset with, for example,
`uv run pytest tests/test_numpy_compatibility.py`. Native C++ tests are run by
pytest when a GCC/Clang compiler and Eigen headers are available; select them
with `uv run pytest -m native`. Missing native prerequisites are reported as
skips.

`sppy.test()` locates checkout tests and does not search installed packages.
Runtime wheels do not include tests or profiling tools; source archives do.

Opt-in profiling tools are in `profiling/`. For example:

```sh
uv run python -m profiling.sparse_array --operation dot --size 1000
uv run python -m profiling.linear_operator --size 1000 --columns 8
uv run python -m profiling.matrix_market --size 1000
uv run python -m profiling.benchmark --operation dot --size 1000
```

See the [profiling guide](https://github.com/charanpald/sppy/blob/master/profiling/README.md) for workload settings, profile files, and benchmark
output. Importing these modules performs no profiling or input generation.
