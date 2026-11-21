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

## Publish documentation to GitHub Pages

In the repository's **Settings → Pages → Build and deployment**, select
**GitHub Actions** as the source. The `documentation` workflow builds MkDocs
on pull requests and publishes `site/` after pushes to `master`. It can also
be started manually from the Actions tab on `master`.

Do not select a branch's `/docs` directory: it contains MkDocs source, not
the generated website. That setting invokes Jekyll and can fail with
`No such file or directory @ dir_chdir0 - /github/workspace/docs` when the
selected publishing branch has no `docs/` directory. Publishing through
Actions replaces the old `gh-pages` branch deployment; no `mkdocs gh-deploy`
command is needed.

The workflow uploads the generated site and deploys it with GitHub's Pages
actions. Pull requests only validate the build and cannot deploy. The site is
published at <https://charanpald.github.io/sppy/>.

## Checks and release builds

With the default development environment:

```sh
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check sppy tests profiling
uv run --locked clang-format --dry-run --Werror sppy/native/*.h sppy/native/*.cpp
uv build
```

`uv build` writes the source distribution and wheel to `dist/`. Native
bindings are compiled from `sppy/native/` using pybind11 and Eigen. `setup.py` and `MANIFEST.in`
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

Python modules, classes, and functions (including private helpers and tests) must
have docstrings. Annotate function parameters and return values using standard
typing tools and `numpy.typing`; use `Any` for dynamic native bindings and
intentionally invalid test inputs where a narrower type would be misleading.
Ruff checks these requirements. The package includes a `py.typed` marker so type
checkers can use its inline annotations. Repository-specific development guidance
is in the root `AGENTS.md`.

The pytest suite lives in the repository's top-level `tests/` directory.
Tests use Arrange–Act–Assert sections, parametrized cases, isolated temporary
files, and local random generators. Run a focused subset with, for example,
`uv run pytest tests/test_numpy_compatibility.py`. Tests of the actual native extension boundary run with
`uv run pytest -m native`; no second compiler invocation is needed.
Numerical solver and storage contracts are in `tests/test_eigen_features.py`.

`sppy.test()` locates checkout tests and does not search installed packages.
Runtime wheels do not include tests or profiling tools; source archives do.

Opt-in profiling tools are in `profiling/`. For example:

```sh
uv run --locked python -m profiling.benchmark --operations all --size 1000 --output /tmp/sppy-benchmark.json
uv run --locked python -m profiling.profile --operation matmat --backend sppy_C --output /tmp/sppy.prof
```

See the [profiling guide](https://github.com/charanpald/sppy/blob/master/profiling/README.md) for workload settings, profile files, and benchmark
output. Importing these modules performs no profiling or input generation.

The native build directory is reused to speed incremental builds. To switch
an existing OpenMP build off, explicitly rebuild with `SPPY_OPENMP=0`.
Do not mutate a sparse array concurrently with a native operation on that same
array. Reusable direct factorization methods serialize access to their state.
