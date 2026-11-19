# Baseline SciPy comparisons

See the [full 24-operation optimization audit](all-operations/README.md) for the
latest complete before/after results, first-solve checks and remaining gaps.

See the [compressed-import and sparse-product follow-up](import-product/README.md)
for the latest targeted optimization measurements.

These are the original unoptimized measurements. See the
[paired optimization follow-up](optimization/README.md) for subsequent code changes and results.

Recorded on **18 November 2026**, on an AMD Ryzen 5 5600X, Linux x86-64,
Python 3.14.3, NumPy 2.5.5, SciPy 1.18.1, and SpPy 1.0.0. The SpPy extension
was built with OpenMP; **all four backends were limited to one numerical thread**.
Each JSON file records library/threadpool details, revision/dirty state, seed,
input fingerprint, per-operation nnz, sample order, raw timings, and validation.

These are local synthetic measurements from one machine and one run per
configuration. They show where to investigate, rather than establishing a
portable performance ranking. No speed thresholds are enforced by tests.

| Input | Workloads | Samples per backend | Readable table | Raw samples |
| --- | --- | ---: | --- | --- |
| Random 1,000 × 1,000, 4,993 nnz, float64 | All 24 | 9 | [table](random-1000-float64.md) | [JSON](random-1000-float64.json) |
| Random 10,000 × 10,000, 49,985 nnz, float64 | 19 array operations | 7 | [table](random-10000-float64.md) | [JSON](random-10000-float64.json) |
| Banded 10,000 × 10,000, 49,994 nnz, float64 | All 24 | 7 | [table](banded-10000-float64.md) | [JSON](banded-10000-float64.json) |
| Random 1,000 × 1,000, 4,993 nnz, complex128 | All 24 | 7 | [table](random-1000-complex128.md) | [JSON](random-1000-complex128.json) |

Solver workloads use a derived Hermitian positive definite system, with a
separate nnz recorded in JSON. All recorded solver relative residuals were below **9.8e-9**. Every numerical
comparison passed.

## Selected results

The following are medians for SpPy C order and SciPy CSR. See the full tables
for F order/CSC, the complex case, and the remaining operations.

| Input | Operation | SpPy C | SciPy CSR | Observation |
| --- | --- | ---: | ---: | --- |
| Random 1,000 | CG | 238.80 µs | 397.53 µs | SpPy 1.66× faster |
| Random 1,000 | Triangular solve | 74.48 µs | 345.42 µs | SpPy 4.64× faster |
| Random 1,000 | LU factorization | 44.21 ms | 39.68 ms | SciPy 1.11× faster |
| Random 10,000 | Copy | 873.31 µs | 24.51 µs | SciPy 35.63× faster |
| Random 10,000 | Stepped slice | 30.63 ms | 250.11 µs | SciPy 122.46× faster |
| Random 10,000 | Sparse/dense matrix product | 651.80 µs | 202.80 µs | SciPy 3.21× faster |
| Banded 10,000 | CG | 1.37 ms | 1.32 ms | Similar; SciPy slightly faster in this run |

The CG advantage on the smaller random system does not carry over uniformly
to the larger banded system. Storage order matters too: the random 10,000
sparse matrix product took 14.56 ms in SpPy C order and 5.34 ms in F order,
compared with roughly 2.5 ms for either SciPy representation.

## Profiling findings

Additional cProfile runs used the random 10,000-row input:

- **Slicing:** five calls produced about 787,000 Python calls. Almost all
  cumulative time was in `SparseArray.__getitem__`, including dictionary lookups,
  list appends, and Python iteration. This identifies native handling of basic
  slices as a useful follow-up optimization.
- **Copying:** `SparseArray.copy` routes through `astype`, coordinate extraction,
  and triplet assembly. Native `assemble` accounted for about 0.023 of the
  0.028 profiled seconds across 20 copies. A direct storage-copy path is a
  useful follow-up optimization.

These are profiling observations, not changes to SpPy's production kernels.
cProfile overhead makes its absolute times unsuitable for comparison with the
benchmark table. Binary profiles were saved to `/tmp/sppy-slice.prof` and
`/tmp/sppy-copy.prof` for this session; reproduce them with the commands below.

## Reproduce

After syncing the development environment with Eigen configured:

```sh
uv run --locked python -m profiling.benchmark --size 1000 --entries 5000 --operations all --repeats 9 --min-time 0.02 --output /tmp/random-1000-float64.json --markdown /tmp/random-1000-float64.md
uv run --locked python -m profiling.benchmark --size 10000 --entries 50000 --operations arrays --repeats 7 --output /tmp/random-10000-float64.json --markdown /tmp/random-10000-float64.md
uv run --locked python -m profiling.benchmark --pattern banded --size 10000 --bandwidth 2 --operations all --repeats 7 --output /tmp/banded-10000-float64.json --markdown /tmp/banded-10000-float64.md
uv run --locked python -m profiling.benchmark --size 1000 --entries 5000 --dtype complex128 --operations all --repeats 7 --output /tmp/random-1000-complex128.json --markdown /tmp/random-1000-complex128.md
uv run --locked python -m profiling.profile --operation slice --backend sppy_C --size 10000 --entries 50000 --repeats 5 --output /tmp/sppy-slice.prof
uv run --locked python -m profiling.profile --operation copy --backend sppy_C --size 10000 --entries 50000 --repeats 20 --output /tmp/sppy-copy.prof
```

The recorded commands used `.venv/bin/python` directly in the already prepared
environment, avoiding an unrelated environment sync during measurement.
Input generation, validation, thread controls, and sampling follow the
[benchmark methodology](../README.md).
