# Compressed import and sparse multiplication

Measured on 19 November 2026, AMD Ryzen 5 5600X, Linux x86-64,
Python 3.14.3, NumPy 2.5.5, SciPy 1.18.1, Eigen 5.0.1, OpenMP build.
Every backend used one numerical thread, nine samples, and at least 20 ms
per timed batch. These are synthetic local measurements, not portable rankings.

The baseline uses the original Python package and native extension copied before
these changes. The optimized runs use the rebuilt extension. Both use the same
benchmark harness, configuration and input fingerprint. The saved baseline is
not reconstructed from an earlier historical timing. All numerical validations
passed. Raw JSON records the environment, input hashes, samples and validation;
its Git revision describes the checkout, not the separately preserved extension.
No compilation or tests ran alongside these measurements.

## Changes

- Compressed import checks pointer bounds before reading offsets, then scans
  index bounds and within-segment ordering using accumulated error flags. This
  removes per-entry exception branches and permits vectorization. Input validation,
  owned copies, unsorted input and duplicate summation remain supported.
- Sparse multiplication accumulates directly by CSR row or CSC column, writes
  compressed output in growing buffers, and skips sorting when traversal already
  produces sorted indices (common for banded inputs). Otherwise, two storage-order
  scatters canonicalize the result in linear time. A prototype that sorted each
  output vector was slower on random inputs and was replaced before these runs.
- The dense scratch accumulator is capped at 1,048,576 inner positions. Larger
  axes use a hash accumulator and sort only touched positions, avoiding allocation
  proportional to an enormous mostly empty axis. Output buffers and the scatter
  temporaries still scale with stored output entries.
- All eight scalar types retain explicit zeros, cancellation entries, independent
  output storage and sorted unique indices. Integer multiply/add uses defined
  fixed-width modular arithmetic, including int64 values beyond float precision.

## Paired timings

Medians in microseconds; speedup is baseline / optimized.

| Input | Order | Import before | Import after | Speedup | Product before | Product after | Speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random 1,000 float64 | C | 9.05 | 8.04 | 1.13× | 240.80 | 248.75 | 0.97× |
| Random 1,000 float64 | F | 9.15 | 8.26 | 1.11× | 407.37 | 248.18 | 1.64× |
| Random 10,000 float64 | C | 103.76 | 75.48 | 1.37× | 4371.65 | 4150.10 | 1.05× |
| Random 10,000 float64 | F | 101.27 | 75.11 | 1.35× | 4777.94 | 4249.70 | 1.12× |
| Banded 10,000 float64 | C | 73.54 | 43.28 | 1.70× | 736.74 | 574.78 | 1.28× |
| Banded 10,000 float64 | F | 73.60 | 46.06 | 1.60× | 1459.63 | 572.10 | 2.55× |
| Random 1,000 complex128 | C | 9.84 | 9.00 | 1.09× | 275.96 | 285.35 | 0.97× |
| Random 1,000 complex128 | F | 10.16 | 9.12 | 1.11× | 333.69 | 283.06 | 1.18× |

Small random C-order products are about 3% slower in this run. Larger random
products improve by 5–12%; the strongest product gains are on banded inputs,
particularly F order. The speedup depends on sparsity and storage order.

## SciPy comparison after optimization

Medians in microseconds. Sparse multiplication includes canonical sorted output
for SpPy; SciPy may return unsorted inner indices. Compressed import includes
SpPy's full bounds and ordering validation. These are the public API workloads
rather than identical internal kernel work.

| Input | Operation | SpPy C | SciPy CSR | SpPy F | SciPy CSC |
| --- | --- | ---: | ---: | ---: | ---: |
| Random 1,000 float64 | construct-compressed | 8.04 | 9.70 | 8.26 | 9.75 |
| Random 1,000 float64 | sparse-matmul | 248.75 | 217.22 | 248.18 | 217.33 |
| Random 10,000 float64 | construct-compressed | 75.48 | 23.14 | 75.11 | 22.93 |
| Random 10,000 float64 | sparse-matmul | 4150.10 | 2466.41 | 4249.70 | 2495.95 |
| Banded 10,000 float64 | construct-compressed | 43.28 | 23.30 | 46.06 | 22.84 |
| Banded 10,000 float64 | sparse-matmul | 574.78 | 542.21 | 572.10 | 543.76 |
| Random 1,000 complex128 | construct-compressed | 9.00 | 11.05 | 9.12 | 11.19 |
| Random 1,000 complex128 | sparse-matmul | 285.35 | 231.65 | 283.06 | 233.09 |

SciPy remains faster for larger imports and these sparse products. Small imports
are faster in SpPy; banded products are now within roughly 5–6% of SciPy.

## Reports and reproduction

- Random 1,000 float64: [before table](before-random-1000.md), [before JSON](before-random-1000.json), [after table](after-random-1000.md), [after JSON](after-random-1000.json).
- Random 10,000 float64: [before table](before-random-10000.md), [before JSON](before-random-10000.json), [after table](after-random-10000.md), [after JSON](after-random-10000.json).
- Banded 10,000 float64: [before table](before-banded-10000.md), [before JSON](before-banded-10000.json), [after table](after-banded-10000.md), [after JSON](after-banded-10000.json).
- Random 1,000 complex128: [before table](before-complex-1000.md), [before JSON](before-complex-1000.json), [after table](after-complex-1000.md), [after JSON](after-complex-1000.json).

Run from the development environment after building the extension:

```sh
.venv/bin/python -m profiling.benchmark --operations construct-compressed sparse-matmul --size 10000 --entries 50000 --threads 1 --repeats 9 --min-time 0.02 --output /tmp/import-product.json --markdown /tmp/import-product.md
```

Use `--pattern banded --bandwidth 2` for the banded case; use `--size 1000
--entries 5000` for the smaller case and add `--dtype complex128` for complex
values. To reproduce the baseline, build revision
`1baf1eb504bef725037c88da5504e744a0349310` separately and run the same
harness/configuration there. See the [methodology](../../README.md).

Validation: 2,316 tests passed, including all dtypes/orders, rectangular mixed
orders, uncompressed sources, cancellation and stored zeros, canonical output,
integer overflow, complex products, empty dimensions, malformed import buffers
and huge inner axes. Ruff, clang-format, strict MkDocs and package builds were
also checked. No runtime SciPy dependency was added.
