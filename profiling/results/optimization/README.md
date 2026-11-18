# Sparse optimization results

Paired local measurements on 18 November 2026, using the same Ryzen 5 5600X, Python 3.14.3, NumPy 2.5.5, SciPy 1.18.1, and one numerical thread. SpPy was built with OpenMP. Baseline Python files and the original native extension were preserved before editing; baseline subprocesses disabled the editable-install redirect and asserted that the preserved package was loaded.

Inputs, seeds, dtypes, index widths, sampling settings, and solver tolerances match in each pair. Input fingerprints and per-operation nnz were verified. Raw reports include interleaved SciPy controls and numerical checks. Gains below are before / after; values above one mean improvement. These are in-process synthetic measurements from one machine, not portable speed guarantees.

## Random 10,000 × 10,000 matrix

49,985 stored entries, float64. Median microseconds per call; SciPy columns show the final control measurement.

| Operation | C before µs | C after µs | C gain | SciPy CSR µs | F before µs | F after µs | F gain | SciPy CSC µs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| construct-triplets | 729.11 | 656.37 | 1.11× | 688.28 | 514.00 | 549.16 | 0.94× | 679.30 |
| construct-compressed | 799.13 | 104.77 | 7.63× | 24.03 | 642.26 | 105.03 | 6.11× | 23.50 |
| copy | 1075.68 | 14.20 | 75.73× | 26.65 | 925.24 | 14.75 | 62.72× | 25.66 |
| copy-assign | 1079.17 | 47.71 | 22.62× | 81.78 | 952.91 | 50.08 | 19.03× | 73.78 |
| gather | 19.74 | 19.17 | 1.03× | 12.40 | 19.43 | 19.95 | 0.97× | 12.89 |
| slice | 31404.69 | 229.19 | 137.02× | 251.65 | 30437.98 | 233.68 | 130.25× | 254.68 |
| transpose | 182.30 | 187.47 | 0.97× | 177.68 | 180.42 | 185.53 | 0.97× | 179.37 |
| adjoint | 187.78 | 182.45 | 1.03× | 215.11 | 183.10 | 181.62 | 1.01× | 211.49 |
| add | 260.57 | 262.11 | 0.99× | 248.34 | 279.36 | 278.86 | 1.00× | 246.36 |
| multiply | 264.60 | 281.34 | 0.94× | 249.31 | 263.35 | 280.09 | 0.94× | 250.74 |
| matvec | 78.26 | 65.58 | 1.19× | 62.87 | 74.83 | 65.70 | 1.14× | 56.91 |
| matmat | 652.59 | 259.40 | 2.52× | 206.03 | 669.27 | 237.78 | 2.81× | 195.32 |
| sparse-matmul | 14134.80 | 5191.66 | 2.72× | 2489.12 | 6217.02 | 5344.20 | 1.16× | 2553.81 |
| sum | 71.30 | 16.34 | 4.36× | 10.38 | 70.01 | 16.12 | 4.34× | 10.50 |
| column-sum | 219.08 | 157.52 | 1.39× | 88.46 | 231.01 | 170.08 | 1.36× | 162.86 |
| mean | 78.00 | 20.52 | 3.80× | 115.98 | 76.97 | 19.69 | 3.91× | 113.44 |
| norm | 59.43 | 8.54 | 6.96× | 6.24 | 58.54 | 8.47 | 6.91× | 6.09 |
| compressed-export | 24.59 | 12.74 | 1.93× | 12.79 | 24.76 | 12.66 | 1.96× | 12.74 |
| copy-prune | 911.85 | 78.45 | 11.62× | 92.67 | 767.55 | 83.71 | 9.17× | 92.42 |

## Changes and profiling evidence

- Same-dtype copies and order changes use native Eigen storage operations; copying no longer extracts coordinates and reassembles triplets.
- Canonical CSR/CSC imports validate and copy compressed buffers directly. Unsorted/duplicate inputs retain the summing assembly fallback.
- Basic integer/slice indexing uses compact native descriptors. Reversals, strides, empty axes, stored zeros, and axis dropping remain supported. Very large sparse vectors/rows can be sliced without allocating full-axis index arrays.
- Sparse matrix products retain the left storage order, avoiding the previous row-to-column-to-row conversions.
- Dense products write directly to NumPy output. Real blocks use contiguous row operations, while vectors and complex blocks retain Eigen specializations. Row blocks can use OpenMP.
- Stored values and compressed exports use contiguous copies when already compressed. Norms reduce contiguous values; uncompressed inputs keep their iterator-based fallback.

The slicing cProfile workload dropped from roughly 787,000 Python calls across five slices to 230. Copying dropped from roughly 1,720 calls across twenty copies to 260, with no triplet assembly in the copy profile. Profiler timings are not used as benchmark measurements.

Copies/imports/exports remain independent of their inputs. Explicit zeros and exact integer storage are preserved. General mixed/fancy slicing still has a Python fallback; this optimization targets basic slices.

## Other inputs

| Input | C copy gain | C slice gain | C dense-block gain | F dense-block gain |
| --- | ---: | ---: | ---: | ---: |
| random-1000-float64 | 30.69× | 292.03× | 1.41× | 1.56× |
| banded-10000-float64 | 62.55× | 409.01× | 1.91× | 2.41× |
| random-1000-complex128 | 27.27× | 299.14× | 1.04× | 0.95× |

## Remaining gaps

SciPy still leads several kernels, including large sparse products, compressed import, some dense products, and LU factorization/solve on these inputs. LU algorithms were not changed. Small differences near 1× should be interpreted with the raw timing distributions and SciPy control measurements. The retained Eigen vector/complex paths avoid the regressions observed during experiments with a generic block kernel.

## Raw paired reports

| Case | Before | After |
| --- | --- | --- |
| banded-10000-float64 | [table](before-banded-10000-float64.md) · [JSON](before-banded-10000-float64.json) | [table](after-banded-10000-float64.md) · [JSON](after-banded-10000-float64.json) |
| random-1000-complex128 | [table](before-random-1000-complex128.md) · [JSON](before-random-1000-complex128.json) | [table](after-random-1000-complex128.md) · [JSON](after-random-1000-complex128.json) |
| random-1000-float64 | [table](before-random-1000-float64.md) · [JSON](before-random-1000-float64.json) | [table](after-random-1000-float64.md) · [JSON](after-random-1000-float64.json) |
| random-10000-float64 | [table](before-random-10000-float64.md) · [JSON](before-random-10000-float64.json) | [table](after-random-10000-float64.md) · [JSON](after-random-10000-float64.json) |

[Machine-readable comparison](comparison.json). To reproduce the final side, use the same commands/settings in [the baseline report](../README.md) with the optimized checkout; compare the saved JSON fingerprints before comparing timings. For profile commands and timing scope, see [profiling methodology](../../README.md).

Validation: 2,224 tests pass, including all workload checks, owned copies, reversed/empty/large-axis slices, compressed-buffer validation, exact integer products, and real/complex threaded products. No machine-dependent speed thresholds were added to tests.
