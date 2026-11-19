# SpPy versus SciPy sparse

AMD Ryzen 5 5600X 6-Core Processor · Python 3.14.3 · NumPy 2.5.5 · SciPy 1.18.1 · SpPy 1.0.0

Pattern: random; shape: [10000, 10000]; nnz: 49985; dtype: float64; threads: 1; samples: 7.

Median microseconds per call. Ratios are SciPy / SpPy in matching storage order; **above 1 means SpPy is faster**.

| Operation | SpPy C µs | SciPy CSR µs | CSR / C | SpPy F µs | SciPy CSC µs | CSC / F |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| construct-triplets | 596.78 | 689.38 | 1.16× | 593.06 | 678.31 | 1.14× |
| construct-compressed | 35.34 | 23.20 | 0.656× | 34.84 | 23.67 | 0.679× |
| copy | 14.03 | 24.70 | 1.76× | 14.07 | 24.90 | 1.77× |
| copy-assign | 41.79 | 66.95 | 1.6× | 42.97 | 68.13 | 1.59× |
| gather | 5.40 | 12.31 | 2.28× | 5.67 | 12.62 | 2.23× |
| slice | 223.42 | 253.41 | 1.13× | 233.88 | 248.27 | 1.06× |
| transpose | 184.56 | 174.96 | 0.948× | 184.08 | 175.50 | 0.953× |
| adjoint | 181.51 | 207.55 | 1.14× | 181.26 | 209.53 | 1.16× |
| add | 93.96 | 242.80 | 2.58× | 116.14 | 242.01 | 2.08× |
| multiply | 93.63 | 244.11 | 2.61× | 115.97 | 242.49 | 2.09× |
| matvec | 64.07 | 61.85 | 0.965× | 62.05 | 56.25 | 0.906× |
| matmat | 137.61 | 205.98 | 1.5× | 150.99 | 195.91 | 1.3× |
| sparse-matmul | 3856.10 | 2431.87 | 0.631× | 3876.45 | 2454.49 | 0.633× |
| sum | 9.26 | 10.50 | 1.13× | 9.25 | 10.54 | 1.14× |
| column-sum | 58.90 | 85.25 | 1.45× | 56.81 | 159.97 | 2.82× |
| mean | 12.24 | 111.77 | 9.13× | 12.26 | 110.71 | 9.03× |
| norm | 4.88 | 7.79 | 1.6× | 4.98 | 7.87 | 1.58× |
| compressed-export | 13.26 | 13.76 | 1.04× | 13.35 | 13.21 | 0.99× |
| copy-prune | 82.60 | 89.87 | 1.09× | 83.37 | 88.79 | 1.06× |

Each result passed numerical checks outside timing. Setup, calibration, warmups, and validation are excluded. Public API allocation/conversion is included; inspect each backend's scope in the JSON report.

LU/iterative/triangular workloads use a derived Hermitian diagonally dominant system; its nnz is recorded per benchmark. These are synthetic, in-process measurements, not universal performance guarantees.
