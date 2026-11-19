# SpPy versus SciPy sparse

AMD Ryzen 5 5600X 6-Core Processor · Python 3.14.3 · NumPy 2.5.5 · SciPy 1.18.1 · SpPy 1.0.0

Pattern: random; shape: [1000, 1000]; nnz: 4993; dtype: complex128; threads: 1; samples: 7.

Median microseconds per call. Ratios are SciPy / SpPy in matching storage order; **above 1 means SpPy is faster**.

| Operation | SpPy C µs | SciPy CSR µs | CSR / C | SpPy F µs | SciPy CSC µs | CSC / F |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| construct-triplets | 51.80 | 73.20 | 1.41× | 51.73 | 73.45 | 1.42× |
| construct-compressed | 8.33 | 11.93 | 1.43× | 8.32 | 11.75 | 1.41× |
| copy | 2.91 | 12.55 | 4.31× | 2.91 | 12.59 | 4.33× |
| copy-assign | 30.57 | 53.58 | 1.75× | 29.92 | 55.22 | 1.85× |
| gather | 5.47 | 13.05 | 2.39× | 5.64 | 12.73 | 2.26× |
| slice | 10.41 | 54.94 | 5.28× | 9.99 | 54.89 | 5.5× |
| transpose | 12.85 | 38.90 | 3.03× | 12.98 | 38.68 | 2.98× |
| adjoint | 13.48 | 58.11 | 4.31× | 13.26 | 58.44 | 4.41× |
| add | 12.48 | 37.72 | 3.02× | 11.31 | 36.60 | 3.24× |
| multiply | 12.78 | 42.82 | 3.35× | 12.97 | 42.43 | 3.27× |
| matvec | 6.13 | 7.11 | 1.16× | 7.19 | 7.36 | 1.02× |
| matmat | 20.64 | 40.58 | 1.97× | 23.45 | 35.55 | 1.52× |
| sparse-matmul | 203.76 | 230.41 | 1.13× | 202.11 | 230.04 | 1.14× |
| sum | 2.42 | 3.99 | 1.65× | 2.40 | 3.89 | 1.62× |
| column-sum | 4.35 | 33.15 | 7.61× | 3.43 | 23.94 | 6.98× |
| mean | 4.90 | 32.05 | 6.54× | 4.80 | 32.08 | 6.69× |
| norm | 1.86 | 4.75 | 2.55× | 1.86 | 4.74 | 2.54× |
| compressed-export | 2.28 | 2.34 | 1.02× | 2.26 | 2.38 | 1.05× |
| copy-prune | 8.43 | 21.13 | 2.51× | 8.72 | 21.58 | 2.48× |
| lu-factor | 99313.84 | 80124.20 | 0.807× | 100154.72 | 80893.30 | 0.808× |
| lu-solve | 2979.58 | 3106.18 | 1.04× | 2979.78 | 3123.00 | 1.05× |
| cg | 360.25 | 547.07 | 1.52× | 380.48 | 600.72 | 1.58× |
| bicgstab | 629.08 | 675.26 | 1.07× | 643.62 | 706.01 | 1.1× |
| triangular-solve | 31.26 | 369.68 | 11.8× | 41.42 | 285.96 | 6.9× |

Each result passed numerical checks outside timing. Setup, calibration, warmups, and validation are excluded. Public API allocation/conversion is included; inspect each backend's scope in the JSON report.

LU/iterative/triangular workloads use a derived Hermitian diagonally dominant system; its nnz is recorded per benchmark. These are synthetic, in-process measurements, not universal performance guarantees.
