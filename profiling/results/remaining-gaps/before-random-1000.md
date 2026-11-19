# SpPy versus SciPy sparse

AMD Ryzen 5 5600X 6-Core Processor · Python 3.14.3 · NumPy 2.5.5 · SciPy 1.18.1 · SpPy 1.0.0

Pattern: random; shape: [1000, 1000]; nnz: 4993; dtype: float64; threads: 1; samples: 7.

Median microseconds per call. Ratios are SciPy / SpPy in matching storage order; **above 1 means SpPy is faster**.

| Operation | SpPy C µs | SciPy CSR µs | CSR / C | SpPy F µs | SciPy CSC µs | CSC / F |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| construct-triplets | 49.79 | 71.22 | 1.43× | 47.85 | 73.77 | 1.54× |
| construct-compressed | 6.91 | 10.11 | 1.46× | 7.20 | 10.45 | 1.45× |
| copy | 2.19 | 11.25 | 5.14× | 2.18 | 10.94 | 5.01× |
| copy-assign | 27.90 | 52.35 | 1.88× | 28.49 | 52.83 | 1.85× |
| gather | 5.51 | 12.55 | 2.28× | 5.55 | 12.71 | 2.29× |
| slice | 10.11 | 54.66 | 5.41× | 9.59 | 54.53 | 5.69× |
| transpose | 11.08 | 36.31 | 3.28× | 11.17 | 36.61 | 3.28× |
| adjoint | 10.33 | 51.59 | 5× | 10.24 | 51.47 | 5.03× |
| add | 10.06 | 31.73 | 3.15× | 12.24 | 31.81 | 2.6× |
| multiply | 10.54 | 34.31 | 3.26× | 15.64 | 34.70 | 2.22× |
| matvec | 4.11 | 4.48 | 1.09× | 4.11 | 5.41 | 1.31× |
| matmat | 9.68 | 18.39 | 1.9× | 11.19 | 17.56 | 1.57× |
| sparse-matmul | 233.88 | 220.54 | 0.943× | 235.52 | 219.85 | 0.933× |
| sum | 1.63 | 3.45 | 2.12× | 1.59 | 3.41 | 2.15× |
| column-sum | 3.70 | 28.80 | 7.79× | 3.19 | 19.41 | 6.09× |
| mean | 4.04 | 28.77 | 7.12× | 4.03 | 28.82 | 7.16× |
| norm | 0.68 | 2.31 | 3.38× | 0.66 | 2.16 | 3.28× |
| compressed-export | 1.55 | 1.63 | 1.05× | 1.62 | 1.61 | 0.996× |
| copy-prune | 5.42 | 16.70 | 3.08× | 5.45 | 16.79 | 3.08× |
| lu-factor | 44386.30 | 39975.19 | 0.901× | 44230.03 | 40549.68 | 0.917× |
| lu-solve | 1384.96 | 1681.89 | 1.21× | 1387.11 | 1678.86 | 1.21× |
| cg | 133.28 | 405.98 | 3.05× | 155.33 | 434.20 | 2.8× |
| bicgstab | 191.31 | 511.69 | 2.67× | 214.89 | 547.24 | 2.55× |
| triangular-solve | 20.41 | 342.18 | 16.8× | 29.09 | 268.66 | 9.24× |

Each result passed numerical checks outside timing. Setup, calibration, warmups, and validation are excluded. Public API allocation/conversion is included; inspect each backend's scope in the JSON report.

LU/iterative/triangular workloads use a derived Hermitian diagonally dominant system; its nnz is recorded per benchmark. These are synthetic, in-process measurements, not universal performance guarantees.
