# SpPy versus SciPy sparse

AMD Ryzen 5 5600X 6-Core Processor · Python 3.14.3 · NumPy 2.5.5 · SciPy 1.18.1 · SpPy 1.0.0

Pattern: random; shape: [1000, 1000]; nnz: 4993; dtype: float64; threads: 1; samples: 9.

Median microseconds per call. Ratios are SciPy / SpPy in matching storage order; **above 1 means SpPy is faster**.

| Operation | SpPy C µs | SciPy CSR µs | CSR / C | SpPy F µs | SciPy CSC µs | CSC / F |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| construct-triplets | 63.98 | 69.51 | 1.09× | 51.80 | 69.80 | 1.35× |
| construct-compressed | 88.55 | 10.50 | 0.119× | 77.50 | 10.51 | 0.136× |
| copy | 69.95 | 10.84 | 0.155× | 58.32 | 10.92 | 0.187× |
| copy-assign | 108.64 | 55.41 | 0.51× | 96.81 | 55.76 | 0.576× |
| gather | 18.66 | 12.31 | 0.66× | 18.96 | 12.40 | 0.654× |
| slice | 2949.76 | 53.28 | 0.0181× | 2948.52 | 53.79 | 0.0182× |
| transpose | 10.85 | 37.40 | 3.45× | 10.98 | 37.48 | 3.41× |
| adjoint | 10.61 | 52.19 | 4.92× | 10.59 | 52.02 | 4.91× |
| add | 23.40 | 32.07 | 1.37× | 26.75 | 32.14 | 1.2× |
| multiply | 25.69 | 35.33 | 1.38× | 25.82 | 34.85 | 1.35× |
| matvec | 5.78 | 4.40 | 0.761× | 5.98 | 5.22 | 0.873× |
| matmat | 31.54 | 18.06 | 0.573× | 33.07 | 17.52 | 0.53× |
| sparse-matmul | 911.46 | 225.91 | 0.248× | 401.10 | 217.06 | 0.541× |
| sum | 4.68 | 3.43 | 0.733× | 4.69 | 3.45 | 0.735× |
| column-sum | 17.57 | 27.51 | 1.57× | 19.67 | 19.59 | 0.996× |
| mean | 9.92 | 27.89 | 2.81× | 10.39 | 27.95 | 2.69× |
| norm | 3.46 | 2.15 | 0.623× | 3.46 | 2.35 | 0.681× |
| compressed-export | 2.57 | 1.63 | 0.636× | 2.63 | 1.62 | 0.617× |
| copy-prune | 75.99 | 16.23 | 0.214× | 62.39 | 16.28 | 0.261× |
| lu-factor | 44212.16 | 39680.06 | 0.897× | 44506.84 | 39811.43 | 0.895× |
| lu-solve | 2175.20 | 1681.19 | 0.773× | 2176.21 | 1679.15 | 0.772× |
| cg | 238.80 | 397.53 | 1.66× | 234.39 | 426.07 | 1.82× |
| bicgstab | 330.49 | 506.73 | 1.53× | 326.81 | 544.52 | 1.67× |
| triangular-solve | 74.48 | 345.42 | 4.64× | 71.44 | 266.43 | 3.73× |

Each result passed numerical checks outside timing. Setup, calibration, warmups, and validation are excluded. Public API allocation/conversion is included; inspect each backend's scope in the JSON report.

LU/iterative/triangular workloads use a derived Hermitian diagonally dominant system; its nnz is recorded per benchmark. These are synthetic, in-process measurements, not universal performance guarantees.
