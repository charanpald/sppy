# SpPy versus SciPy sparse

AMD Ryzen 5 5600X 6-Core Processor · Python 3.14.3 · NumPy 2.5.5 · SciPy 1.18.1 · SpPy 1.0.0

Pattern: random; shape: [1000, 1000]; nnz: 4993; dtype: float64; threads: 1; samples: 7.

Median microseconds per call. Ratios are SciPy / SpPy in matching storage order; **above 1 means SpPy is faster**.

| Operation | SpPy C µs | SciPy CSR µs | CSR / C | SpPy F µs | SciPy CSC µs | CSC / F |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| construct-triplets | 48.59 | 67.31 | 1.39× | 47.50 | 67.01 | 1.41× |
| construct-compressed | 6.84 | 9.90 | 1.45× | 6.84 | 9.73 | 1.42× |
| copy | 2.14 | 10.93 | 5.11× | 2.17 | 10.85 | 4.99× |
| copy-assign | 27.43 | 51.14 | 1.86× | 28.09 | 51.48 | 1.83× |
| gather | 5.41 | 12.24 | 2.26× | 5.41 | 12.35 | 2.28× |
| slice | 10.23 | 53.73 | 5.25× | 9.77 | 54.00 | 5.53× |
| transpose | 11.40 | 37.02 | 3.25× | 11.32 | 36.97 | 3.27× |
| adjoint | 10.16 | 51.15 | 5.04× | 10.40 | 51.52 | 4.96× |
| add | 10.15 | 31.24 | 3.08× | 12.24 | 31.56 | 2.58× |
| multiply | 10.40 | 34.32 | 3.3× | 12.55 | 34.74 | 2.77× |
| matvec | 4.02 | 4.40 | 1.09× | 3.98 | 5.21 | 1.31× |
| matmat | 9.48 | 17.98 | 1.9× | 11.27 | 17.36 | 1.54× |
| sparse-matmul | 255.31 | 216.24 | 0.847× | 254.06 | 217.78 | 0.857× |
| sum | 1.58 | 3.43 | 2.17× | 1.58 | 3.43 | 2.17× |
| column-sum | 3.36 | 27.97 | 8.31× | 3.22 | 19.98 | 6.21× |
| mean | 3.95 | 28.50 | 7.21× | 3.98 | 28.61 | 7.19× |
| norm | 0.66 | 2.08 | 3.17× | 0.66 | 2.07 | 3.14× |
| compressed-export | 1.57 | 1.63 | 1.04× | 1.55 | 1.66 | 1.07× |
| copy-prune | 5.37 | 16.17 | 3.01× | 5.36 | 16.34 | 3.05× |
| lu-factor | 44308.60 | 39916.64 | 0.901× | 44980.76 | 40293.43 | 0.896× |
| lu-solve | 1391.39 | 1693.31 | 1.22× | 1380.01 | 1691.86 | 1.23× |
| cg | 134.70 | 404.18 | 3× | 156.73 | 429.21 | 2.74× |
| bicgstab | 196.67 | 514.82 | 2.62× | 221.85 | 547.88 | 2.47× |
| triangular-solve | 20.54 | 348.65 | 17× | 29.40 | 273.63 | 9.31× |

Each result passed numerical checks outside timing. Setup, calibration, warmups, and validation are excluded. Public API allocation/conversion is included; inspect each backend's scope in the JSON report.

LU/iterative/triangular workloads use a derived Hermitian diagonally dominant system; its nnz is recorded per benchmark. These are synthetic, in-process measurements, not universal performance guarantees.
