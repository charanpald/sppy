# SpPy versus SciPy sparse

AMD Ryzen 5 5600X 6-Core Processor · Python 3.14.3 · NumPy 2.5.5 · SciPy 1.18.1 · SpPy 1.0.0

Pattern: banded; shape: [10000, 10000]; nnz: 49994; dtype: float64; threads: 1; samples: 7.

Median microseconds per call. Ratios are SciPy / SpPy in matching storage order; **above 1 means SpPy is faster**.

| Operation | SpPy C µs | SciPy CSR µs | CSR / C | SpPy F µs | SciPy CSC µs | CSC / F |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| construct-triplets | 194.15 | 167.21 | 0.861× | 229.49 | 314.53 | 1.37× |
| construct-compressed | 30.88 | 23.37 | 0.757× | 30.40 | 23.66 | 0.778× |
| copy | 14.00 | 24.57 | 1.76× | 14.50 | 24.83 | 1.71× |
| copy-assign | 41.99 | 65.79 | 1.57× | 41.40 | 66.21 | 1.6× |
| gather | 5.37 | 12.24 | 2.28× | 5.22 | 11.99 | 2.3× |
| slice | 76.33 | 129.58 | 1.7× | 72.28 | 132.08 | 1.83× |
| transpose | 91.07 | 141.28 | 1.55× | 91.42 | 140.25 | 1.53× |
| adjoint | 91.04 | 175.81 | 1.93× | 89.65 | 173.39 | 1.93× |
| add | 96.90 | 118.57 | 1.22× | 123.57 | 117.82 | 0.953× |
| multiply | 101.08 | 121.04 | 1.2× | 124.43 | 121.16 | 0.974× |
| matvec | 25.97 | 27.27 | 1.05× | 31.90 | 36.71 | 1.15× |
| matmat | 83.77 | 136.59 | 1.63× | 102.63 | 120.89 | 1.18× |
| sparse-matmul | 481.96 | 575.98 | 1.2× | 478.00 | 568.84 | 1.19× |
| sum | 9.37 | 10.82 | 1.16× | 9.36 | 11.21 | 1.2× |
| column-sum | 30.43 | 66.95 | 2.2× | 21.42 | 131.00 | 6.12× |
| mean | 12.53 | 72.79 | 5.81× | 12.58 | 72.44 | 5.76× |
| norm | 5.15 | 6.62 | 1.29× | 5.05 | 8.22 | 1.63× |
| compressed-export | 13.39 | 13.81 | 1.03× | 13.43 | 13.78 | 1.03× |
| copy-prune | 50.99 | 63.99 | 1.25× | 51.11 | 64.14 | 1.25× |
| lu-factor | 3168.72 | 2844.16 | 0.898× | 3087.45 | 2659.33 | 0.861× |
| lu-solve | 421.14 | 527.71 | 1.25× | 424.50 | 535.91 | 1.26× |
| cg | 1097.09 | 1427.53 | 1.3× | 1213.27 | 1652.64 | 1.36× |
| bicgstab | 1499.19 | 1917.51 | 1.28× | 1564.49 | 2196.88 | 1.4× |
| triangular-solve | 129.12 | 830.73 | 6.43× | 187.01 | 627.73 | 3.36× |

Each result passed numerical checks outside timing. Setup, calibration, warmups, and validation are excluded. Public API allocation/conversion is included; inspect each backend's scope in the JSON report.

LU/iterative/triangular workloads use a derived Hermitian diagonally dominant system; its nnz is recorded per benchmark. These are synthetic, in-process measurements, not universal performance guarantees.
