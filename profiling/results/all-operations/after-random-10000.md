# SpPy versus SciPy sparse

AMD Ryzen 5 5600X 6-Core Processor · Python 3.14.3 · NumPy 2.5.5 · SciPy 1.18.1 · SpPy 1.0.0

Pattern: random; shape: [10000, 10000]; nnz: 49985; dtype: float64; threads: 1; samples: 7.

Median microseconds per call. Ratios are SciPy / SpPy in matching storage order; **above 1 means SpPy is faster**.

| Operation | SpPy C µs | SciPy CSR µs | CSR / C | SpPy F µs | SciPy CSC µs | CSC / F |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| construct-triplets | 578.42 | 675.95 | 1.17× | 577.86 | 669.99 | 1.16× |
| construct-compressed | 33.99 | 22.60 | 0.665× | 33.98 | 22.88 | 0.673× |
| copy | 14.12 | 23.94 | 1.69× | 14.01 | 24.11 | 1.72× |
| copy-assign | 41.22 | 64.45 | 1.56× | 43.13 | 64.62 | 1.5× |
| gather | 5.44 | 12.47 | 2.29× | 5.75 | 12.78 | 2.22× |
| slice | 222.72 | 242.58 | 1.09× | 232.99 | 245.02 | 1.05× |
| transpose | 182.41 | 173.69 | 0.952× | 181.68 | 174.27 | 0.959× |
| adjoint | 182.04 | 207.18 | 1.14× | 181.06 | 208.79 | 1.15× |
| add | 92.45 | 240.37 | 2.6× | 115.07 | 239.91 | 2.09× |
| multiply | 92.81 | 244.35 | 2.63× | 115.55 | 242.31 | 2.1× |
| matvec | 65.06 | 61.50 | 0.945× | 61.99 | 55.88 | 0.901× |
| matmat | 136.83 | 202.87 | 1.48× | 149.24 | 190.81 | 1.28× |
| sparse-matmul | 3910.29 | 2443.85 | 0.625× | 3949.00 | 2446.08 | 0.619× |
| sum | 9.35 | 10.45 | 1.12× | 9.19 | 10.59 | 1.15× |
| column-sum | 59.20 | 86.17 | 1.46× | 57.11 | 162.83 | 2.85× |
| mean | 12.12 | 110.88 | 9.15× | 12.09 | 111.11 | 9.19× |
| norm | 4.86 | 6.23 | 1.28× | 4.93 | 6.15 | 1.25× |
| compressed-export | 13.18 | 13.44 | 1.02× | 13.30 | 13.38 | 1.01× |
| copy-prune | 83.29 | 92.71 | 1.11× | 83.98 | 90.30 | 1.08× |

Each result passed numerical checks outside timing. Setup, calibration, warmups, and validation are excluded. Public API allocation/conversion is included; inspect each backend's scope in the JSON report.

LU/iterative/triangular workloads use a derived Hermitian diagonally dominant system; its nnz is recorded per benchmark. These are synthetic, in-process measurements, not universal performance guarantees.
