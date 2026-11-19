# SpPy versus SciPy sparse

AMD Ryzen 5 5600X 6-Core Processor · Python 3.14.3 · NumPy 2.5.5 · SciPy 1.18.1 · SpPy 1.0.0

Pattern: banded; shape: [10000, 10000]; nnz: 49994; dtype: float64; threads: 1; samples: 7.

Median microseconds per call. Ratios are SciPy / SpPy in matching storage order; **above 1 means SpPy is faster**.

| Operation | SpPy C µs | SciPy CSR µs | CSR / C | SpPy F µs | SciPy CSC µs | CSC / F |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| construct-triplets | 395.16 | 158.72 | 0.402× | 371.27 | 316.26 | 0.852× |
| construct-compressed | 43.99 | 23.90 | 0.543× | 45.93 | 23.90 | 0.52× |
| copy | 14.27 | 25.99 | 1.82× | 14.34 | 25.94 | 1.81× |
| copy-assign | 42.21 | 68.00 | 1.61× | 41.97 | 68.16 | 1.62× |
| gather | 18.44 | 12.13 | 0.658× | 18.28 | 12.13 | 0.664× |
| slice | 74.31 | 127.51 | 1.72× | 71.11 | 128.64 | 1.81× |
| transpose | 87.76 | 138.34 | 1.58× | 88.59 | 139.78 | 1.58× |
| adjoint | 89.12 | 172.31 | 1.93× | 97.79 | 177.17 | 1.81× |
| add | 177.29 | 115.78 | 0.653× | 203.93 | 114.70 | 0.562× |
| multiply | 205.73 | 115.98 | 0.564× | 188.46 | 116.01 | 0.616× |
| matvec | 29.53 | 26.74 | 0.905× | 31.86 | 35.75 | 1.12× |
| matmat | 170.40 | 126.52 | 0.742× | 157.73 | 113.60 | 0.72× |
| sparse-matmul | 572.42 | 541.75 | 0.946× | 572.33 | 553.06 | 0.966× |
| sum | 15.88 | 10.62 | 0.669× | 15.99 | 10.45 | 0.653× |
| column-sum | 126.12 | 63.09 | 0.5× | 133.34 | 126.06 | 0.945× |
| mean | 19.26 | 66.21 | 3.44× | 19.35 | 66.46 | 3.43× |
| norm | 8.53 | 6.27 | 0.736× | 8.50 | 7.90 | 0.929× |
| compressed-export | 12.79 | 13.67 | 1.07× | 12.96 | 13.19 | 1.02× |
| copy-prune | 45.49 | 55.79 | 1.23× | 45.49 | 55.85 | 1.23× |
| lu-factor | 3989.75 | 2749.94 | 0.689× | 4205.91 | 2602.86 | 0.619× |
| lu-solve | 720.26 | 501.68 | 0.697× | 726.39 | 506.31 | 0.697× |
| cg | 1343.88 | 1326.21 | 0.987× | 1374.23 | 1518.01 | 1.1× |
| bicgstab | 2180.82 | 1774.92 | 0.814× | 2216.34 | 2028.79 | 0.915× |
| triangular-solve | 392.98 | 769.03 | 1.96× | 405.49 | 589.52 | 1.45× |

Each result passed numerical checks outside timing. Setup, calibration, warmups, and validation are excluded. Public API allocation/conversion is included; inspect each backend's scope in the JSON report.

LU/iterative/triangular workloads use a derived Hermitian diagonally dominant system; its nnz is recorded per benchmark. These are synthetic, in-process measurements, not universal performance guarantees.
