# SpPy versus SciPy sparse

AMD Ryzen 5 5600X 6-Core Processor · Python 3.14.3 · NumPy 2.5.5 · SciPy 1.18.1 · SpPy 1.0.0

Pattern: random; shape: [1000, 1000]; nnz: 4993; dtype: complex128; threads: 1; samples: 7.

Median microseconds per call. Ratios are SciPy / SpPy in matching storage order; **above 1 means SpPy is faster**.

| Operation | SpPy C µs | SciPy CSR µs | CSR / C | SpPy F µs | SciPy CSC µs | CSC / F |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| construct-triplets | 56.95 | 74.48 | 1.31× | 56.49 | 74.04 | 1.31× |
| construct-compressed | 8.93 | 10.96 | 1.23× | 9.23 | 10.99 | 1.19× |
| copy | 2.98 | 12.44 | 4.18× | 2.97 | 12.45 | 4.2× |
| copy-assign | 31.91 | 57.15 | 1.79× | 31.68 | 57.24 | 1.81× |
| gather | 18.83 | 12.44 | 0.661× | 19.21 | 12.54 | 0.653× |
| slice | 10.43 | 54.01 | 5.18× | 9.66 | 54.05 | 5.6× |
| transpose | 12.94 | 38.33 | 2.96× | 13.03 | 38.33 | 2.94× |
| adjoint | 13.01 | 55.52 | 4.27× | 13.06 | 55.62 | 4.26× |
| add | 26.41 | 35.73 | 1.35× | 25.37 | 35.94 | 1.42× |
| multiply | 26.82 | 42.25 | 1.58× | 27.05 | 43.50 | 1.61× |
| matvec | 9.53 | 7.04 | 0.739× | 8.47 | 7.32 | 0.864× |
| matmat | 76.20 | 42.44 | 0.557× | 68.94 | 37.97 | 0.551× |
| sparse-matmul | 298.63 | 237.54 | 0.795× | 296.39 | 238.17 | 0.804× |
| sum | 3.98 | 4.11 | 1.03× | 4.46 | 4.01 | 0.899× |
| column-sum | 20.64 | 35.25 | 1.71× | 21.71 | 25.70 | 1.18× |
| mean | 7.59 | 35.30 | 4.65× | 7.94 | 35.24 | 4.44× |
| norm | 1.95 | 4.82 | 2.48× | 1.96 | 4.83 | 2.46× |
| compressed-export | 2.45 | 2.43 | 0.995× | 2.42 | 2.24 | 0.929× |
| copy-prune | 32.12 | 22.20 | 0.691× | 32.26 | 21.57 | 0.669× |
| lu-factor | 120295.79 | 82349.09 | 0.685× | 120706.05 | 82126.69 | 0.68× |
| lu-solve | 5043.10 | 3078.31 | 0.61× | 5082.71 | 3075.10 | 0.605× |
| cg | 417.63 | 556.61 | 1.33× | 412.10 | 570.52 | 1.38× |
| bicgstab | 889.68 | 678.30 | 0.762× | 896.19 | 710.64 | 0.793× |
| triangular-solve | 98.84 | 370.55 | 3.75× | 97.03 | 288.91 | 2.98× |

Each result passed numerical checks outside timing. Setup, calibration, warmups, and validation are excluded. Public API allocation/conversion is included; inspect each backend's scope in the JSON report.

LU/iterative/triangular workloads use a derived Hermitian diagonally dominant system; its nnz is recorded per benchmark. These are synthetic, in-process measurements, not universal performance guarantees.
