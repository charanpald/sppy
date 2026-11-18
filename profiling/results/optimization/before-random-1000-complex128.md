# SpPy versus SciPy sparse

AMD Ryzen 5 5600X 6-Core Processor · Python 3.14.3 · NumPy 2.5.5 · SciPy 1.18.1 · SpPy 1.0.0

Pattern: random; shape: [1000, 1000]; nnz: 4993; dtype: complex128; threads: 1; samples: 7.

Median microseconds per call. Ratios are SciPy / SpPy in matching storage order; **above 1 means SpPy is faster**.

| Operation | SpPy C µs | SciPy CSR µs | CSR / C | SpPy F µs | SciPy CSC µs | CSC / F |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| construct-triplets | 68.75 | 76.11 | 1.11× | 55.49 | 75.03 | 1.35× |
| construct-compressed | 96.41 | 11.07 | 0.115× | 79.95 | 10.95 | 0.137× |
| copy | 78.62 | 12.12 | 0.154× | 65.49 | 12.37 | 0.189× |
| copy-assign | 112.59 | 56.48 | 0.502× | 99.02 | 55.13 | 0.557× |
| gather | 18.70 | 12.50 | 0.668× | 18.98 | 12.92 | 0.681× |
| slice | 3058.03 | 57.92 | 0.0189× | 2974.91 | 57.05 | 0.0192× |
| transpose | 12.90 | 39.32 | 3.05× | 13.24 | 38.74 | 2.93× |
| adjoint | 12.87 | 56.63 | 4.4× | 12.93 | 56.21 | 4.35× |
| add | 25.01 | 36.26 | 1.45× | 26.30 | 36.71 | 1.4× |
| multiply | 27.70 | 40.31 | 1.46× | 27.77 | 41.20 | 1.48× |
| matvec | 11.15 | 6.74 | 0.604× | 9.27 | 7.11 | 0.766× |
| matmat | 75.88 | 39.12 | 0.516× | 61.99 | 35.09 | 0.566× |
| sparse-matmul | 1115.48 | 232.48 | 0.208× | 652.01 | 233.33 | 0.358× |
| sum | 5.51 | 3.78 | 0.687× | 5.33 | 3.83 | 0.719× |
| column-sum | 19.84 | 30.58 | 1.54× | 20.81 | 22.29 | 1.07× |
| mean | 10.67 | 30.72 | 2.88× | 10.43 | 30.91 | 2.96× |
| norm | 3.47 | 4.55 | 1.31× | 3.49 | 4.62 | 1.33× |
| compressed-export | 4.03 | 2.24 | 0.557× | 3.98 | 2.27 | 0.571× |
| copy-prune | 135.06 | 19.39 | 0.144× | 106.19 | 19.66 | 0.185× |
| lu-factor | 119141.76 | 78772.84 | 0.661× | 118713.21 | 79022.21 | 0.666× |
| lu-solve | 5061.76 | 3097.28 | 0.612× | 5091.52 | 3086.66 | 0.606× |
| cg | 419.68 | 552.88 | 1.32× | 416.26 | 567.47 | 1.36× |
| bicgstab | 898.90 | 683.70 | 0.761× | 894.68 | 695.65 | 0.778× |
| triangular-solve | 102.47 | 367.18 | 3.58× | 101.87 | 291.33 | 2.86× |

Each result passed numerical checks outside timing. Setup, calibration, warmups, and validation are excluded. Public API allocation/conversion is included; inspect each backend's scope in the JSON report.

LU/iterative/triangular workloads use a derived Hermitian diagonally dominant system; its nnz is recorded per benchmark. These are synthetic, in-process measurements, not universal performance guarantees.
