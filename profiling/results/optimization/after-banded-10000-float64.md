# SpPy versus SciPy sparse

AMD Ryzen 5 5600X 6-Core Processor · Python 3.14.3 · NumPy 2.5.5 · SciPy 1.18.1 · SpPy 1.0.0

Pattern: banded; shape: [10000, 10000]; nnz: 49994; dtype: float64; threads: 1; samples: 7.

Median microseconds per call. Ratios are SciPy / SpPy in matching storage order; **above 1 means SpPy is faster**.

| Operation | SpPy C µs | SciPy CSR µs | CSR / C | SpPy F µs | SciPy CSC µs | CSC / F |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| construct-triplets | 409.25 | 156.77 | 0.383× | 388.24 | 315.31 | 0.812× |
| construct-compressed | 56.03 | 22.75 | 0.406× | 56.84 | 22.81 | 0.401× |
| copy | 14.23 | 25.85 | 1.82× | 14.82 | 24.52 | 1.65× |
| copy-assign | 41.84 | 69.62 | 1.66× | 43.01 | 69.05 | 1.61× |
| gather | 18.99 | 12.20 | 0.643× | 19.15 | 12.07 | 0.63× |
| slice | 75.49 | 129.22 | 1.71× | 71.80 | 129.94 | 1.81× |
| transpose | 91.54 | 144.65 | 1.58× | 90.62 | 139.44 | 1.54× |
| adjoint | 90.65 | 174.12 | 1.92× | 90.28 | 171.79 | 1.9× |
| add | 172.92 | 111.72 | 0.646× | 195.44 | 112.94 | 0.578× |
| multiply | 254.18 | 116.80 | 0.459× | 252.15 | 117.62 | 0.466× |
| matvec | 28.97 | 26.18 | 0.904× | 30.87 | 35.38 | 1.15× |
| matmat | 168.93 | 125.64 | 0.744× | 152.57 | 111.82 | 0.733× |
| sparse-matmul | 712.42 | 546.30 | 0.767× | 849.73 | 549.34 | 0.646× |
| sum | 17.39 | 11.36 | 0.653× | 17.81 | 11.40 | 0.64× |
| column-sum | 122.83 | 63.79 | 0.519× | 131.45 | 125.82 | 0.957× |
| mean | 19.22 | 67.89 | 3.53× | 19.00 | 67.30 | 3.54× |
| norm | 8.48 | 7.88 | 0.929× | 8.53 | 6.43 | 0.754× |
| compressed-export | 13.40 | 13.78 | 1.03× | 13.31 | 13.48 | 1.01× |
| copy-prune | 47.78 | 58.82 | 1.23× | 48.68 | 58.11 | 1.19× |
| lu-factor | 4526.42 | 2996.86 | 0.662× | 4136.03 | 2549.14 | 0.616× |
| lu-solve | 737.14 | 514.71 | 0.698× | 742.02 | 507.39 | 0.684× |
| cg | 1434.60 | 1428.74 | 0.996× | 1442.16 | 1645.93 | 1.14× |
| bicgstab | 2349.14 | 1924.26 | 0.819× | 2408.01 | 2203.10 | 0.915× |
| triangular-solve | 397.28 | 789.29 | 1.99× | 416.29 | 601.56 | 1.45× |

Each result passed numerical checks outside timing. Setup, calibration, warmups, and validation are excluded. Public API allocation/conversion is included; inspect each backend's scope in the JSON report.

LU/iterative/triangular workloads use a derived Hermitian diagonally dominant system; its nnz is recorded per benchmark. These are synthetic, in-process measurements, not universal performance guarantees.
