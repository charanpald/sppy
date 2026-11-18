# SpPy versus SciPy sparse

AMD Ryzen 5 5600X 6-Core Processor · Python 3.14.3 · NumPy 2.5.5 · SciPy 1.18.1 · SpPy 1.0.0

Pattern: banded; shape: [10000, 10000]; nnz: 49994; dtype: float64; threads: 1; samples: 7.

Median microseconds per call. Ratios are SciPy / SpPy in matching storage order; **above 1 means SpPy is faster**.

| Operation | SpPy C µs | SciPy CSR µs | CSR / C | SpPy F µs | SciPy CSC µs | CSC / F |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| construct-triplets | 791.26 | 162.20 | 0.205× | 417.73 | 307.42 | 0.736× |
| construct-compressed | 550.89 | 23.96 | 0.0435× | 508.39 | 24.25 | 0.0477× |
| copy | 903.53 | 25.02 | 0.0277× | 577.02 | 25.06 | 0.0434× |
| copy-assign | 600.96 | 67.20 | 0.112× | 747.32 | 67.95 | 0.0909× |
| gather | 18.63 | 12.18 | 0.654× | 18.58 | 12.12 | 0.652× |
| slice | 30667.20 | 129.11 | 0.00421× | 30817.07 | 130.09 | 0.00422× |
| transpose | 91.07 | 137.99 | 1.52× | 91.08 | 138.84 | 1.52× |
| adjoint | 92.66 | 178.92 | 1.93× | 92.99 | 170.72 | 1.84× |
| add | 169.86 | 113.07 | 0.666× | 195.71 | 114.32 | 0.584× |
| multiply | 235.18 | 113.25 | 0.482× | 234.93 | 115.32 | 0.491× |
| matvec | 40.12 | 25.98 | 0.647× | 41.13 | 34.98 | 0.85× |
| matmat | 323.00 | 124.67 | 0.386× | 370.39 | 111.47 | 0.301× |
| sparse-matmul | 3481.92 | 538.34 | 0.155× | 835.43 | 540.80 | 0.647× |
| sum | 31.21 | 10.54 | 0.338× | 31.05 | 10.57 | 0.34× |
| column-sum | 136.19 | 63.47 | 0.466× | 146.35 | 123.90 | 0.847× |
| mean | 37.48 | 67.41 | 1.8× | 37.16 | 67.95 | 1.83× |
| norm | 32.85 | 8.02 | 0.244× | 32.85 | 6.44 | 0.196× |
| compressed-export | 24.34 | 12.92 | 0.531× | 24.28 | 12.62 | 0.52× |
| copy-prune | 599.45 | 57.76 | 0.0964× | 546.27 | 57.86 | 0.106× |
| lu-factor | 4308.84 | 3030.19 | 0.703× | 4491.51 | 2808.53 | 0.625× |
| lu-solve | 717.97 | 493.13 | 0.687× | 715.81 | 495.15 | 0.692× |
| cg | 1372.78 | 1315.45 | 0.958× | 1394.35 | 1514.78 | 1.09× |
| bicgstab | 2184.25 | 1773.01 | 0.812× | 2218.93 | 2037.00 | 0.918× |
| triangular-solve | 416.10 | 784.13 | 1.88× | 439.38 | 608.20 | 1.38× |

Each result passed numerical checks outside timing. Setup, calibration, warmups, and validation are excluded. Public API allocation/conversion is included; inspect each backend's scope in the JSON report.

LU/iterative/triangular workloads use a derived Hermitian diagonally dominant system; its nnz is recorded per benchmark. These are synthetic, in-process measurements, not universal performance guarantees.
