# SpPy versus SciPy sparse

AMD Ryzen 5 5600X 6-Core Processor · Python 3.14.3 · NumPy 2.5.5 · SciPy 1.18.1 · SpPy 1.0.0

Pattern: banded; shape: [10000, 10000]; nnz: 49994; dtype: float64; threads: 1; samples: 7.

Median microseconds per call. Ratios are SciPy / SpPy in matching storage order; **above 1 means SpPy is faster**.

| Operation | SpPy C µs | SciPy CSR µs | CSR / C | SpPy F µs | SciPy CSC µs | CSC / F |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| construct-triplets | 702.09 | 161.83 | 0.23× | 426.64 | 328.08 | 0.769× |
| construct-compressed | 544.83 | 23.83 | 0.0437× | 491.85 | 23.90 | 0.0486× |
| copy | 890.17 | 24.84 | 0.0279× | 565.08 | 24.94 | 0.0441× |
| copy-assign | 627.35 | 67.08 | 0.107× | 646.31 | 67.41 | 0.104× |
| gather | 18.64 | 12.35 | 0.663× | 18.62 | 12.07 | 0.648× |
| slice | 30875.58 | 130.72 | 0.00423× | 31794.60 | 132.81 | 0.00418× |
| transpose | 91.36 | 142.24 | 1.56× | 89.82 | 139.63 | 1.55× |
| adjoint | 91.84 | 173.99 | 1.89× | 89.16 | 173.18 | 1.94× |
| add | 170.24 | 113.96 | 0.669× | 197.73 | 112.56 | 0.569× |
| multiply | 235.87 | 112.99 | 0.479× | 235.99 | 113.85 | 0.482× |
| matvec | 40.43 | 26.38 | 0.652× | 41.34 | 35.49 | 0.859× |
| matmat | 323.42 | 126.58 | 0.391× | 368.23 | 112.15 | 0.305× |
| sparse-matmul | 3371.82 | 542.79 | 0.161× | 887.08 | 541.82 | 0.611× |
| sum | 31.35 | 10.47 | 0.334× | 31.28 | 10.45 | 0.334× |
| column-sum | 136.28 | 63.81 | 0.468× | 146.83 | 125.36 | 0.854× |
| mean | 37.46 | 66.98 | 1.79× | 37.01 | 67.51 | 1.82× |
| norm | 32.90 | 6.49 | 0.197× | 32.80 | 7.80 | 0.238× |
| compressed-export | 24.55 | 13.36 | 0.544× | 24.35 | 12.81 | 0.526× |
| copy-prune | 593.55 | 58.27 | 0.0982× | 520.50 | 58.39 | 0.112× |
| lu-factor | 4292.64 | 2841.68 | 0.662× | 4725.34 | 2841.38 | 0.601× |
| lu-solve | 723.25 | 503.67 | 0.696× | 736.28 | 499.77 | 0.679× |
| cg | 1380.55 | 1320.29 | 0.956× | 1407.10 | 1517.90 | 1.08× |
| bicgstab | 2217.99 | 1768.80 | 0.797× | 2239.46 | 2026.16 | 0.905× |
| triangular-solve | 420.14 | 777.07 | 1.85× | 431.64 | 596.71 | 1.38× |

Each result passed numerical checks outside timing. Setup, calibration, warmups, and validation are excluded. Public API allocation/conversion is included; inspect each backend's scope in the JSON report.

LU/iterative/triangular workloads use a derived Hermitian diagonally dominant system; its nnz is recorded per benchmark. These are synthetic, in-process measurements, not universal performance guarantees.
