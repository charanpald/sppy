# SpPy versus SciPy sparse

AMD Ryzen 5 5600X 6-Core Processor · Python 3.14.3 · NumPy 2.5.5 · SciPy 1.18.1 · SpPy 1.0.0

Pattern: random; shape: [1000, 1000]; nnz: 4993; dtype: complex128; threads: 1; samples: 7.

Median microseconds per call. Ratios are SciPy / SpPy in matching storage order; **above 1 means SpPy is faster**.

| Operation | SpPy C µs | SciPy CSR µs | CSR / C | SpPy F µs | SciPy CSC µs | CSC / F |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| construct-triplets | 57.60 | 75.47 | 1.31× | 58.19 | 76.82 | 1.32× |
| construct-compressed | 10.28 | 11.25 | 1.1× | 10.56 | 11.09 | 1.05× |
| copy | 2.88 | 12.45 | 4.32× | 2.95 | 12.36 | 4.18× |
| copy-assign | 29.54 | 54.39 | 1.84× | 30.33 | 54.73 | 1.8× |
| gather | 19.15 | 12.40 | 0.647× | 19.79 | 12.64 | 0.639× |
| slice | 10.22 | 54.89 | 5.37× | 9.76 | 55.28 | 5.66× |
| transpose | 12.87 | 39.49 | 3.07× | 12.67 | 39.36 | 3.11× |
| adjoint | 13.08 | 57.84 | 4.42× | 13.06 | 57.62 | 4.41× |
| add | 26.70 | 36.74 | 1.38× | 24.73 | 37.89 | 1.53× |
| multiply | 28.15 | 40.48 | 1.44× | 27.90 | 41.21 | 1.48× |
| matvec | 9.40 | 6.87 | 0.731× | 8.89 | 7.32 | 0.823× |
| matmat | 72.88 | 39.30 | 0.539× | 64.95 | 34.79 | 0.536× |
| sparse-matmul | 291.94 | 236.26 | 0.809× | 345.67 | 232.63 | 0.673× |
| sum | 3.59 | 3.94 | 1.1× | 3.65 | 3.89 | 1.07× |
| column-sum | 18.07 | 31.29 | 1.73× | 19.51 | 22.76 | 1.17× |
| mean | 6.07 | 31.29 | 5.15× | 6.29 | 31.33 | 4.98× |
| norm | 1.86 | 4.67 | 2.52× | 1.86 | 4.60 | 2.47× |
| compressed-export | 2.23 | 2.37 | 1.06× | 2.35 | 2.34 | 0.998× |
| copy-prune | 31.22 | 20.07 | 0.643× | 31.33 | 20.43 | 0.652× |
| lu-factor | 119569.22 | 80697.60 | 0.675× | 119336.72 | 80781.48 | 0.677× |
| lu-solve | 5080.61 | 3123.28 | 0.615× | 5198.99 | 3136.85 | 0.603× |
| cg | 418.42 | 546.86 | 1.31× | 415.06 | 562.73 | 1.36× |
| bicgstab | 897.86 | 675.61 | 0.752× | 889.39 | 692.48 | 0.779× |
| triangular-solve | 99.49 | 376.28 | 3.78× | 98.54 | 295.50 | 3× |

Each result passed numerical checks outside timing. Setup, calibration, warmups, and validation are excluded. Public API allocation/conversion is included; inspect each backend's scope in the JSON report.

LU/iterative/triangular workloads use a derived Hermitian diagonally dominant system; its nnz is recorded per benchmark. These are synthetic, in-process measurements, not universal performance guarantees.
