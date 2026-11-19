# SpPy versus SciPy sparse

AMD Ryzen 5 5600X 6-Core Processor · Python 3.14.3 · NumPy 2.5.5 · SciPy 1.18.1 · SpPy 1.0.0

Pattern: random; shape: [10000, 10000]; nnz: 49985; dtype: float64; threads: 1; samples: 9.

Median microseconds per call. Ratios are SciPy / SpPy in matching storage order; **above 1 means SpPy is faster**.

| Operation | SpPy C µs | SciPy CSR µs | CSR / C | SpPy F µs | SciPy CSC µs | CSC / F |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| construct-compressed | 103.76 | 22.80 | 0.22× | 101.27 | 23.76 | 0.235× |
| sparse-matmul | 4371.65 | 2635.36 | 0.603× | 4777.94 | 2449.87 | 0.513× |

Each result passed numerical checks outside timing. Setup, calibration, warmups, and validation are excluded. Public API allocation/conversion is included; inspect each backend's scope in the JSON report.

LU/iterative/triangular workloads use a derived Hermitian diagonally dominant system; its nnz is recorded per benchmark. These are synthetic, in-process measurements, not universal performance guarantees.
