"""Measure randomized SVD time and separately tracked Python/NumPy peak memory."""

from __future__ import annotations

import argparse
import json
import statistics
import tracemalloc
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

import sppy
from profiling._common import environment, numerical_threads, positive_int
from sppy.linalg import rsvd


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """Benchmark deterministic real/complex dense/sparse SVD inputs and save JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=positive_int, default=600)
    parser.add_argument("--columns", type=positive_int, default=300)
    parser.add_argument("--rank", type=positive_int, default=20)
    parser.add_argument("--repeats", type=positive_int, default=7)
    parser.add_argument("--output", type=Path)
    options = parser.parse_args(argv)
    if options.rank > min(options.rows, options.columns):
        parser.error("rank must not exceed matrix dimensions")
    cases = []
    with numerical_threads(1):
        for dtype in (np.float64, np.complex128):
            rng = np.random.default_rng(21)
            dense = rng.standard_normal((options.rows, options.columns)).astype(dtype)
            if dtype == np.complex128:
                dense += 1j * rng.standard_normal(dense.shape)
            dense[rng.random(dense.shape) > 0.02] = 0
            for storage in ("dense", "sparse"):
                matrix = dense if storage == "dense" else sppy.asarray(dense)
                for power in (0, 2):
                    rsvd(matrix, options.rank, q=power, rng=21)
                    samples = []
                    for _ in range(options.repeats):
                        start = perf_counter()
                        u, s, v = rsvd(matrix, options.rank, q=power, rng=21)
                        samples.append(perf_counter() - start)
                    residual = float(np.linalg.norm(dense - (u * s) @ v.conj().T))
                    np.testing.assert_allclose(
                        u.conj().T @ u, np.eye(options.rank), atol=1e-10
                    )
                    np.testing.assert_allclose(
                        v.conj().T @ v, np.eye(options.rank), atol=1e-10
                    )
                    assert np.isfinite(residual) and residual <= np.linalg.norm(
                        dense
                    ) * (1 + 1e-12)
                    tracemalloc.start()
                    try:
                        rsvd(matrix, options.rank, q=power, rng=21)
                        _, peak = tracemalloc.get_traced_memory()
                    finally:
                        tracemalloc.stop()
                    cases.append(
                        {
                            "dtype": np.dtype(dtype).name,
                            "storage": storage,
                            "q": power,
                            "samples_seconds": samples,
                            "median_seconds": statistics.median(samples),
                            "tracked_peak_bytes": peak,
                            "residual_norm": residual,
                        }
                    )
        report = {
            "environment": environment(),
            "configuration": {
                key: value for key, value in vars(options).items() if key != "output"
            },
            "scope": "One numerical thread; fixed RNG seed; input preparation and validation excluded from timing. Memory measured in a separate call with tracemalloc; excludes untracked native allocations.",
            "cases": cases,
        }
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if options.output is not None:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(encoded)
    else:
        print(encoded, end="")
    return report


if __name__ == "__main__":
    main()
