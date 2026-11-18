"""Shared CLI, deterministic input generation, and standard-library profiling."""

import argparse
import cProfile
from pathlib import Path
import pstats
from collections.abc import Callable

import numpy as np
from scipy import sparse

import sppy


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def parser(description: str) -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--size", type=positive_int, default=1000)
    result.add_argument("--entries", type=positive_int, default=5000)
    result.add_argument("--repeats", type=positive_int, default=5)
    result.add_argument("--seed", type=int, default=21)
    result.add_argument("--order", choices=["C", "F"], default="C")
    result.add_argument(
        "--output", type=Path, help="Optional .prof file for pstats or a profile viewer"
    )
    return result


def make_matrix(size: int, entries: int, seed: int, order: str = "C"):
    rng = np.random.default_rng(seed)
    rows = rng.integers(0, size, entries, dtype=np.int32)
    columns = rng.integers(0, size, entries, dtype=np.int32)
    values = rng.normal(size=entries)
    coordinates = sparse.coo_matrix((values, (rows, columns)), shape=(size, size))
    return sppy.asarray(coordinates, order=order), rng


def run_profile(
    workload: Callable[[], object], repeats: int, output: Path | None
) -> None:
    """Profile only the repeated operation; input creation happens in the caller."""
    profiler = cProfile.Profile()
    for _ in range(repeats):
        profiler.runcall(workload)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        profiler.dump_stats(str(output))
    pstats.Stats(profiler).strip_dirs().sort_stats("cumulative").print_stats(20)
