"""CLI configuration, thread controls, and reproducibility metadata."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import subprocess
import sys

BACKENDS = ("sppy_C", "sppy_F", "scipy_csr", "scipy_csc")
ARRAY_OPERATIONS = (
    "construct-triplets",
    "construct-compressed",
    "copy",
    "copy-assign",
    "gather",
    "slice",
    "transpose",
    "adjoint",
    "add",
    "multiply",
    "matvec",
    "matmat",
    "sparse-matmul",
    "sum",
    "column-sum",
    "mean",
    "norm",
    "compressed-export",
    "copy-prune",
)
SOLVER_OPERATIONS = ("lu-factor", "lu-solve", "cg", "bicgstab", "triangular-solve")
OPERATIONS = ARRAY_OPERATIONS + SOLVER_OPERATIONS


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return number


def nonnegative_float(value):
    import math

    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative and finite")
    return number


def parser(description):
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--size", type=positive_int, default=1000)
    result.add_argument(
        "--entries",
        type=nonnegative_int,
        default=5000,
        help="Sampled triplets for the random pattern (duplicates are summed)",
    )
    result.add_argument("--pattern", choices=["random", "banded"], default="random")
    result.add_argument(
        "--bandwidth",
        type=nonnegative_int,
        default=2,
        help="Off-diagonals on each side for the banded pattern",
    )
    result.add_argument(
        "--dtype",
        choices=["float32", "float64", "complex64", "complex128"],
        default="float64",
    )
    result.add_argument(
        "--columns",
        type=positive_int,
        default=8,
        help="Dense RHS columns for matmat and reused LU solves",
    )
    result.add_argument("--repeats", type=positive_int, default=7)
    result.add_argument("--warmups", type=nonnegative_int, default=1)
    result.add_argument("--seed", type=int, default=21)
    result.add_argument(
        "--threads",
        type=positive_int,
        default=1,
        help="Eigen, OpenMP, and BLAS thread limit",
    )
    result.add_argument(
        "--rtol",
        type=nonnegative_float,
        help="Iterative tolerance; defaults to 1e-5 for single precision, 1e-8 otherwise",
    )
    result.add_argument("--maxiter", type=positive_int, default=1000)
    return result


def validate_options(arguments, options):
    if options.size > 2**31 - 1 or options.entries > 2**31 - 1:
        arguments.error("dimensions and triplet counts must fit signed 32-bit storage")
    if options.rtol == 0:
        arguments.error("--rtol must be positive")


@contextmanager
def numerical_threads(count):
    """Control already-loaded native libraries and restore their thread settings."""
    import sppy
    from threadpoolctl import threadpool_limits

    previous = sppy.get_num_threads()
    with threadpool_limits(limits=count):
        sppy.set_num_threads(count)
        try:
            yield
        finally:
            sppy.set_num_threads(previous)


def environment():
    import numpy as np
    import scipy
    import sppy
    from threadpoolctl import threadpool_info

    root = Path(__file__).resolve().parents[1]
    revision, dirty = None, None
    try:
        revision_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if revision_result.returncode == status_result.returncode == 0:
            revision, dirty = revision_result.stdout.strip(), bool(status_result.stdout)
    except (OSError, subprocess.TimeoutExpired):
        pass
    cpu = platform.processor() or platform.machine()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    return {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sppy": sppy.__version__,
        "platform": platform.platform(),
        "cpu": cpu,
        "available_cpus": len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else os.cpu_count(),
        "openmp_enabled": sppy.openmp_enabled,
        "runtime_simd": getattr(sppy._native, "runtime_simd", "baseline"),
        "eigen_threads": sppy.get_num_threads(),
        "threadpools": [
            {
                **{key: value for key, value in pool.items() if key != "filepath"},
                "library": Path(pool["filepath"]).name,
            }
            for pool in threadpool_info()
        ],
        "git_revision": revision,
        "git_dirty": dirty,
    }
