"""Shared deterministic data; each test receives independent arrays and generators."""

from pathlib import Path
import os
import shlex
import shutil
import subprocess

import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(21)


@pytest.fixture(params=["C", "F"])
def order(request):
    return request.param


@pytest.fixture(params=[np.int8, np.int16, np.int32, np.int64, np.float32, np.float64])
def dtype(request):
    return np.dtype(request.param)


@pytest.fixture(
    params=[
        "matrix",
        "vector",
        "zero_matrix",
        "zero_vector",
        "empty_matrix",
        "empty_vector",
    ]
)
def dense(request):
    cases = {
        "matrix": [[0.0, 1.0, 0.0, -2.0], [3.0, 0.0, 4.0, 0.0], [0.0, -5.0, 0.0, 6.0]],
        "vector": [0.0, 3.0, -2.0, 0.0, 1.0],
        "zero_matrix": np.zeros((3, 4)),
        "zero_vector": np.zeros(5),
        "empty_matrix": np.zeros((0, 0)),
        "empty_vector": np.zeros(0),
    }
    return np.array(cases[request.param])


@pytest.fixture(scope="session")
def native_executable(request, tmp_path_factory):
    """Compile header-level contracts once; pytest runs and reports each executable."""
    root = Path(__file__).resolve().parents[1]
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    if not compiler or shutil.which(compiler[0]) is None:
        pytest.skip("Native tests require a C++ compiler (set CXX)")
    if Path(compiler[-1]).stem.lower() == "cl":
        pytest.skip("This native test harness uses GCC/Clang compiler flags")
    candidates = [
        Path(os.environ.get("EIGEN_INCLUDE_DIR", root / "vendor/eigen")),
        Path("/usr/include/eigen3"),
        Path("/usr/local/include/eigen3"),
        Path("/opt/homebrew/include/eigen3"),
    ]
    eigen = next((path for path in candidates if (path / "Eigen/Core").is_file()), None)
    if eigen is None:
        pytest.skip("Set EIGEN_INCLUDE_DIR to run native header tests")
    executable = tmp_path_factory.mktemp("native") / request.param
    command = [
        *compiler,
        "-std=c++17",
        "-UNDEBUG",
        f"-I{eigen}",
        f"-I{root / 'sppy/include'}",
        str(root / "tests/native" / f"{request.param}.cpp"),
        "-o",
        str(executable),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    if result.returncode:
        pytest.fail(
            f"Native test compilation failed:\n{result.stdout}\n{result.stderr}"
        )
    return executable
