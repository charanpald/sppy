"""Shared deterministic data; each test receives independent arrays and generators."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray


@pytest.fixture
def rng() -> np.random.Generator:
    """Create an independent reproducible random generator."""
    return np.random.default_rng(21)


@pytest.fixture(params=["C", "F"])
def order(request: pytest.FixtureRequest) -> str:
    """Provide each supported sparse storage order."""
    return request.param


@pytest.fixture(
    params=[
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.float32,
        np.float64,
        np.complex64,
        np.complex128,
    ]
)
def dtype(request: pytest.FixtureRequest) -> np.dtype[Any]:
    """Provide each supported scalar dtype."""
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
def dense(request: pytest.FixtureRequest) -> NDArray[Any]:
    """Provide independent dense test data, including empty and zero arrays."""
    cases = {
        "matrix": [[0.0, 1.0, 0.0, -2.0], [3.0, 0.0, 4.0, 0.0], [0.0, -5.0, 0.0, 6.0]],
        "vector": [0.0, 3.0, -2.0, 0.0, 1.0],
        "zero_matrix": np.zeros((3, 4)),
        "zero_vector": np.zeros(5),
        "empty_matrix": np.zeros((0, 0)),
        "empty_vector": np.zeros(0),
    }
    return np.array(cases[request.param])
