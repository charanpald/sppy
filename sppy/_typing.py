"""Shared scalar types for the Python interface to native sparse storage."""

from typing import Any, TypeAlias

import numpy as np

Scalar: TypeAlias = int | float | complex | np.number[Any]
