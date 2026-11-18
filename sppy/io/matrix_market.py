"""Matrix Market interoperability using SciPy's maintained reader and writer."""

import numpy as np

from sppy import SparseArray


def mmwrite(filename, A, comment="", field=None, precision=None):
    """Write a two-dimensional sparse array to a Matrix Market file.

    SciPy is required. Paths and open binary streams are accepted.
    """
    from scipy.io import mmwrite as scipy_mmwrite

    if A.ndim != 2:
        raise ValueError("Matrix Market requires a two-dimensional array")
    scipy_mmwrite(
        filename,
        A.to_scipy_csc(),
        comment=comment,
        field=field,
        precision=precision,
        symmetry="general",
    )


def mmread(filename, *, order="C"):
    """Read a Matrix Market file into a sparse array.

    SciPy is required. Symmetric files, comments, and compressed files are
    handled by SciPy. Real, integer, and complex files are supported.
    """
    from scipy.io import mmread as scipy_mmread

    matrix = scipy_mmread(filename, spmatrix=False)
    if matrix.dtype.kind not in "iufc":
        raise ValueError(f"Unsupported Matrix Market dtype: {matrix.dtype}")
    dtype = (
        np.int64
        if matrix.dtype.kind in "iu"
        else (np.complex128 if matrix.dtype.kind == "c" else np.float64)
    )
    return SparseArray(matrix, dtype=dtype, order=order)
