"""Built-in Eigen sparse solvers, preconditioners, orderings, and decompositions."""

from sppy.linalg.general_linear_operator import GeneralLinearOperator
from sppy.linalg.core import rsvd, norm
from sppy.linalg.solvers import (
    ComputationInfo,
    IterativeResult,
    Preconditioner,
    SparseFactorization,
    SimplicialLLT,
    SimplicialLDLT,
    SparseLU,
    SparseQR,
    factorize,
    solve,
    ordering,
    triangular_solve,
    cg,
    bicgstab,
    lscg,
)

__all__ = [
    "GeneralLinearOperator",
    "rsvd",
    "norm",
    "ComputationInfo",
    "IterativeResult",
    "Preconditioner",
    "SparseFactorization",
    "SimplicialLLT",
    "SimplicialLDLT",
    "SparseLU",
    "SparseQR",
    "factorize",
    "solve",
    "ordering",
    "triangular_solve",
    "cg",
    "bicgstab",
    "lscg",
]
