"""Built-in Eigen sparse solvers, preconditioners, orderings, and decompositions."""

from sppy.linalg.core import norm, rsvd
from sppy.linalg.general_linear_operator import GeneralLinearOperator
from sppy.linalg.solvers import (
    ComputationInfo,
    IterativeResult,
    Preconditioner,
    SimplicialLDLT,
    SimplicialLLT,
    SparseFactorization,
    SparseLU,
    SparseQR,
    bicgstab,
    cg,
    factorize,
    lscg,
    ordering,
    solve,
    triangular_solve,
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
