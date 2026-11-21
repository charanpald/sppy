"""Typed signatures for the pybind11 boundary; dtype-specialized APIs share contracts."""

import typing

import numpy
import numpy.typing

from sppy._typing import Scalar

_T = typing.TypeVar("_T", bound=numpy.generic)
_Triplets: typing.TypeAlias = tuple[
    int,
    int,
    numpy.typing.NDArray[numpy.int32],
    numpy.typing.NDArray[numpy.int32],
    numpy.typing.NDArray[_T],
]
openmp_enabled: bool
runtime_simd: str

class _Matrix(typing.Generic[_T]):
    def __init__(
        self,
        arg0: typing.SupportsInt | typing.SupportsIndex,
        arg1: typing.SupportsInt | typing.SupportsIndex,
    ) -> None: ...
    def add(self, arg0: _Matrix[_T]) -> _Matrix[_T]: ...
    def adjoint(self) -> _Matrix[_T]: ...
    def all_finite(self) -> bool: ...
    def assemble(
        self,
        arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
        arg1: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
        arg2: typing.Annotated[numpy.typing.ArrayLike, _T],
    ) -> None: ...
    def assign(
        self,
        arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
        arg1: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
        arg2: typing.Annotated[numpy.typing.ArrayLike, _T],
    ) -> None: ...
    def coeff(
        self,
        arg0: typing.SupportsInt | typing.SupportsIndex,
        arg1: typing.SupportsInt | typing.SupportsIndex,
    ) -> float: ...
    def compress(self) -> None: ...
    def compressed_arrays(
        self,
    ) -> tuple[
        numpy.typing.NDArray[_T],
        numpy.typing.NDArray[numpy.int32],
        numpy.typing.NDArray[numpy.int32],
    ]: ...
    def coordinates(
        self,
    ) -> tuple[
        numpy.typing.NDArray[numpy.int32], numpy.typing.NDArray[numpy.int32]
    ]: ...
    def copy(self) -> _Matrix[_T]: ...
    def dense_matmul(
        self, arg0: typing.Annotated[numpy.typing.ArrayLike, _T]
    ) -> numpy.typing.NDArray[_T]: ...
    @staticmethod
    def from_compressed(
        arg0: typing.SupportsInt | typing.SupportsIndex,
        arg1: typing.SupportsInt | typing.SupportsIndex,
        arg2: typing.Annotated[numpy.typing.ArrayLike, _T],
        arg3: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
        arg4: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    ) -> _Matrix[_T]: ...
    def gather(
        self,
        arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
        arg1: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    ) -> numpy.typing.NDArray[_T]: ...
    def gather_signed(
        self,
        arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.int64],
        arg1: typing.Annotated[numpy.typing.ArrayLike, numpy.int64],
    ) -> numpy.typing.NDArray[_T]: ...
    def matmul(self, arg0: _Matrix[_T]) -> _Matrix[_T]: ...
    def multiply(self, arg0: _Matrix[_T]) -> _Matrix[_T]: ...
    def norm(self) -> float: ...
    def prune(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None: ...
    def reserve(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None: ...
    def reserve_per_outer(
        self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.int32]
    ) -> None: ...
    def resize(
        self,
        arg0: typing.SupportsInt | typing.SupportsIndex,
        arg1: typing.SupportsInt | typing.SupportsIndex,
        arg2: bool,
    ) -> None: ...
    def selfadjoint(self, arg0: bool) -> _Matrix[_T]: ...
    def set(
        self,
        arg0: typing.SupportsInt | typing.SupportsIndex,
        arg1: typing.SupportsInt | typing.SupportsIndex,
        arg2: typing.SupportsFloat | typing.SupportsIndex,
    ) -> None: ...
    def set_zero(self) -> None: ...
    def slice_basic(
        self,
        arg0: typing.SupportsInt | typing.SupportsIndex,
        arg1: typing.SupportsInt | typing.SupportsIndex,
        arg2: typing.SupportsInt | typing.SupportsIndex,
        arg3: typing.SupportsInt | typing.SupportsIndex,
        arg4: typing.SupportsInt | typing.SupportsIndex,
        arg5: typing.SupportsInt | typing.SupportsIndex,
        arg6: typing.SupportsInt | typing.SupportsIndex,
    ) -> _Matrix[_T]: ...
    def squeeze(self) -> None: ...
    def subtract(self, arg0: _Matrix[_T]) -> _Matrix[_T]: ...
    def sum(
        self, arg0: typing.SupportsInt | typing.SupportsIndex
    ) -> numpy.typing.NDArray[typing.Any]: ...
    def to_column_major(self) -> _Matrix[_T]: ...
    def to_row_major(self) -> _Matrix[_T]: ...
    def transpose(self) -> _Matrix[_T]: ...
    def triangular(self, arg0: bool, arg1: bool) -> _Matrix[_T]: ...
    def uncompress(self) -> None: ...
    def values(self) -> numpy.typing.NDArray[_T]: ...
    @property
    def inner_size(self) -> int: ...
    @property
    def is_compressed(self) -> bool: ...
    @property
    def nnz(self) -> int: ...
    @property
    def outer_size(self) -> int: ...
    @property
    def shape(self) -> tuple[int, int]: ...

class _Factor(typing.Generic[_T]):
    def __init__(
        self,
        arg0: str,
        arg1: str,
        arg2: typing.SupportsFloat | typing.SupportsIndex,
        arg3: typing.SupportsFloat | typing.SupportsIndex,
        arg4: typing.SupportsFloat | typing.SupportsIndex,
    ) -> None: ...
    def analyze(
        self,
        arg0: typing.SupportsInt | typing.SupportsIndex,
        arg1: typing.SupportsInt | typing.SupportsIndex,
        arg2: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
        arg3: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
        arg4: typing.Annotated[numpy.typing.ArrayLike, _T],
    ) -> None: ...
    def analyze_matrix(self, arg0: object) -> None: ...
    def apply_q(
        self, arg0: typing.Annotated[numpy.typing.ArrayLike, _T], arg1: bool
    ) -> numpy.typing.NDArray[_T]: ...
    def compute_matrix(self, arg0: object) -> int: ...
    def determinant(self) -> Scalar: ...
    def factor(self, arg0: str) -> numpy.typing.NDArray[typing.Any] | _Triplets[_T]: ...
    def factorize(
        self,
        arg0: typing.SupportsInt | typing.SupportsIndex,
        arg1: typing.SupportsInt | typing.SupportsIndex,
        arg2: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
        arg3: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
        arg4: typing.Annotated[numpy.typing.ArrayLike, _T],
    ) -> int: ...
    def factorize_matrix(self, arg0: object) -> int: ...
    def slogdet(self) -> tuple[Scalar, float]: ...
    def solve(
        self, arg0: typing.Annotated[numpy.typing.ArrayLike, _T]
    ) -> numpy.typing.NDArray[_T]: ...
    def solve_sparse(
        self,
        arg0: typing.SupportsInt | typing.SupportsIndex,
        arg1: typing.SupportsInt | typing.SupportsIndex,
        arg2: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
        arg3: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
        arg4: typing.Annotated[numpy.typing.ArrayLike, _T],
    ) -> _Triplets[_T]: ...
    @property
    def rank(self) -> int: ...

class _Preconditioner(typing.Generic[_T]):
    def __init__(
        self,
        arg0: typing.SupportsInt | typing.SupportsIndex,
        arg1: typing.SupportsInt | typing.SupportsIndex,
        arg2: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
        arg3: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
        arg4: typing.Annotated[numpy.typing.ArrayLike, _T],
        arg5: str,
        arg6: typing.SupportsFloat | typing.SupportsIndex,
        arg7: typing.SupportsInt | typing.SupportsIndex,
        arg8: typing.SupportsFloat | typing.SupportsIndex,
    ) -> None: ...
    def solve(
        self, arg0: typing.Annotated[numpy.typing.ArrayLike, _T]
    ) -> numpy.typing.NDArray[_T]: ...

class Factor_complex128(_Factor[numpy.complex128]): ...
class Factor_complex64(_Factor[numpy.complex64]): ...
class Factor_float32(_Factor[numpy.float32]): ...
class Factor_float64(_Factor[numpy.float64]): ...
class Preconditioner_complex128(_Preconditioner[numpy.complex128]): ...
class Preconditioner_complex64(_Preconditioner[numpy.complex64]): ...
class Preconditioner_float32(_Preconditioner[numpy.float32]): ...
class Preconditioner_float64(_Preconditioner[numpy.float64]): ...
class complex128_C(_Matrix[numpy.complex128]): ...
class complex128_F(_Matrix[numpy.complex128]): ...
class complex64_C(_Matrix[numpy.complex64]): ...
class complex64_F(_Matrix[numpy.complex64]): ...
class float32_C(_Matrix[numpy.float32]): ...
class float32_F(_Matrix[numpy.float32]): ...
class float64_C(_Matrix[numpy.float64]): ...
class float64_F(_Matrix[numpy.float64]): ...
class int16_C(_Matrix[numpy.int16]): ...
class int16_F(_Matrix[numpy.int16]): ...
class int32_C(_Matrix[numpy.int32]): ...
class int32_F(_Matrix[numpy.int32]): ...
class int64_C(_Matrix[numpy.int64]): ...
class int64_F(_Matrix[numpy.int64]): ...
class int8_C(_Matrix[numpy.int8]): ...
class int8_F(_Matrix[numpy.int8]): ...

def get_num_threads() -> int: ...
def iterate_complex128(
    arg0: typing.SupportsInt | typing.SupportsIndex,
    arg1: typing.SupportsInt | typing.SupportsIndex,
    arg2: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    arg3: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    arg4: typing.Annotated[numpy.typing.ArrayLike, numpy.complex128],
    arg5: typing.Annotated[numpy.typing.ArrayLike, numpy.complex128],
    arg6: typing.Annotated[numpy.typing.ArrayLike, numpy.complex128],
    arg7: str,
    arg8: str,
    arg9: typing.SupportsInt | typing.SupportsIndex,
    arg10: typing.SupportsFloat | typing.SupportsIndex,
    arg11: typing.SupportsFloat | typing.SupportsIndex,
    arg12: typing.SupportsInt | typing.SupportsIndex,
    arg13: typing.SupportsFloat | typing.SupportsIndex,
) -> tuple[numpy.typing.NDArray[numpy.complex128], int, int, float]: ...
def iterate_complex64(
    arg0: typing.SupportsInt | typing.SupportsIndex,
    arg1: typing.SupportsInt | typing.SupportsIndex,
    arg2: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    arg3: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    arg4: typing.Annotated[numpy.typing.ArrayLike, numpy.complex64],
    arg5: typing.Annotated[numpy.typing.ArrayLike, numpy.complex64],
    arg6: typing.Annotated[numpy.typing.ArrayLike, numpy.complex64],
    arg7: str,
    arg8: str,
    arg9: typing.SupportsInt | typing.SupportsIndex,
    arg10: typing.SupportsFloat | typing.SupportsIndex,
    arg11: typing.SupportsFloat | typing.SupportsIndex,
    arg12: typing.SupportsInt | typing.SupportsIndex,
    arg13: typing.SupportsFloat | typing.SupportsIndex,
) -> tuple[numpy.typing.NDArray[numpy.complex64], int, int, float]: ...
def iterate_float32(
    arg0: typing.SupportsInt | typing.SupportsIndex,
    arg1: typing.SupportsInt | typing.SupportsIndex,
    arg2: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    arg3: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    arg4: typing.Annotated[numpy.typing.ArrayLike, numpy.float32],
    arg5: typing.Annotated[numpy.typing.ArrayLike, numpy.float32],
    arg6: typing.Annotated[numpy.typing.ArrayLike, numpy.float32],
    arg7: str,
    arg8: str,
    arg9: typing.SupportsInt | typing.SupportsIndex,
    arg10: typing.SupportsFloat | typing.SupportsIndex,
    arg11: typing.SupportsFloat | typing.SupportsIndex,
    arg12: typing.SupportsInt | typing.SupportsIndex,
    arg13: typing.SupportsFloat | typing.SupportsIndex,
) -> tuple[numpy.typing.NDArray[numpy.float32], int, int, float]: ...
def iterate_float64(
    arg0: typing.SupportsInt | typing.SupportsIndex,
    arg1: typing.SupportsInt | typing.SupportsIndex,
    arg2: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    arg3: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    arg4: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    arg5: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    arg6: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    arg7: str,
    arg8: str,
    arg9: typing.SupportsInt | typing.SupportsIndex,
    arg10: typing.SupportsFloat | typing.SupportsIndex,
    arg11: typing.SupportsFloat | typing.SupportsIndex,
    arg12: typing.SupportsInt | typing.SupportsIndex,
    arg13: typing.SupportsFloat | typing.SupportsIndex,
) -> tuple[numpy.typing.NDArray[numpy.float64], int, int, float]: ...
def iterate_matrix_complex128(
    arg0: object,
    arg1: typing.Annotated[numpy.typing.ArrayLike, numpy.complex128],
    arg2: typing.Annotated[numpy.typing.ArrayLike, numpy.complex128],
    arg3: str,
    arg4: str,
    arg5: typing.SupportsInt | typing.SupportsIndex,
    arg6: typing.SupportsFloat | typing.SupportsIndex,
    arg7: typing.SupportsFloat | typing.SupportsIndex,
    arg8: typing.SupportsInt | typing.SupportsIndex,
    arg9: typing.SupportsFloat | typing.SupportsIndex,
) -> tuple[numpy.typing.NDArray[numpy.complex128], int, int, float]: ...
def iterate_matrix_complex64(
    arg0: object,
    arg1: typing.Annotated[numpy.typing.ArrayLike, numpy.complex64],
    arg2: typing.Annotated[numpy.typing.ArrayLike, numpy.complex64],
    arg3: str,
    arg4: str,
    arg5: typing.SupportsInt | typing.SupportsIndex,
    arg6: typing.SupportsFloat | typing.SupportsIndex,
    arg7: typing.SupportsFloat | typing.SupportsIndex,
    arg8: typing.SupportsInt | typing.SupportsIndex,
    arg9: typing.SupportsFloat | typing.SupportsIndex,
) -> tuple[numpy.typing.NDArray[numpy.complex64], int, int, float]: ...
def iterate_matrix_float32(
    arg0: object,
    arg1: typing.Annotated[numpy.typing.ArrayLike, numpy.float32],
    arg2: typing.Annotated[numpy.typing.ArrayLike, numpy.float32],
    arg3: str,
    arg4: str,
    arg5: typing.SupportsInt | typing.SupportsIndex,
    arg6: typing.SupportsFloat | typing.SupportsIndex,
    arg7: typing.SupportsFloat | typing.SupportsIndex,
    arg8: typing.SupportsInt | typing.SupportsIndex,
    arg9: typing.SupportsFloat | typing.SupportsIndex,
) -> tuple[numpy.typing.NDArray[numpy.float32], int, int, float]: ...
def iterate_matrix_float64(
    arg0: object,
    arg1: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    arg2: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    arg3: str,
    arg4: str,
    arg5: typing.SupportsInt | typing.SupportsIndex,
    arg6: typing.SupportsFloat | typing.SupportsIndex,
    arg7: typing.SupportsFloat | typing.SupportsIndex,
    arg8: typing.SupportsInt | typing.SupportsIndex,
    arg9: typing.SupportsFloat | typing.SupportsIndex,
) -> tuple[numpy.typing.NDArray[numpy.float64], int, int, float]: ...
def ordering_complex128(
    arg0: typing.SupportsInt | typing.SupportsIndex,
    arg1: typing.SupportsInt | typing.SupportsIndex,
    arg2: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    arg3: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    arg4: typing.Annotated[numpy.typing.ArrayLike, numpy.complex128],
    arg5: str,
) -> numpy.typing.NDArray[numpy.int32]: ...
def ordering_complex64(
    arg0: typing.SupportsInt | typing.SupportsIndex,
    arg1: typing.SupportsInt | typing.SupportsIndex,
    arg2: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    arg3: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    arg4: typing.Annotated[numpy.typing.ArrayLike, numpy.complex64],
    arg5: str,
) -> numpy.typing.NDArray[numpy.int32]: ...
def ordering_float32(
    arg0: typing.SupportsInt | typing.SupportsIndex,
    arg1: typing.SupportsInt | typing.SupportsIndex,
    arg2: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    arg3: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    arg4: typing.Annotated[numpy.typing.ArrayLike, numpy.float32],
    arg5: str,
) -> numpy.typing.NDArray[numpy.int32]: ...
def ordering_float64(
    arg0: typing.SupportsInt | typing.SupportsIndex,
    arg1: typing.SupportsInt | typing.SupportsIndex,
    arg2: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    arg3: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    arg4: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    arg5: str,
) -> numpy.typing.NDArray[numpy.int32]: ...
def set_num_threads(arg0: typing.SupportsInt | typing.SupportsIndex) -> None: ...
def triangular_solve_complex128(
    arg0: typing.SupportsInt | typing.SupportsIndex,
    arg1: typing.SupportsInt | typing.SupportsIndex,
    arg2: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    arg3: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    arg4: typing.Annotated[numpy.typing.ArrayLike, numpy.complex128],
    arg5: typing.Annotated[numpy.typing.ArrayLike, numpy.complex128],
    arg6: bool,
    arg7: bool,
) -> numpy.typing.NDArray[numpy.complex128]: ...
def triangular_solve_complex64(
    arg0: typing.SupportsInt | typing.SupportsIndex,
    arg1: typing.SupportsInt | typing.SupportsIndex,
    arg2: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    arg3: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    arg4: typing.Annotated[numpy.typing.ArrayLike, numpy.complex64],
    arg5: typing.Annotated[numpy.typing.ArrayLike, numpy.complex64],
    arg6: bool,
    arg7: bool,
) -> numpy.typing.NDArray[numpy.complex64]: ...
def triangular_solve_float32(
    arg0: typing.SupportsInt | typing.SupportsIndex,
    arg1: typing.SupportsInt | typing.SupportsIndex,
    arg2: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    arg3: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    arg4: typing.Annotated[numpy.typing.ArrayLike, numpy.float32],
    arg5: typing.Annotated[numpy.typing.ArrayLike, numpy.float32],
    arg6: bool,
    arg7: bool,
) -> numpy.typing.NDArray[numpy.float32]: ...
def triangular_solve_float64(
    arg0: typing.SupportsInt | typing.SupportsIndex,
    arg1: typing.SupportsInt | typing.SupportsIndex,
    arg2: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    arg3: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    arg4: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    arg5: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    arg6: bool,
    arg7: bool,
) -> numpy.typing.NDArray[numpy.float64]: ...
def triangular_solve_matrix_complex128(
    arg0: object,
    arg1: typing.Annotated[numpy.typing.ArrayLike, numpy.complex128],
    arg2: bool,
    arg3: bool,
) -> numpy.typing.NDArray[numpy.complex128]: ...
def triangular_solve_matrix_complex64(
    arg0: object,
    arg1: typing.Annotated[numpy.typing.ArrayLike, numpy.complex64],
    arg2: bool,
    arg3: bool,
) -> numpy.typing.NDArray[numpy.complex64]: ...
def triangular_solve_matrix_float32(
    arg0: object,
    arg1: typing.Annotated[numpy.typing.ArrayLike, numpy.float32],
    arg2: bool,
    arg3: bool,
) -> numpy.typing.NDArray[numpy.float32]: ...
def triangular_solve_matrix_float64(
    arg0: object,
    arg1: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    arg2: bool,
    arg3: bool,
) -> numpy.typing.NDArray[numpy.float64]: ...
def _gemm_float32(
    arg0: numpy.typing.ArrayLike,
    arg1: numpy.typing.ArrayLike,
    arg2: numpy.typing.ArrayLike,
    arg3: Scalar,
    arg4: bool,
    arg5: int,
) -> numpy.typing.NDArray[numpy.float32]: ...
def _gemm_float64(
    arg0: numpy.typing.ArrayLike,
    arg1: numpy.typing.ArrayLike,
    arg2: numpy.typing.ArrayLike,
    arg3: Scalar,
    arg4: bool,
    arg5: int,
) -> numpy.typing.NDArray[numpy.float64]: ...
def _gemm_complex64(
    arg0: numpy.typing.ArrayLike,
    arg1: numpy.typing.ArrayLike,
    arg2: numpy.typing.ArrayLike,
    arg3: Scalar,
    arg4: bool,
    arg5: int,
) -> numpy.typing.NDArray[numpy.complex64]: ...
def _gemm_complex128(
    arg0: numpy.typing.ArrayLike,
    arg1: numpy.typing.ArrayLike,
    arg2: numpy.typing.ArrayLike,
    arg3: Scalar,
    arg4: bool,
    arg5: int,
) -> numpy.typing.NDArray[numpy.complex128]: ...
