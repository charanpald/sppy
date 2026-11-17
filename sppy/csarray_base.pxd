from libc.stdint cimport int64_t
import numpy
cimport numpy
from libcpp.vector cimport vector

cdef extern from *:
    ctypedef int colMajor "0"
    ctypedef int rowMajor "1"

cdef extern from "include/SparseMatrixExt.h":
   cdef cppclass SparseMatrixExt[T, S]:
      double norm() except +
      int biCGSTAB(T*, int, T*, int, double) except +
      int64_t cols() except +
      int64_t nonZeros() except +
      int64_t rows() except +
      int64_t size() except +
      SparseMatrixExt() except +
      SparseMatrixExt(int, int) except +
      SparseMatrixExt(SparseMatrixExt[T, S]) except +
      SparseMatrixExt[T, S] abs() except +
      SparseMatrixExt[T, S] add(SparseMatrixExt[T, S]&) except +
      SparseMatrixExt[T, S] dot(SparseMatrixExt[T, S]&) except +
      SparseMatrixExt[T, S] hadamard(SparseMatrixExt[T, S]&) except +
      SparseMatrixExt[T, S] negate() except +
      SparseMatrixExt[T, S] submatrix(int, int, int, int) except +
      SparseMatrixExt[T, S] subtract(SparseMatrixExt[T, S]&) except +
      SparseMatrixExt[T, S] trans() except +
      T coeff(int, int) except +
      T sum() except +
      T sumValues() except +
      vector[int] getIndsCol(int) except +
      vector[int] getIndsRow(int) except +
      void dot1d(double*, double*) except + nogil
      void dot2d(double*, int, double*) except + nogil
      void dotSub1d(double*, int,  int, double*) except + nogil
      void dotSub2d(double*, int,  int, int, double*) except + nogil
      void fill(T) except +
      void insertVal(int, int, T) except +
      void makeCompressed() except +
      void nonZeroInds(int*, int*) except +
      void nonZeroVals(T*) except +
      void printValues() except +
      void prune(double, double) except +
      void putSorted2(int64_t*, int64_t*, T, int, int64_t*) except +
      void putSorted(int64_t*, int64_t*, T*, int, int64_t*) except +
      void putUsingTriplets2(int*, int*, T, int) except +
      void putUsingTriplets(int*, int*, T*, int) except +
      void reserve(int) except +
      void scalarMultiply(double) except +
      void setZero() except +
      void slice(int*, int, int*, int, SparseMatrixExt[T, S]*) except +
      void unsafeInsertVal2(int, int, T) except +
      void unsafeInsertVal(int, int, T) except +

cdef template[DataType, StorageType] class csarray:
    cdef SparseMatrixExt[DataType, StorageType] *thisPtr
