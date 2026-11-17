from libc.stdint cimport int64_t


cdef extern from "include/SparseVectorExt.h":
   cdef cppclass SparseVectorExt[T]:
      int nonZeros() except +
      int rows() except +
      int size() except +
      SparseVectorExt() except +
      SparseVectorExt(int) except +
      SparseVectorExt(SparseVectorExt[T]) except +
      SparseVectorExt[T] abs() except +
      SparseVectorExt[T] add(SparseVectorExt[T]&) except +
      SparseVectorExt[T] hadamard(SparseVectorExt[T]&) except +
      SparseVectorExt[T] negate() except +
      SparseVectorExt[T] subtract(SparseVectorExt[T]&) except +
      T coeff(int) except +
      T dot(SparseVectorExt[T]&) except +
      T sumValues() except +
      void fill(T) except +
      void insertVal(int, T) except +
      void nonZeroInds(int64_t*) except +
      void nonZeroVals(T*) except +
      void reserve(int) except +
      void scalarMultiply(double) except +
      void slice(int*, int, SparseVectorExt[T]*) except +

cdef template[DataType] class csarray1d:
    cdef SparseVectorExt[DataType] *thisPtr
