
from libcpp.vector cimport vector

cdef extern from *:
    ctypedef int colMajor "0" 
    ctypedef int rowMajor "1" 

cdef extern from "include/SparseMatrixExt.h":  
   cdef cppclass SparseMatrixExt[T, S]:  
      SparseMatrixExt() 
      SparseMatrixExt(SparseMatrixExt[T, S]) 
      SparseMatrixExt(int, int)
      double norm()
      int cols() 
      int nonZeros()
      int rows()
      int size() 
      SparseMatrixExt[T, S] abs()
      SparseMatrixExt[T, S] add(SparseMatrixExt[T, S]&)
      SparseMatrixExt[T, S] dot(SparseMatrixExt[T, S]&)
      SparseMatrixExt[T, S] hadamard(SparseMatrixExt[T, S]&)
      SparseMatrixExt[T, S] negate()
      SparseMatrixExt[T, S] subtract(SparseMatrixExt[T, S]&)
      SparseMatrixExt[T, S] trans()
      T coeff(int, int)
      T sum()
      T sumValues()
      void insertVal(int, int, T) 
      void fill(T)
      void makeCompressed()
      void nonZeroInds(long*, long*)
      void nonZeroVals(T*)
      void printValues()
      void reserve(int)
      void scalarMultiply(double)
      void slice(int*, int, int*, int, SparseMatrixExt[T, S]*) 
      vector[long] getIndsRow(int)
      vector[long] getIndsCol(int)
      void setZero()
      void unsafeInsertVal(int, int, T)
      void unsafeInsertVal2(int, int, T)
      void putSorted(long*, long*, T*, int, long*)
      void putSorted2(long*, long*, T, int, long*) 
      
cdef template[DataType, StorageType] class csarray:
    cdef SparseMatrixExt[DataType, StorageType] *thisPtr     
