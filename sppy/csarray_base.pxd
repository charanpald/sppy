import numpy 
cimport numpy 
from libcpp.vector cimport vector

cdef extern from *:
    ctypedef int colMajor "0" 
    ctypedef int rowMajor "1" 

cdef extern from "include/SparseMatrixExt.h":  
   cdef cppclass SparseMatrixExt[T, S]:  
      double norm()
      int biCGSTAB(T*, int, T*, int, double) 
      int cols() 
      int nonZeros()
      int rows()
      int size() 
      SparseMatrixExt() 
      SparseMatrixExt(int, int)
      SparseMatrixExt(SparseMatrixExt[T, S]) 
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
      vector[int] getIndsCol(int)
      vector[int] getIndsRow(int)
      void dot1d(double*, double*) nogil 
      void dot2d(double*, int, double*) nogil 
      void dotSub1d(double*, int,  int, double*) nogil 
      void dotSub2d(double*, int,  int, int, double*) nogil 
      void fill(T)
      void insertVal(int, int, T) 
      void makeCompressed()
      void nonZeroInds(int*, int*)
      void nonZeroVals(T*)
      void printValues()
      void prune(double)
      void putSorted2(long*, long*, T, int, long*) 
      void putSorted(long*, long*, T*, int, long*)
      void putUsingTriplets2(int*, int*, T, int)
      void putUsingTriplets(int*, int*, T*, int) 
      void reserve(int)
      void scalarMultiply(double)
      void setZero()
      void slice(int*, int, int*, int, SparseMatrixExt[T, S]*) 
      void unsafeInsertVal2(int, int, T)
      void unsafeInsertVal(int, int, T)
      
cdef template[DataType, StorageType] class csarray:
    cdef SparseMatrixExt[DataType, StorageType] *thisPtr  


