

cdef extern from "include/SparseVectorExt.h":  
   cdef cppclass SparseVectorExt[T]:  
      SparseVectorExt() 
      SparseVectorExt(SparseVectorExt[T]) 
      SparseVectorExt(int)
#      double norm()
      int nonZeros()
      int rows()
      int size() 
      SparseVectorExt[T] abs()
      SparseVectorExt[T] add(SparseVectorExt[T]&)
      T dot(SparseVectorExt[T]&)
      SparseVectorExt[T] hadamard(SparseVectorExt[T]&)
      SparseVectorExt[T] negate()
      SparseVectorExt[T] subtract(SparseVectorExt[T]&)
      T coeff(int)
#      T sum()
      T sumValues()
      void insertVal(int, T) 
      void fill(T)
      void nonZeroInds(long*)
      void reserve(int)
      void scalarMultiply(double)
      void slice(int*, int, SparseVectorExt[T]*) 
      
cdef template[DataType] class csarray1d:
    cdef SparseVectorExt[DataType] *thisPtr     