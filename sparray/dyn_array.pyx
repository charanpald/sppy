# cython: profile=True
from cython.operator cimport dereference as deref, preincrement as inc 
import numpy 
cimport numpy 
numpy.import_array()

cdef extern from "include/DynamicSparseMatrixExt.h": 
   cdef cppclass DynamicSparseMatrixExt[T]:  
      DynamicSparseMatrixExt()
      DynamicSparseMatrixExt(DynamicSparseMatrixExt[T]) 
      DynamicSparseMatrixExt(int, int)
      int rows()
      int cols() 
      int size() 
      void insertVal(int, int, T)
      int nonZeros()
      void nonZeroInds(int*, int*)
      T coeff(int, int)
      T sum()
      void slice(int*, int, int*, int, DynamicSparseMatrixExt[T]*) 
      void scalarMultiply(double)

cdef class dyn_array:
    cdef DynamicSparseMatrixExt[double] *thisPtr     
    def __cinit__(self, shape, dtype=numpy.float):
        """
        Create a new column major dynamic array. One can pass in a numpy 
        data type but the only option is numpy.float currently. 
        """
        if dtype==numpy.float: 
            self.thisPtr = new DynamicSparseMatrixExt[double](shape[0], shape[1]) 
        else: 
            raise ValueError("Unsupported dtype: " + str(dtype))
            
    def __dealloc__(self): 
        """
        Deallocate the DynamicSparseMatrixExt object.  
        """
        del self.thisPtr
        
    def __getNDim(self): 
        """
        Return the number of dimensions of this array. 
        """
        return 2 
        
    def __getShape(self):
        """
        Return the shape of this array (rows, cols)
        """
        return (self.thisPtr.rows(), self.thisPtr.cols())
        
    def __getSize(self): 
        """
        Return the size of this array, that is rows*cols 
        """
        return self.thisPtr.size()   
        
    def getnnz(self): 
        """
        Return the number of non-zero elements in the array 
        """
        return self.thisPtr.nonZeros()
        
    def __getDType(self): 
        """
        Get the dtype of this array. 
        """
        return numpy.float
        
    def __getitem__(self, inds):
        """
        Get a value or set of values from the array (denoted A). Currently 3 types of parameters 
        are supported. If i,j = inds are integers then the corresponding elements of the array 
        are returned. If i,j are both arrays of ints then we return the corresponding 
        values of A[i[k], j[k]] (note: i,j must be sorted in ascending order). If one of 
        i or j is a slice e.g. A[[1,2], :] then we return the submatrix corresponding to 
        the slice. 
        """        
        
        i, j = inds 
        
        if type(i) == numpy.ndarray and type(j) == numpy.ndarray: 
            return self.__adArraySlice(numpy.ascontiguousarray(i, dtype=numpy.int) , numpy.ascontiguousarray(j, dtype=numpy.int) )
        elif (type(i) == numpy.ndarray or type(i) == slice) and (type(j) == slice or type(j) == numpy.ndarray):
            indList = []            
            for k in range(len(inds)):  
                index = inds[k] 
                if type(index) == numpy.ndarray: 
                    indList.append(index) 
                elif type(index) == slice: 
                    if index.start == None: 
                        start = 0
                    else: 
                        start = index.start
                    if index.stop == None: 
                        stop = self.shape[k]
                    else:
                        stop = index.stop  
                    indArr = numpy.arange(start, stop)
                    indList.append(indArr)
            
            return self.subArray(indList[0], indList[1])
        else:
            i = int(i) 
            j = int(j)
            
            #Deal with negative indices
            if i<0: 
                i += self.thisPtr.rows()
            if j<0:
                j += self.thisPtr.cols()    

            if i < 0 or i>=self.thisPtr.rows(): 
                raise ValueError("Invalid row index " + str(i)) 
            if j < 0 or j>=self.thisPtr.cols(): 
                raise ValueError("Invalid col index " + str(j))      
            return self.thisPtr.coeff(i, j)            
    
    def __adArraySlice(self, numpy.ndarray[numpy.int_t, ndim=1, mode="c"] rowInds, numpy.ndarray[numpy.int_t, ndim=1, mode="c"] colInds): 
        """
        Array slicing where one passes two arrays of the same length and elements are picked 
        according to self[rowInds[i], colInds[i]). 
        """
        cdef int ix 
        cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] result = numpy.zeros(rowInds.shape[0])
        
        if (rowInds >= self.shape[0]).any() or (colInds >= self.shape[1]).any(): 
            raise ValueError("Indices out of range")
        
        for ix in range(rowInds.shape[0]): 
                result[ix] = self.thisPtr.coeff(rowInds[ix], colInds[ix])
        return result
    
    def subArray(self, numpy.ndarray[numpy.int_t, ndim=1, mode="c"] rowInds, numpy.ndarray[numpy.int_t, ndim=1, mode="c"] colInds): 
        """
        Explicitly perform an array slice to return a submatrix with the given
        indices. Only works with ascending ordered indices. This is similar 
        to using numpy.ix_. 
        """
        cdef numpy.ndarray[int, ndim=1, mode="c"] rowIndsC 
        cdef numpy.ndarray[int, ndim=1, mode="c"] colIndsC 
        
        result = dyn_array((rowInds.shape[0], colInds.shape[0]))     
        
        rowIndsC = numpy.ascontiguousarray(rowInds, dtype=numpy.int32) 
        colIndsC = numpy.ascontiguousarray(colInds, dtype=numpy.int32) 
        
        if rowInds.shape[0] != 0 and colInds.shape[0] != 0: 
            self.thisPtr.slice(&rowIndsC[0], rowIndsC.shape[0], &colIndsC[0], colIndsC.shape[0], result.thisPtr) 
        return result 
        
    def nonzero(self): 
        """
        Return a tuple of arrays corresponding to nonzero elements. 
        """
        cdef numpy.ndarray[int, ndim=1, mode="c"] rowInds = numpy.zeros(self.getnnz(), dtype=numpy.int32) 
        cdef numpy.ndarray[int, ndim=1, mode="c"] colInds = numpy.zeros(self.getnnz(), dtype=numpy.int32)  
        
        if self.getnnz() != 0:
            self.thisPtr.nonZeroInds(&rowInds[0], &colInds[0])
        
        return (rowInds, colInds)
                    
    def __setitem__(self, inds, val):
        """
        Set elements of the array. If i,j = inds are integers then the corresponding 
        value in the array is set. 
        """
        i, j = inds 
        if type(i) == int and type(j) == int: 
            if i < 0 or i>=self.thisPtr.rows(): 
                raise ValueError("Invalid row index " + str(i)) 
            if j < 0 or j>=self.thisPtr.cols(): 
                raise ValueError("Invalid col index " + str(j))      
            
            self.thisPtr.insertVal(i, j, val)
        elif type(i) == numpy.ndarray and type(j) == numpy.ndarray: 
            for ix in range(len(i)): 
                self.thisPtr.insertVal(i[ix], j[ix], val)  
    
    def put(self, double val, numpy.ndarray[numpy.int_t, ndim=1] rowInds not None , numpy.ndarray[numpy.int_t, ndim=1] colInds not None): 
        """
        Select rowInds 
        """
        cdef unsigned int ix 
        for ix in range(len(rowInds)): 
            self.thisPtr.insertVal(rowInds[ix], colInds[ix], val)

    def sum(self): 
        """
        Sum all of the elements in this array. 
        """
        return self.thisPtr.sum()
        
    def mean(self): 
        """
        Find the mean value of this array. 
        """
        if self.thisPtr.size() != 0: 
            return self.thisPtr.sum()/self.thisPtr.size() 
        else: 
            return float("nan")
     
    def __str__(self): 
        """
        Return a string representation of the non-zero elements of the array. 
        """
        outputStr = "dyn_array shape:" + str(self.shape) + " non-zeros:" + str(self.getnnz()) + "\n"
        (rowInds, colInds) = self.nonzero()
        vals = self[rowInds, colInds]
        
        for i in range(self.getnnz()): 
            outputStr += "(" + str(rowInds[i]) + ", " + str(colInds[i]) + ")" + " " + str(vals[i]) + "\n"
        
        return outputStr 
        
    def diag(self): 
        """
        Return a numpy array containing the diagonal entries of this matrix. If 
        the matrix is non-square then the diagonal array is the same size as the 
        smallest dimension. 
        """
        cdef unsigned int maxInd = min(self.shape[0], self.shape[1])
        cdef unsigned int i   
        cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] result = numpy.zeros(maxInd)
        
        for i in range(maxInd): 
            result[i] = self.thisPtr.coeff(i, i)
            
        return result
         
    def __mul__(self, double x):
        """
        Return a new array multiplied by a scalar value x. 
        """
        cdef dyn_array result = self.copy() 
        result.thisPtr.scalarMultiply(x)
        return result 
        
    def copy(self): 
        """
        Return a copied version of this array. 
        """
        cdef dyn_array result = dyn_array(self.shape)
        del result.thisPtr
        result.thisPtr = new DynamicSparseMatrixExt[double](deref(self.thisPtr))
        return result 
        
    
    shape = property(__getShape)
    size = property(__getSize)
    ndim = property(__getNDim)
    dtype = property(__getDType)

    
