# cython: profile=False
from cython.operator cimport dereference as deref, preincrement as inc 
import numpy 
cimport numpy
import cython 
numpy.import_array()

cdef extern from "include/SparseVectorExt.h":  
   cdef cppclass SparseVectorExt[T]:  
      #SparseVectorExt() 
      #SparseVectorExt(SparseVectorExt[T]) 
      SparseVectorExt(int)
#      double norm()
#      int cols() 
      int nonZeros()
      int rows()
      int size() 
#      SparseVectorExt[T] abs()
#      SparseVectorExt[T] add(SparseVectorExt[T]&)
#      SparseVectorExt[T] dot(SparseVectorExt[T]&)
#      SparseVectorExt[T] hadamard(SparseVectorExt[T]&)
#      SparseVectorExt[T] negate()
#      SparseVectorExt[T] subtract(SparseVectorExt[T]&)
#      SparseVectorExt[T] trans()
      T coeff(int)
#      T sum()
#      T sumValues()
      void insertVal(int, T) 
#      void fill(T)
#      void makeCompressed()
      void nonZeroInds(long*)
#      void printValues()
      void reserve(int)
#      void scalarMultiply(double)
      void slice(int*, int, SparseVectorExt[T]*) 
      
      
cdef template[DataType] class csarray1d:
    cdef SparseVectorExt[DataType] *thisPtr     
    def __cinit__(self, shape):
        """
        Create a new column major dynamic array.
        """
        if type(shape) == tuple: 
            self.thisPtr = new SparseVectorExt[DataType](shape[0]) 
        else: 
            self.thisPtr = new SparseVectorExt[DataType](shape) 
    def __dealloc__(self): 
        """
        Deallocate the SparseVectorExt object.  
        """
        del self.thisPtr
        
    def __getNDim(self): 
        """
        Return the number of dimensions of this array. 
        """
        return 1 
        
    def __getShape(self):
        """
        Return the shape of this array
        """
        return (self.thisPtr.size(), )
        
    def __getSize(self): 
        """
        Return the size of this array, that is the number of elements. 
        """
        return self.thisPtr.size()   
        
    def getnnz(self): 
        """
        Return the number of non-zero elements in the array 
        """
        return self.thisPtr.nonZeros()

    def __setitem__(self, ind, val):
        """
        Set elements of the array. If i is integers then the corresponding 
        value in the array is set. 
        """

        if type(ind) == numpy.ndarray : 
            self.put(val, ind)
        else:
            ind = int(ind) 
            if ind < 0 or ind>=self.thisPtr.rows(): 
                raise ValueError("Invalid row index " + str(ind))       
            
            self.thisPtr.insertVal(ind, val) 

    def put(self, val, numpy.ndarray[numpy.int_t, ndim=1] inds not None): 
        """
        Insert a value or array of values into the array. 
        """
        cdef unsigned int ix 
        self.reserve(len(inds))
        
        if type(val) == numpy.ndarray: 
            for ix in range(len(inds)): 
                self.thisPtr.insertVal(inds[ix], val[ix])
        else:
            for ix in range(len(inds)): 
                self.thisPtr.insertVal(inds[ix], val)
                
    def reserve(self, int n): 
        """
        Reserve n nonzero entries  
        """
        self.thisPtr.reserve(n)
        
    def toarray(self): 
        """
        Convert this sparse matrix into a numpy array. 
        """
        cdef numpy.ndarray[double, ndim=1, mode="c"] result = numpy.zeros(self.shape, numpy.float)
        cdef numpy.ndarray[long, ndim=1, mode="c"] inds
        cdef unsigned int i
        
        (inds, ) = self.nonzero()
            
        for i in range(inds.shape[0]): 
            result[inds[i]] += self.thisPtr.coeff(inds[i])   
            
        return result 

    def nonzero(self): 
        """
        Return a tuple of arrays corresponding to nonzero elements. 
        """
        cdef numpy.ndarray[long, ndim=1, mode="c"] inds = numpy.zeros(self.getnnz(), dtype=numpy.int64) 
        
        if self.getnnz() != 0:
            self.thisPtr.nonZeroInds(&inds[0])
        
        return (inds, )

    def __getitem__(self, ind):
        """
        Get a value or set of values from the array (denoted A). Currently 3 types of parameters 
        are supported. If i is an integer then the corresponding element of the array 
        is returned. If i is an arrays of ints then we return the corresponding 
        values of A[i[k]] (note: i,j must be sorted in ascending order). If i
         is a slice e.g. a[1:5] then we return the submatrix corresponding to 
        the slice. 
        """        
        
        if (type(ind) == numpy.ndarray or type(ind) == slice):
            indList = []            
            if type(ind) == numpy.ndarray: 
                indList.append(ind) 
            elif type(ind) == slice: 
                if ind.start == None: 
                    start = 0
                else: 
                    start = ind.start
                if ind.stop == None: 
                    stop = self.shape[0]
                else:
                    stop = ind.stop  
                indArr = numpy.arange(start, stop)
                indList.append(indArr)
            
            return self.subArray(indList[0])
        else:
            #Deal with negative indices
            if ind < 0: 
                ind += self.thisPtr.rows()

            if ind < 0 or ind>=self.thisPtr.rows(): 
                raise ValueError("Invalid row index " + str(ind))       
            return self.thisPtr.coeff(ind)     

    def subArray(self, numpy.ndarray[numpy.int_t, ndim=1, mode="c"] inds): 
        """
        Explicitly perform an array slice to return a submatrix with the given
        indices. Only works with ascending ordered indices. This is similar 
        to using numpy.ix_. 
        """
        cdef numpy.ndarray[int, ndim=1, mode="c"] indsC 
        cdef csarray1d[DataType] result = csarray1d[DataType](inds.shape[0])     
        
        indsC = numpy.ascontiguousarray(inds, dtype=numpy.int32) 
        
        if inds.shape[0] != 0: 
            self.thisPtr.slice(&indsC[0], indsC.shape[0], result.thisPtr) 
        return result 
    
    shape = property(__getShape)
    size = property(__getSize)
    ndim = property(__getNDim)        
        
       
#    
#    def __adArraySlice(self, numpy.ndarray[numpy.int_t, ndim=1, mode="c"] rowInds, numpy.ndarray[numpy.int_t, ndim=1, mode="c"] colInds): 
#        """
#        Array slicing where one passes two arrays of the same length and elements are picked 
#        according to self[rowInds[i], colInds[i]). 
#        """
#        cdef int ix 
#        cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] result = numpy.zeros(rowInds.shape[0])
#        
#        if (rowInds >= self.shape[0]).any() or (colInds >= self.shape[1]).any(): 
#            raise ValueError("Indices out of range")
#        
#        for ix in range(rowInds.shape[0]): 
#                result[ix] = self.thisPtr.coeff(rowInds[ix], colInds[ix])
#        return result
#    

#        

#                    

#
    

            
#
#    def sum(self, axis=None): 
#        """
#        Sum all of the elements in this array. If one specifies an axis 
#        then we sum along the axis. 
#        """
#        cdef numpy.ndarray[double, ndim=1, mode="c"] result    
#        cdef numpy.ndarray[long, ndim=1, mode="c"] rowInds
#        cdef numpy.ndarray[long, ndim=1, mode="c"] colInds
#        cdef unsigned int i
#        
#        if axis==None: 
#            """
#            scalarResult = 0 
#            (rowInds, colInds) = self.nonzero()
#            
#            for i in range(rowInds.shape[0]): 
#                scalarResult += self.thisPtr.coeff(rowInds[i], colInds[i])  
#            
#            return scalarResult
#            """
#            return self.thisPtr.sumValues()
#            #There seems to be a very temporamental problem with thisPtr.sum()
#            #return self.thisPtr.sum()
#        elif axis==0: 
#            result = numpy.zeros(self.shape[1], dtype=numpy.float) 
#            (rowInds, colInds) = self.nonzero()
#            
#            for i in range(rowInds.shape[0]): 
#                result[colInds[i]] += self.thisPtr.coeff(rowInds[i], colInds[i])   
#        elif axis==1: 
#            result = numpy.zeros(self.shape[0], dtype=numpy.float) 
#            (rowInds, colInds) = self.nonzero()
#            
#            for i in range(rowInds.shape[0]): 
#                result[rowInds[i]] += self.thisPtr.coeff(rowInds[i], colInds[i])  
#        else:
#            raise ValueError("Invalid axis: " + str(axis))
#            
#        return result 
#                
#        
#    def mean(self, axis=None): 
#        """
#        Find the mean value of this array. 
#        """
#        if self.thisPtr.size() != 0:
#            if axis ==None: 
#                return self.sum()/float(self.thisPtr.size())
#            elif axis == 0: 
#                return self.sum(0)/float(self.shape[0])
#            elif axis == 1: 
#                return self.sum(1)/float(self.shape[1])
#        else: 
#            return float("nan")
#     
#    def diag(self): 
#        """
#        Return a numpy array containing the diagonal entries of this matrix. If 
#        the matrix is non-square then the diagonal array is the same size as the 
#        smallest dimension. 
#        """
#        cdef unsigned int maxInd = min(self.shape[0], self.shape[1])
#        cdef unsigned int i   
#        cdef numpy.ndarray[numpy.float_t, ndim=1, mode="c"] result = numpy.zeros(maxInd)
#        
#        for i in range(maxInd): 
#            result[i] = self.thisPtr.coeff(i, i)
#            
#        return result
#        
#    def trace(self): 
#        """
#        Returns the trace of the array which is simply the sum of the diagonal 
#        entries. 
#        """
#        return self.diag().sum()
#         
#    def __mul__(self, double x):
#        """
#        Return a new array multiplied by a scalar value x. 
#        """
#        cdef csarray[DataType] result = self.copy() 
#        result.thisPtr.scalarMultiply(x)
#        return result 
#        
#    def copy(self): 
#        """
#        Return a copied version of this array. 
#        """
#        cdef csarray[DataType] result = csarray[DataType](self.shape)
#        del result.thisPtr
#        result.thisPtr = new SparseVectorExt[DataType](deref(self.thisPtr))
#        return result 
#        

#        
#        
#    def min(self): 
#        """
#        Find the minimum element of this array. 
#        """
#        cdef numpy.ndarray[long, ndim=1, mode="c"] rowInds
#        cdef numpy.ndarray[long, ndim=1, mode="c"] colInds
#        cdef unsigned int i
#        cdef DataType minVal 
#        
#        if self.size == 0: 
#            return float("nan")
#        elif self.getnnz() != self.size: 
#            minVal = 0 
#        
#        (rowInds, colInds) = self.nonzero()
#            
#        for i in range(rowInds.shape[0]): 
#            if i == 0: 
#                minVal = self.thisPtr.coeff(rowInds[i], colInds[i])
#            
#            if self.thisPtr.coeff(rowInds[i], colInds[i]) < minVal: 
#                minVal = self.thisPtr.coeff(rowInds[i], colInds[i])
#            
#        return minVal 
#        
#    def max(self): 
#        """
#        Find the maximum element of this array. 
#        """
#        cdef numpy.ndarray[long, ndim=1, mode="c"] rowInds
#        cdef numpy.ndarray[long, ndim=1, mode="c"] colInds
#        cdef unsigned int i
#        cdef DataType maxVal
#        
#        if self.size == 0: 
#            return float("nan")
#        elif self.getnnz() != self.size: 
#            maxVal = 0 
#        
#        (rowInds, colInds) = self.nonzero()
#            
#        for i in range(rowInds.shape[0]): 
#            if i == 0: 
#                maxVal = self.thisPtr.coeff(rowInds[i], colInds[i])            
#            
#            if self.thisPtr.coeff(rowInds[i], colInds[i]) > maxVal: 
#                maxVal = self.thisPtr.coeff(rowInds[i], colInds[i])
#            
#        return maxVal 
#        
#    def var(self): 
#        """
#        Return the variance of the elements of this array. 
#        """
#        cdef double mean = self.mean() 
#        cdef numpy.ndarray[long, ndim=1, mode="c"] rowInds
#        cdef numpy.ndarray[long, ndim=1, mode="c"] colInds
#        cdef unsigned int i
#        cdef double result = 0
#        
#        if self.size == 0: 
#            return float("nan")
#        
#        (rowInds, colInds) = self.nonzero()
#            
#        for i in range(rowInds.shape[0]): 
#            result += (self.thisPtr.coeff(rowInds[i], colInds[i]) - mean)**2
#        
#        result += (self.size - self.getnnz())*mean**2
#        result /= float(self.size)
#        
#        return result 
#    
#    def std(self): 
#        """
#        Return the standard deviation of the array elements. 
#        """
#        return numpy.sqrt(self.var())
#        
#    def __abs__(self): 
#        """
#        Return a matrix whose elements are the absolute values of this array. 
#        """
#        cdef csarray[DataType] result = csarray[DataType]((self.shape[1], self.shape[0]))
#        del result.thisPtr
#        result.thisPtr = new SparseVectorExt[DataType](self.thisPtr.abs())
#        return result 
#    
#    def __neg__(self): 
#        """
#        Return the negation of this array. 
#        """
#        cdef csarray[DataType] result = csarray[DataType]((self.shape[1], self.shape[0]))
#        del result.thisPtr
#        result.thisPtr = new SparseVectorExt[DataType](self.thisPtr.negate())
#        return result 
#    
#
#    def __add__(csarray[DataType] self, csarray[DataType] A): 
#        """
#        Add two matrices together. 
#        """
#        if self.shape != A.shape: 
#            raise ValueError("Cannot add matrices of shapes " + str(self.shape) + " and " + str(A.shape))
#        
#        cdef csarray[DataType] result = csarray[DataType]((self.shape[0], A.shape[1]))
#        del result.thisPtr
#        result.thisPtr = new SparseVectorExt[DataType](self.thisPtr.add(deref(A.thisPtr)))
#        return result     
#        
#    def __sub__(csarray[DataType] self, csarray[DataType] A): 
#        """
#        Subtract one matrix from another.  
#        """
#        if self.shape != A.shape: 
#            raise ValueError("Cannot subtract matrices of shapes " + str(self.shape) + " and " + str(A.shape))
#        
#        cdef csarray[DataType] result = csarray[DataType]((self.shape[0], A.shape[1]))
#        del result.thisPtr
#        result.thisPtr = new SparseVectorExt[DataType](self.thisPtr.subtract(deref(A.thisPtr)))
#        return result    
#     
#    def hadamard(self, csarray[DataType] A): 
#        """
#        Find the element-wise matrix (hadamard) product. 
#        """
#        if self.shape != A.shape: 
#            raise ValueError("Cannot elementwise multiply matrices of shapes " + str(self.shape) + " and " + str(A.shape))
#        
#        cdef csarray[DataType] result = csarray[DataType]((self.shape[0], A.shape[1]))
#        del result.thisPtr
#        result.thisPtr = new SparseVectorExt[DataType](self.thisPtr.hadamard(deref(A.thisPtr)))
#        return result 
#
#    def compress(self): 
#        """
#        Turn this matrix into compressed sparse format by freeing extra memory 
#        space in the buffer. 
#        """
#        self.thisPtr.makeCompressed()
#        
#    

#        
#    def dot(self, csarray[DataType] A): 
#        if self.shape[1] != A.shape[0]: 
#            raise ValueError("Cannot multiply matrices of shapes " + str(self.shape) + " and " + str(A.shape))
#        
#        cdef csarray[DataType] result = csarray[DataType]((self.shape[0], A.shape[1]))
#        del result.thisPtr
#        result.thisPtr = new SparseVectorExt[DataType](self.thisPtr.dot(deref(A.thisPtr)))
#        return result 
#        
#    def transpose(self): 
#        """
#        Find the transpose of this matrix. 
#        """
#        cdef csarray[DataType] result = csarray[DataType]((self.shape[1], self.shape[0]))
#        del result.thisPtr
#        result.thisPtr = new SparseVectorExt[DataType](self.thisPtr.trans())
#        return result 
#   
#    #def norm(self, ord="fro"): 
#        """
#        Return the norm of this array. Currently only the Frobenius norm is 
#        supported. 
#        """
#        #return self.thisPtr.norm()
#   
#    def ones(self): 
#        """
#        Fill the array with ones. 
#        """
#        self.thisPtr.fill(1)
#        
   

    

    
