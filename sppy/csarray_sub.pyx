# cython: profile=False
from cython.operator cimport dereference as deref, preincrement as inc 
import numpy 
cimport numpy
import cython 
numpy.import_array()

cdef extern from "include/SparseMatrixExt.h":  
   cdef cppclass SparseMatrixExt[T]:  
      SparseMatrixExt() 
      SparseMatrixExt(SparseMatrixExt[T]) 
      SparseMatrixExt(int, int)
      double norm()
      int cols() 
      int nonZeros()
      int rows()
      int size() 
      SparseMatrixExt[T] abs()
      SparseMatrixExt[T] add(SparseMatrixExt[T]&)
      SparseMatrixExt[T] dot(SparseMatrixExt[T]&)
      SparseMatrixExt[T] hadamard(SparseMatrixExt[T]&)
      SparseMatrixExt[T] negate()
      SparseMatrixExt[T] subtract(SparseMatrixExt[T]&)
      SparseMatrixExt[T] trans()
      T coeff(int, int)
      T sum()
      T sumValues()
      void insertVal(int, int, T) 
      void fill(T)
      void makeCompressed()
      void nonZeroInds(long*, long*)
      void printValues()
      void reserve(int)
      void scalarMultiply(double)
      void slice(int*, int, int*, int, SparseMatrixExt[T]*) 
      
      
cdef class csarray_signed_char:
    cdef SparseMatrixExt[signed char] *thisPtr     
    def __cinit__(self, shape):
        """
        Create a new column major dynamic array.
        """

        self.thisPtr = new SparseMatrixExt[signed char](shape[0], shape[1]) 
            
    def __dealloc__(self): 
        """
        Deallocate the SparseMatrixExt object.  
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
            for k, index in enumerate(inds):  
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
        
        cdef csarray_signed_char result = csarray_signed_char((rowInds.shape[0], colInds.shape[0]))     
        
        rowIndsC = numpy.ascontiguousarray(rowInds, dtype=numpy.int32) 
        colIndsC = numpy.ascontiguousarray(colInds, dtype=numpy.int32) 
        
        if rowInds.shape[0] != 0 and colInds.shape[0] != 0: 
            self.thisPtr.slice(&rowIndsC[0], rowIndsC.shape[0], &colIndsC[0], colIndsC.shape[0], result.thisPtr) 
        return result 
        
    def nonzero(self): 
        """
        Return a tuple of arrays corresponding to nonzero elements. 
        """
        cdef numpy.ndarray[long, ndim=1, mode="c"] rowInds = numpy.zeros(self.getnnz(), dtype=numpy.int64) 
        cdef numpy.ndarray[long, ndim=1, mode="c"] colInds = numpy.zeros(self.getnnz(), dtype=numpy.int64)  
        
        if self.getnnz() != 0:
            self.thisPtr.nonZeroInds(&rowInds[0], &colInds[0])
        
        return (rowInds, colInds)
                    
    def __setitem__(self, inds, val):
        """
        Set elements of the array. If i,j = inds are integers then the corresponding 
        value in the array is set. 
        """
        i, j = inds 
        
        if type(i) == numpy.ndarray and type(j) == numpy.ndarray: 
            self.put(val, i, j)
        else:
            i = int(i) 
            j = int(j)
            if i < 0 or i>=self.thisPtr.rows(): 
                raise ValueError("Invalid row index " + str(i)) 
            if j < 0 or j>=self.thisPtr.cols(): 
                raise ValueError("Invalid col index " + str(j))        
            
            self.thisPtr.insertVal(i, j, val) 

    
    def put(self, val, numpy.ndarray[numpy.int_t, ndim=1] rowInds not None, numpy.ndarray[numpy.int_t, ndim=1] colInds not None): 
        """
        Select rowInds and colInds
        """
        cdef unsigned int ix 
        self.reserve(len(rowInds))
        
        if type(val) == numpy.ndarray: 
            for ix in range(len(rowInds)): 
                self.thisPtr.insertVal(rowInds[ix], colInds[ix], val[ix])
        else:
            for ix in range(len(rowInds)): 
                self.thisPtr.insertVal(rowInds[ix], colInds[ix], val)
            

    def sum(self, axis=None): 
        """
        Sum all of the elements in this array. If one specifies an axis 
        then we sum along the axis. 
        """
        cdef numpy.ndarray[double, ndim=1, mode="c"] result    
        cdef numpy.ndarray[long, ndim=1, mode="c"] rowInds
        cdef numpy.ndarray[long, ndim=1, mode="c"] colInds
        cdef unsigned int i
        
        if axis==None: 
            """
            scalarResult = 0 
            (rowInds, colInds) = self.nonzero()
            
            for i in range(rowInds.shape[0]): 
                scalarResult += self.thisPtr.coeff(rowInds[i], colInds[i])  
            
            return scalarResult
            """
            return self.thisPtr.sumValues()
            #There seems to be a very temporamental problem with thisPtr.sum()
            #return self.thisPtr.sum()
        elif axis==0: 
            result = numpy.zeros(self.shape[1], dtype=numpy.float) 
            (rowInds, colInds) = self.nonzero()
            
            for i in range(rowInds.shape[0]): 
                result[colInds[i]] += self.thisPtr.coeff(rowInds[i], colInds[i])   
        elif axis==1: 
            result = numpy.zeros(self.shape[0], dtype=numpy.float) 
            (rowInds, colInds) = self.nonzero()
            
            for i in range(rowInds.shape[0]): 
                result[rowInds[i]] += self.thisPtr.coeff(rowInds[i], colInds[i])  
        else:
            raise ValueError("Invalid axis: " + str(axis))
            
        return result 
                
        
    def mean(self, axis=None): 
        """
        Find the mean value of this array. 
        """
        if self.thisPtr.size() != 0:
            if axis ==None: 
                return self.sum()/float(self.thisPtr.size())
            elif axis == 0: 
                return self.sum(0)/float(self.shape[0])
            elif axis == 1: 
                return self.sum(1)/float(self.shape[1])
        else: 
            return float("nan")
     
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
        
    def trace(self): 
        """
        Returns the trace of the array which is simply the sum of the diagonal 
        entries. 
        """
        return self.diag().sum()
         
    def __mul__(self, double x):
        """
        Return a new array multiplied by a scalar value x. 
        """
        cdef csarray_signed_char result = self.copy() 
        result.thisPtr.scalarMultiply(x)
        return result 
        
    def copy(self): 
        """
        Return a copied version of this array. 
        """
        cdef csarray_signed_char result = csarray_signed_char(self.shape)
        del result.thisPtr
        result.thisPtr = new SparseMatrixExt[signed char](deref(self.thisPtr))
        return result 
        
    def toarray(self): 
        """
        Convert this sparse matrix into a numpy array. 
        """
        cdef numpy.ndarray[double, ndim=2, mode="c"] result = numpy.zeros(self.shape, numpy.float)
        cdef numpy.ndarray[long, ndim=1, mode="c"] rowInds
        cdef numpy.ndarray[long, ndim=1, mode="c"] colInds
        cdef unsigned int i
        
        (rowInds, colInds) = self.nonzero()
            
        for i in range(rowInds.shape[0]): 
            result[rowInds[i], colInds[i]] += self.thisPtr.coeff(rowInds[i], colInds[i])   
            
        return result 
        
        
    def min(self): 
        """
        Find the minimum element of this array. 
        """
        cdef numpy.ndarray[long, ndim=1, mode="c"] rowInds
        cdef numpy.ndarray[long, ndim=1, mode="c"] colInds
        cdef unsigned int i
        cdef signed char minVal 
        
        if self.size == 0: 
            return float("nan")
        elif self.getnnz() != self.size: 
            minVal = 0 
        
        (rowInds, colInds) = self.nonzero()
            
        for i in range(rowInds.shape[0]): 
            if i == 0: 
                minVal = self.thisPtr.coeff(rowInds[i], colInds[i])
            
            if self.thisPtr.coeff(rowInds[i], colInds[i]) < minVal: 
                minVal = self.thisPtr.coeff(rowInds[i], colInds[i])
            
        return minVal 
        
    def max(self): 
        """
        Find the maximum element of this array. 
        """
        cdef numpy.ndarray[long, ndim=1, mode="c"] rowInds
        cdef numpy.ndarray[long, ndim=1, mode="c"] colInds
        cdef unsigned int i
        cdef signed char maxVal
        
        if self.size == 0: 
            return float("nan")
        elif self.getnnz() != self.size: 
            maxVal = 0 
        
        (rowInds, colInds) = self.nonzero()
            
        for i in range(rowInds.shape[0]): 
            if i == 0: 
                maxVal = self.thisPtr.coeff(rowInds[i], colInds[i])            
            
            if self.thisPtr.coeff(rowInds[i], colInds[i]) > maxVal: 
                maxVal = self.thisPtr.coeff(rowInds[i], colInds[i])
            
        return maxVal 
        
    def var(self): 
        """
        Return the variance of the elements of this array. 
        """
        cdef double mean = self.mean() 
        cdef numpy.ndarray[long, ndim=1, mode="c"] rowInds
        cdef numpy.ndarray[long, ndim=1, mode="c"] colInds
        cdef unsigned int i
        cdef double result = 0
        
        if self.size == 0: 
            return float("nan")
        
        (rowInds, colInds) = self.nonzero()
            
        for i in range(rowInds.shape[0]): 
            result += (self.thisPtr.coeff(rowInds[i], colInds[i]) - mean)**2
        
        result += (self.size - self.getnnz())*mean**2
        result /= float(self.size)
        
        return result 
    
    def std(self): 
        """
        Return the standard deviation of the array elements. 
        """
        return numpy.sqrt(self.var())
        
    def __abs__(self): 
        """
        Return a matrix whose elements are the absolute values of this array. 
        """
        cdef csarray_signed_char result = csarray_signed_char((self.shape[1], self.shape[0]))
        del result.thisPtr
        result.thisPtr = new SparseMatrixExt[signed char](self.thisPtr.abs())
        return result 
    
    def __neg__(self): 
        """
        Return the negation of this array. 
        """
        cdef csarray_signed_char result = csarray_signed_char((self.shape[1], self.shape[0]))
        del result.thisPtr
        result.thisPtr = new SparseMatrixExt[signed char](self.thisPtr.negate())
        return result 
    

    def __add__(csarray_signed_char self, csarray_signed_char A): 
        """
        Add two matrices together. 
        """
        if self.shape != A.shape: 
            raise ValueError("Cannot add matrices of shapes " + str(self.shape) + " and " + str(A.shape))
        
        cdef csarray_signed_char result = csarray_signed_char((self.shape[0], A.shape[1]))
        del result.thisPtr
        result.thisPtr = new SparseMatrixExt[signed char](self.thisPtr.add(deref(A.thisPtr)))
        return result     
        
    def __sub__(csarray_signed_char self, csarray_signed_char A): 
        """
        Subtract one matrix from another.  
        """
        if self.shape != A.shape: 
            raise ValueError("Cannot subtract matrices of shapes " + str(self.shape) + " and " + str(A.shape))
        
        cdef csarray_signed_char result = csarray_signed_char((self.shape[0], A.shape[1]))
        del result.thisPtr
        result.thisPtr = new SparseMatrixExt[signed char](self.thisPtr.subtract(deref(A.thisPtr)))
        return result    
     
    def hadamard(self, csarray_signed_char A): 
        """
        Find the element-wise matrix (hadamard) product. 
        """
        if self.shape != A.shape: 
            raise ValueError("Cannot elementwise multiply matrices of shapes " + str(self.shape) + " and " + str(A.shape))
        
        cdef csarray_signed_char result = csarray_signed_char((self.shape[0], A.shape[1]))
        del result.thisPtr
        result.thisPtr = new SparseMatrixExt[signed char](self.thisPtr.hadamard(deref(A.thisPtr)))
        return result 

    def compress(self): 
        """
        Turn this matrix into compressed sparse format by freeing extra memory 
        space in the buffer. 
        """
        self.thisPtr.makeCompressed()
        
    
    def reserve(self, int n): 
        """
        Reserve n nonzero entries and turns the matrix into uncompressed mode. 
        """
        self.thisPtr.reserve(n)
        
    def dot(self, csarray_signed_char A): 
        if self.shape[1] != A.shape[0]: 
            raise ValueError("Cannot multiply matrices of shapes " + str(self.shape) + " and " + str(A.shape))
        
        cdef csarray_signed_char result = csarray_signed_char((self.shape[0], A.shape[1]))
        del result.thisPtr
        result.thisPtr = new SparseMatrixExt[signed char](self.thisPtr.dot(deref(A.thisPtr)))
        return result 
        
    def transpose(self): 
        """
        Find the transpose of this matrix. 
        """
        cdef csarray_signed_char result = csarray_signed_char((self.shape[1], self.shape[0]))
        del result.thisPtr
        result.thisPtr = new SparseMatrixExt[signed char](self.thisPtr.trans())
        return result 
   
    #def norm(self, ord="fro"): 
        """
        Return the norm of this array. Currently only the Frobenius norm is 
        supported. 
        """
        #return self.thisPtr.norm()
   
    def ones(self): 
        """
        Fill the array with ones. 
        """
        self.thisPtr.fill(1)
        
   
    shape = property(__getShape)
    size = property(__getSize)
    ndim = property(__getNDim)
    dtype = property(__getDType)
    

    
cdef class csarray_short:
    cdef SparseMatrixExt[short] *thisPtr     
    def __cinit__(self, shape):
        """
        Create a new column major dynamic array.
        """

        self.thisPtr = new SparseMatrixExt[short](shape[0], shape[1]) 
            
    def __dealloc__(self): 
        """
        Deallocate the SparseMatrixExt object.  
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
            for k, index in enumerate(inds):  
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
        
        cdef csarray_short result = csarray_short((rowInds.shape[0], colInds.shape[0]))     
        
        rowIndsC = numpy.ascontiguousarray(rowInds, dtype=numpy.int32) 
        colIndsC = numpy.ascontiguousarray(colInds, dtype=numpy.int32) 
        
        if rowInds.shape[0] != 0 and colInds.shape[0] != 0: 
            self.thisPtr.slice(&rowIndsC[0], rowIndsC.shape[0], &colIndsC[0], colIndsC.shape[0], result.thisPtr) 
        return result 
        
    def nonzero(self): 
        """
        Return a tuple of arrays corresponding to nonzero elements. 
        """
        cdef numpy.ndarray[long, ndim=1, mode="c"] rowInds = numpy.zeros(self.getnnz(), dtype=numpy.int64) 
        cdef numpy.ndarray[long, ndim=1, mode="c"] colInds = numpy.zeros(self.getnnz(), dtype=numpy.int64)  
        
        if self.getnnz() != 0:
            self.thisPtr.nonZeroInds(&rowInds[0], &colInds[0])
        
        return (rowInds, colInds)
                    
    def __setitem__(self, inds, val):
        """
        Set elements of the array. If i,j = inds are integers then the corresponding 
        value in the array is set. 
        """
        i, j = inds 
        
        if type(i) == numpy.ndarray and type(j) == numpy.ndarray: 
            self.put(val, i, j)
        else:
            i = int(i) 
            j = int(j)
            if i < 0 or i>=self.thisPtr.rows(): 
                raise ValueError("Invalid row index " + str(i)) 
            if j < 0 or j>=self.thisPtr.cols(): 
                raise ValueError("Invalid col index " + str(j))        
            
            self.thisPtr.insertVal(i, j, val) 

    
    def put(self, val, numpy.ndarray[numpy.int_t, ndim=1] rowInds not None, numpy.ndarray[numpy.int_t, ndim=1] colInds not None): 
        """
        Select rowInds and colInds
        """
        cdef unsigned int ix 
        self.reserve(len(rowInds))
        
        if type(val) == numpy.ndarray: 
            for ix in range(len(rowInds)): 
                self.thisPtr.insertVal(rowInds[ix], colInds[ix], val[ix])
        else:
            for ix in range(len(rowInds)): 
                self.thisPtr.insertVal(rowInds[ix], colInds[ix], val)
            

    def sum(self, axis=None): 
        """
        Sum all of the elements in this array. If one specifies an axis 
        then we sum along the axis. 
        """
        cdef numpy.ndarray[double, ndim=1, mode="c"] result    
        cdef numpy.ndarray[long, ndim=1, mode="c"] rowInds
        cdef numpy.ndarray[long, ndim=1, mode="c"] colInds
        cdef unsigned int i
        
        if axis==None: 
            """
            scalarResult = 0 
            (rowInds, colInds) = self.nonzero()
            
            for i in range(rowInds.shape[0]): 
                scalarResult += self.thisPtr.coeff(rowInds[i], colInds[i])  
            
            return scalarResult
            """
            return self.thisPtr.sumValues()
            #There seems to be a very temporamental problem with thisPtr.sum()
            #return self.thisPtr.sum()
        elif axis==0: 
            result = numpy.zeros(self.shape[1], dtype=numpy.float) 
            (rowInds, colInds) = self.nonzero()
            
            for i in range(rowInds.shape[0]): 
                result[colInds[i]] += self.thisPtr.coeff(rowInds[i], colInds[i])   
        elif axis==1: 
            result = numpy.zeros(self.shape[0], dtype=numpy.float) 
            (rowInds, colInds) = self.nonzero()
            
            for i in range(rowInds.shape[0]): 
                result[rowInds[i]] += self.thisPtr.coeff(rowInds[i], colInds[i])  
        else:
            raise ValueError("Invalid axis: " + str(axis))
            
        return result 
                
        
    def mean(self, axis=None): 
        """
        Find the mean value of this array. 
        """
        if self.thisPtr.size() != 0:
            if axis ==None: 
                return self.sum()/float(self.thisPtr.size())
            elif axis == 0: 
                return self.sum(0)/float(self.shape[0])
            elif axis == 1: 
                return self.sum(1)/float(self.shape[1])
        else: 
            return float("nan")
     
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
        
    def trace(self): 
        """
        Returns the trace of the array which is simply the sum of the diagonal 
        entries. 
        """
        return self.diag().sum()
         
    def __mul__(self, double x):
        """
        Return a new array multiplied by a scalar value x. 
        """
        cdef csarray_short result = self.copy() 
        result.thisPtr.scalarMultiply(x)
        return result 
        
    def copy(self): 
        """
        Return a copied version of this array. 
        """
        cdef csarray_short result = csarray_short(self.shape)
        del result.thisPtr
        result.thisPtr = new SparseMatrixExt[short](deref(self.thisPtr))
        return result 
        
    def toarray(self): 
        """
        Convert this sparse matrix into a numpy array. 
        """
        cdef numpy.ndarray[double, ndim=2, mode="c"] result = numpy.zeros(self.shape, numpy.float)
        cdef numpy.ndarray[long, ndim=1, mode="c"] rowInds
        cdef numpy.ndarray[long, ndim=1, mode="c"] colInds
        cdef unsigned int i
        
        (rowInds, colInds) = self.nonzero()
            
        for i in range(rowInds.shape[0]): 
            result[rowInds[i], colInds[i]] += self.thisPtr.coeff(rowInds[i], colInds[i])   
            
        return result 
        
        
    def min(self): 
        """
        Find the minimum element of this array. 
        """
        cdef numpy.ndarray[long, ndim=1, mode="c"] rowInds
        cdef numpy.ndarray[long, ndim=1, mode="c"] colInds
        cdef unsigned int i
        cdef short minVal 
        
        if self.size == 0: 
            return float("nan")
        elif self.getnnz() != self.size: 
            minVal = 0 
        
        (rowInds, colInds) = self.nonzero()
            
        for i in range(rowInds.shape[0]): 
            if i == 0: 
                minVal = self.thisPtr.coeff(rowInds[i], colInds[i])
            
            if self.thisPtr.coeff(rowInds[i], colInds[i]) < minVal: 
                minVal = self.thisPtr.coeff(rowInds[i], colInds[i])
            
        return minVal 
        
    def max(self): 
        """
        Find the maximum element of this array. 
        """
        cdef numpy.ndarray[long, ndim=1, mode="c"] rowInds
        cdef numpy.ndarray[long, ndim=1, mode="c"] colInds
        cdef unsigned int i
        cdef short maxVal
        
        if self.size == 0: 
            return float("nan")
        elif self.getnnz() != self.size: 
            maxVal = 0 
        
        (rowInds, colInds) = self.nonzero()
            
        for i in range(rowInds.shape[0]): 
            if i == 0: 
                maxVal = self.thisPtr.coeff(rowInds[i], colInds[i])            
            
            if self.thisPtr.coeff(rowInds[i], colInds[i]) > maxVal: 
                maxVal = self.thisPtr.coeff(rowInds[i], colInds[i])
            
        return maxVal 
        
    def var(self): 
        """
        Return the variance of the elements of this array. 
        """
        cdef double mean = self.mean() 
        cdef numpy.ndarray[long, ndim=1, mode="c"] rowInds
        cdef numpy.ndarray[long, ndim=1, mode="c"] colInds
        cdef unsigned int i
        cdef double result = 0
        
        if self.size == 0: 
            return float("nan")
        
        (rowInds, colInds) = self.nonzero()
            
        for i in range(rowInds.shape[0]): 
            result += (self.thisPtr.coeff(rowInds[i], colInds[i]) - mean)**2
        
        result += (self.size - self.getnnz())*mean**2
        result /= float(self.size)
        
        return result 
    
    def std(self): 
        """
        Return the standard deviation of the array elements. 
        """
        return numpy.sqrt(self.var())
        
    def __abs__(self): 
        """
        Return a matrix whose elements are the absolute values of this array. 
        """
        cdef csarray_short result = csarray_short((self.shape[1], self.shape[0]))
        del result.thisPtr
        result.thisPtr = new SparseMatrixExt[short](self.thisPtr.abs())
        return result 
    
    def __neg__(self): 
        """
        Return the negation of this array. 
        """
        cdef csarray_short result = csarray_short((self.shape[1], self.shape[0]))
        del result.thisPtr
        result.thisPtr = new SparseMatrixExt[short](self.thisPtr.negate())
        return result 
    

    def __add__(csarray_short self, csarray_short A): 
        """
        Add two matrices together. 
        """
        if self.shape != A.shape: 
            raise ValueError("Cannot add matrices of shapes " + str(self.shape) + " and " + str(A.shape))
        
        cdef csarray_short result = csarray_short((self.shape[0], A.shape[1]))
        del result.thisPtr
        result.thisPtr = new SparseMatrixExt[short](self.thisPtr.add(deref(A.thisPtr)))
        return result     
        
    def __sub__(csarray_short self, csarray_short A): 
        """
        Subtract one matrix from another.  
        """
        if self.shape != A.shape: 
            raise ValueError("Cannot subtract matrices of shapes " + str(self.shape) + " and " + str(A.shape))
        
        cdef csarray_short result = csarray_short((self.shape[0], A.shape[1]))
        del result.thisPtr
        result.thisPtr = new SparseMatrixExt[short](self.thisPtr.subtract(deref(A.thisPtr)))
        return result    
     
    def hadamard(self, csarray_short A): 
        """
        Find the element-wise matrix (hadamard) product. 
        """
        if self.shape != A.shape: 
            raise ValueError("Cannot elementwise multiply matrices of shapes " + str(self.shape) + " and " + str(A.shape))
        
        cdef csarray_short result = csarray_short((self.shape[0], A.shape[1]))
        del result.thisPtr
        result.thisPtr = new SparseMatrixExt[short](self.thisPtr.hadamard(deref(A.thisPtr)))
        return result 

    def compress(self): 
        """
        Turn this matrix into compressed sparse format by freeing extra memory 
        space in the buffer. 
        """
        self.thisPtr.makeCompressed()
        
    
    def reserve(self, int n): 
        """
        Reserve n nonzero entries and turns the matrix into uncompressed mode. 
        """
        self.thisPtr.reserve(n)
        
    def dot(self, csarray_short A): 
        if self.shape[1] != A.shape[0]: 
            raise ValueError("Cannot multiply matrices of shapes " + str(self.shape) + " and " + str(A.shape))
        
        cdef csarray_short result = csarray_short((self.shape[0], A.shape[1]))
        del result.thisPtr
        result.thisPtr = new SparseMatrixExt[short](self.thisPtr.dot(deref(A.thisPtr)))
        return result 
        
    def transpose(self): 
        """
        Find the transpose of this matrix. 
        """
        cdef csarray_short result = csarray_short((self.shape[1], self.shape[0]))
        del result.thisPtr
        result.thisPtr = new SparseMatrixExt[short](self.thisPtr.trans())
        return result 
   
    #def norm(self, ord="fro"): 
        """
        Return the norm of this array. Currently only the Frobenius norm is 
        supported. 
        """
        #return self.thisPtr.norm()
   
    def ones(self): 
        """
        Fill the array with ones. 
        """
        self.thisPtr.fill(1)
        
   
    shape = property(__getShape)
    size = property(__getSize)
    ndim = property(__getNDim)
    dtype = property(__getDType)
    

    
cdef class csarray_int:
    cdef SparseMatrixExt[int] *thisPtr     
    def __cinit__(self, shape):
        """
        Create a new column major dynamic array.
        """

        self.thisPtr = new SparseMatrixExt[int](shape[0], shape[1]) 
            
    def __dealloc__(self): 
        """
        Deallocate the SparseMatrixExt object.  
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
            for k, index in enumerate(inds):  
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
        
        cdef csarray_int result = csarray_int((rowInds.shape[0], colInds.shape[0]))     
        
        rowIndsC = numpy.ascontiguousarray(rowInds, dtype=numpy.int32) 
        colIndsC = numpy.ascontiguousarray(colInds, dtype=numpy.int32) 
        
        if rowInds.shape[0] != 0 and colInds.shape[0] != 0: 
            self.thisPtr.slice(&rowIndsC[0], rowIndsC.shape[0], &colIndsC[0], colIndsC.shape[0], result.thisPtr) 
        return result 
        
    def nonzero(self): 
        """
        Return a tuple of arrays corresponding to nonzero elements. 
        """
        cdef numpy.ndarray[long, ndim=1, mode="c"] rowInds = numpy.zeros(self.getnnz(), dtype=numpy.int64) 
        cdef numpy.ndarray[long, ndim=1, mode="c"] colInds = numpy.zeros(self.getnnz(), dtype=numpy.int64)  
        
        if self.getnnz() != 0:
            self.thisPtr.nonZeroInds(&rowInds[0], &colInds[0])
        
        return (rowInds, colInds)
                    
    def __setitem__(self, inds, val):
        """
        Set elements of the array. If i,j = inds are integers then the corresponding 
        value in the array is set. 
        """
        i, j = inds 
        
        if type(i) == numpy.ndarray and type(j) == numpy.ndarray: 
            self.put(val, i, j)
        else:
            i = int(i) 
            j = int(j)
            if i < 0 or i>=self.thisPtr.rows(): 
                raise ValueError("Invalid row index " + str(i)) 
            if j < 0 or j>=self.thisPtr.cols(): 
                raise ValueError("Invalid col index " + str(j))        
            
            self.thisPtr.insertVal(i, j, val) 

    
    def put(self, val, numpy.ndarray[numpy.int_t, ndim=1] rowInds not None, numpy.ndarray[numpy.int_t, ndim=1] colInds not None): 
        """
        Select rowInds and colInds
        """
        cdef unsigned int ix 
        self.reserve(len(rowInds))
        
        if type(val) == numpy.ndarray: 
            for ix in range(len(rowInds)): 
                self.thisPtr.insertVal(rowInds[ix], colInds[ix], val[ix])
        else:
            for ix in range(len(rowInds)): 
                self.thisPtr.insertVal(rowInds[ix], colInds[ix], val)
            

    def sum(self, axis=None): 
        """
        Sum all of the elements in this array. If one specifies an axis 
        then we sum along the axis. 
        """
        cdef numpy.ndarray[double, ndim=1, mode="c"] result    
        cdef numpy.ndarray[long, ndim=1, mode="c"] rowInds
        cdef numpy.ndarray[long, ndim=1, mode="c"] colInds
        cdef unsigned int i
        
        if axis==None: 
            """
            scalarResult = 0 
            (rowInds, colInds) = self.nonzero()
            
            for i in range(rowInds.shape[0]): 
                scalarResult += self.thisPtr.coeff(rowInds[i], colInds[i])  
            
            return scalarResult
            """
            return self.thisPtr.sumValues()
            #There seems to be a very temporamental problem with thisPtr.sum()
            #return self.thisPtr.sum()
        elif axis==0: 
            result = numpy.zeros(self.shape[1], dtype=numpy.float) 
            (rowInds, colInds) = self.nonzero()
            
            for i in range(rowInds.shape[0]): 
                result[colInds[i]] += self.thisPtr.coeff(rowInds[i], colInds[i])   
        elif axis==1: 
            result = numpy.zeros(self.shape[0], dtype=numpy.float) 
            (rowInds, colInds) = self.nonzero()
            
            for i in range(rowInds.shape[0]): 
                result[rowInds[i]] += self.thisPtr.coeff(rowInds[i], colInds[i])  
        else:
            raise ValueError("Invalid axis: " + str(axis))
            
        return result 
                
        
    def mean(self, axis=None): 
        """
        Find the mean value of this array. 
        """
        if self.thisPtr.size() != 0:
            if axis ==None: 
                return self.sum()/float(self.thisPtr.size())
            elif axis == 0: 
                return self.sum(0)/float(self.shape[0])
            elif axis == 1: 
                return self.sum(1)/float(self.shape[1])
        else: 
            return float("nan")
     
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
        
    def trace(self): 
        """
        Returns the trace of the array which is simply the sum of the diagonal 
        entries. 
        """
        return self.diag().sum()
         
    def __mul__(self, double x):
        """
        Return a new array multiplied by a scalar value x. 
        """
        cdef csarray_int result = self.copy() 
        result.thisPtr.scalarMultiply(x)
        return result 
        
    def copy(self): 
        """
        Return a copied version of this array. 
        """
        cdef csarray_int result = csarray_int(self.shape)
        del result.thisPtr
        result.thisPtr = new SparseMatrixExt[int](deref(self.thisPtr))
        return result 
        
    def toarray(self): 
        """
        Convert this sparse matrix into a numpy array. 
        """
        cdef numpy.ndarray[double, ndim=2, mode="c"] result = numpy.zeros(self.shape, numpy.float)
        cdef numpy.ndarray[long, ndim=1, mode="c"] rowInds
        cdef numpy.ndarray[long, ndim=1, mode="c"] colInds
        cdef unsigned int i
        
        (rowInds, colInds) = self.nonzero()
            
        for i in range(rowInds.shape[0]): 
            result[rowInds[i], colInds[i]] += self.thisPtr.coeff(rowInds[i], colInds[i])   
            
        return result 
        
        
    def min(self): 
        """
        Find the minimum element of this array. 
        """
        cdef numpy.ndarray[long, ndim=1, mode="c"] rowInds
        cdef numpy.ndarray[long, ndim=1, mode="c"] colInds
        cdef unsigned int i
        cdef int minVal 
        
        if self.size == 0: 
            return float("nan")
        elif self.getnnz() != self.size: 
            minVal = 0 
        
        (rowInds, colInds) = self.nonzero()
            
        for i in range(rowInds.shape[0]): 
            if i == 0: 
                minVal = self.thisPtr.coeff(rowInds[i], colInds[i])
            
            if self.thisPtr.coeff(rowInds[i], colInds[i]) < minVal: 
                minVal = self.thisPtr.coeff(rowInds[i], colInds[i])
            
        return minVal 
        
    def max(self): 
        """
        Find the maximum element of this array. 
        """
        cdef numpy.ndarray[long, ndim=1, mode="c"] rowInds
        cdef numpy.ndarray[long, ndim=1, mode="c"] colInds
        cdef unsigned int i
        cdef int maxVal
        
        if self.size == 0: 
            return float("nan")
        elif self.getnnz() != self.size: 
            maxVal = 0 
        
        (rowInds, colInds) = self.nonzero()
            
        for i in range(rowInds.shape[0]): 
            if i == 0: 
                maxVal = self.thisPtr.coeff(rowInds[i], colInds[i])            
            
            if self.thisPtr.coeff(rowInds[i], colInds[i]) > maxVal: 
                maxVal = self.thisPtr.coeff(rowInds[i], colInds[i])
            
        return maxVal 
        
    def var(self): 
        """
        Return the variance of the elements of this array. 
        """
        cdef double mean = self.mean() 
        cdef numpy.ndarray[long, ndim=1, mode="c"] rowInds
        cdef numpy.ndarray[long, ndim=1, mode="c"] colInds
        cdef unsigned int i
        cdef double result = 0
        
        if self.size == 0: 
            return float("nan")
        
        (rowInds, colInds) = self.nonzero()
            
        for i in range(rowInds.shape[0]): 
            result += (self.thisPtr.coeff(rowInds[i], colInds[i]) - mean)**2
        
        result += (self.size - self.getnnz())*mean**2
        result /= float(self.size)
        
        return result 
    
    def std(self): 
        """
        Return the standard deviation of the array elements. 
        """
        return numpy.sqrt(self.var())
        
    def __abs__(self): 
        """
        Return a matrix whose elements are the absolute values of this array. 
        """
        cdef csarray_int result = csarray_int((self.shape[1], self.shape[0]))
        del result.thisPtr
        result.thisPtr = new SparseMatrixExt[int](self.thisPtr.abs())
        return result 
    
    def __neg__(self): 
        """
        Return the negation of this array. 
        """
        cdef csarray_int result = csarray_int((self.shape[1], self.shape[0]))
        del result.thisPtr
        result.thisPtr = new SparseMatrixExt[int](self.thisPtr.negate())
        return result 
    

    def __add__(csarray_int self, csarray_int A): 
        """
        Add two matrices together. 
        """
        if self.shape != A.shape: 
            raise ValueError("Cannot add matrices of shapes " + str(self.shape) + " and " + str(A.shape))
        
        cdef csarray_int result = csarray_int((self.shape[0], A.shape[1]))
        del result.thisPtr
        result.thisPtr = new SparseMatrixExt[int](self.thisPtr.add(deref(A.thisPtr)))
        return result     
        
    def __sub__(csarray_int self, csarray_int A): 
        """
        Subtract one matrix from another.  
        """
        if self.shape != A.shape: 
            raise ValueError("Cannot subtract matrices of shapes " + str(self.shape) + " and " + str(A.shape))
        
        cdef csarray_int result = csarray_int((self.shape[0], A.shape[1]))
        del result.thisPtr
        result.thisPtr = new SparseMatrixExt[int](self.thisPtr.subtract(deref(A.thisPtr)))
        return result    
     
    def hadamard(self, csarray_int A): 
        """
        Find the element-wise matrix (hadamard) product. 
        """
        if self.shape != A.shape: 
            raise ValueError("Cannot elementwise multiply matrices of shapes " + str(self.shape) + " and " + str(A.shape))
        
        cdef csarray_int result = csarray_int((self.shape[0], A.shape[1]))
        del result.thisPtr
        result.thisPtr = new SparseMatrixExt[int](self.thisPtr.hadamard(deref(A.thisPtr)))
        return result 

    def compress(self): 
        """
        Turn this matrix into compressed sparse format by freeing extra memory 
        space in the buffer. 
        """
        self.thisPtr.makeCompressed()
        
    
    def reserve(self, int n): 
        """
        Reserve n nonzero entries and turns the matrix into uncompressed mode. 
        """
        self.thisPtr.reserve(n)
        
    def dot(self, csarray_int A): 
        if self.shape[1] != A.shape[0]: 
            raise ValueError("Cannot multiply matrices of shapes " + str(self.shape) + " and " + str(A.shape))
        
        cdef csarray_int result = csarray_int((self.shape[0], A.shape[1]))
        del result.thisPtr
        result.thisPtr = new SparseMatrixExt[int](self.thisPtr.dot(deref(A.thisPtr)))
        return result 
        
    def transpose(self): 
        """
        Find the transpose of this matrix. 
        """
        cdef csarray_int result = csarray_int((self.shape[1], self.shape[0]))
        del result.thisPtr
        result.thisPtr = new SparseMatrixExt[int](self.thisPtr.trans())
        return result 
   
    #def norm(self, ord="fro"): 
        """
        Return the norm of this array. Currently only the Frobenius norm is 
        supported. 
        """
        #return self.thisPtr.norm()
   
    def ones(self): 
        """
        Fill the array with ones. 
        """
        self.thisPtr.fill(1)
        
   
    shape = property(__getShape)
    size = property(__getSize)
    ndim = property(__getNDim)
    dtype = property(__getDType)
    

    
cdef class csarray_long:
    cdef SparseMatrixExt[long] *thisPtr     
    def __cinit__(self, shape):
        """
        Create a new column major dynamic array.
        """

        self.thisPtr = new SparseMatrixExt[long](shape[0], shape[1]) 
            
    def __dealloc__(self): 
        """
        Deallocate the SparseMatrixExt object.  
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
            for k, index in enumerate(inds):  
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
        
        cdef csarray_long result = csarray_long((rowInds.shape[0], colInds.shape[0]))     
        
        rowIndsC = numpy.ascontiguousarray(rowInds, dtype=numpy.int32) 
        colIndsC = numpy.ascontiguousarray(colInds, dtype=numpy.int32) 
        
        if rowInds.shape[0] != 0 and colInds.shape[0] != 0: 
            self.thisPtr.slice(&rowIndsC[0], rowIndsC.shape[0], &colIndsC[0], colIndsC.shape[0], result.thisPtr) 
        return result 
        
    def nonzero(self): 
        """
        Return a tuple of arrays corresponding to nonzero elements. 
        """
        cdef numpy.ndarray[long, ndim=1, mode="c"] rowInds = numpy.zeros(self.getnnz(), dtype=numpy.int64) 
        cdef numpy.ndarray[long, ndim=1, mode="c"] colInds = numpy.zeros(self.getnnz(), dtype=numpy.int64)  
        
        if self.getnnz() != 0:
            self.thisPtr.nonZeroInds(&rowInds[0], &colInds[0])
        
        return (rowInds, colInds)
                    
    def __setitem__(self, inds, val):
        """
        Set elements of the array. If i,j = inds are integers then the corresponding 
        value in the array is set. 
        """
        i, j = inds 
        
        if type(i) == numpy.ndarray and type(j) == numpy.ndarray: 
            self.put(val, i, j)
        else:
            i = int(i) 
            j = int(j)
            if i < 0 or i>=self.thisPtr.rows(): 
                raise ValueError("Invalid row index " + str(i)) 
            if j < 0 or j>=self.thisPtr.cols(): 
                raise ValueError("Invalid col index " + str(j))        
            
            self.thisPtr.insertVal(i, j, val) 

    
    def put(self, val, numpy.ndarray[numpy.int_t, ndim=1] rowInds not None, numpy.ndarray[numpy.int_t, ndim=1] colInds not None): 
        """
        Select rowInds and colInds
        """
        cdef unsigned int ix 
        self.reserve(len(rowInds))
        
        if type(val) == numpy.ndarray: 
            for ix in range(len(rowInds)): 
                self.thisPtr.insertVal(rowInds[ix], colInds[ix], val[ix])
        else:
            for ix in range(len(rowInds)): 
                self.thisPtr.insertVal(rowInds[ix], colInds[ix], val)
            

    def sum(self, axis=None): 
        """
        Sum all of the elements in this array. If one specifies an axis 
        then we sum along the axis. 
        """
        cdef numpy.ndarray[double, ndim=1, mode="c"] result    
        cdef numpy.ndarray[long, ndim=1, mode="c"] rowInds
        cdef numpy.ndarray[long, ndim=1, mode="c"] colInds
        cdef unsigned int i
        
        if axis==None: 
            """
            scalarResult = 0 
            (rowInds, colInds) = self.nonzero()
            
            for i in range(rowInds.shape[0]): 
                scalarResult += self.thisPtr.coeff(rowInds[i], colInds[i])  
            
            return scalarResult
            """
            return self.thisPtr.sumValues()
            #There seems to be a very temporamental problem with thisPtr.sum()
            #return self.thisPtr.sum()
        elif axis==0: 
            result = numpy.zeros(self.shape[1], dtype=numpy.float) 
            (rowInds, colInds) = self.nonzero()
            
            for i in range(rowInds.shape[0]): 
                result[colInds[i]] += self.thisPtr.coeff(rowInds[i], colInds[i])   
        elif axis==1: 
            result = numpy.zeros(self.shape[0], dtype=numpy.float) 
            (rowInds, colInds) = self.nonzero()
            
            for i in range(rowInds.shape[0]): 
                result[rowInds[i]] += self.thisPtr.coeff(rowInds[i], colInds[i])  
        else:
            raise ValueError("Invalid axis: " + str(axis))
            
        return result 
                
        
    def mean(self, axis=None): 
        """
        Find the mean value of this array. 
        """
        if self.thisPtr.size() != 0:
            if axis ==None: 
                return self.sum()/float(self.thisPtr.size())
            elif axis == 0: 
                return self.sum(0)/float(self.shape[0])
            elif axis == 1: 
                return self.sum(1)/float(self.shape[1])
        else: 
            return float("nan")
     
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
        
    def trace(self): 
        """
        Returns the trace of the array which is simply the sum of the diagonal 
        entries. 
        """
        return self.diag().sum()
         
    def __mul__(self, double x):
        """
        Return a new array multiplied by a scalar value x. 
        """
        cdef csarray_long result = self.copy() 
        result.thisPtr.scalarMultiply(x)
        return result 
        
    def copy(self): 
        """
        Return a copied version of this array. 
        """
        cdef csarray_long result = csarray_long(self.shape)
        del result.thisPtr
        result.thisPtr = new SparseMatrixExt[long](deref(self.thisPtr))
        return result 
        
    def toarray(self): 
        """
        Convert this sparse matrix into a numpy array. 
        """
        cdef numpy.ndarray[double, ndim=2, mode="c"] result = numpy.zeros(self.shape, numpy.float)
        cdef numpy.ndarray[long, ndim=1, mode="c"] rowInds
        cdef numpy.ndarray[long, ndim=1, mode="c"] colInds
        cdef unsigned int i
        
        (rowInds, colInds) = self.nonzero()
            
        for i in range(rowInds.shape[0]): 
            result[rowInds[i], colInds[i]] += self.thisPtr.coeff(rowInds[i], colInds[i])   
            
        return result 
        
        
    def min(self): 
        """
        Find the minimum element of this array. 
        """
        cdef numpy.ndarray[long, ndim=1, mode="c"] rowInds
        cdef numpy.ndarray[long, ndim=1, mode="c"] colInds
        cdef unsigned int i
        cdef long minVal 
        
        if self.size == 0: 
            return float("nan")
        elif self.getnnz() != self.size: 
            minVal = 0 
        
        (rowInds, colInds) = self.nonzero()
            
        for i in range(rowInds.shape[0]): 
            if i == 0: 
                minVal = self.thisPtr.coeff(rowInds[i], colInds[i])
            
            if self.thisPtr.coeff(rowInds[i], colInds[i]) < minVal: 
                minVal = self.thisPtr.coeff(rowInds[i], colInds[i])
            
        return minVal 
        
    def max(self): 
        """
        Find the maximum element of this array. 
        """
        cdef numpy.ndarray[long, ndim=1, mode="c"] rowInds
        cdef numpy.ndarray[long, ndim=1, mode="c"] colInds
        cdef unsigned int i
        cdef long maxVal
        
        if self.size == 0: 
            return float("nan")
        elif self.getnnz() != self.size: 
            maxVal = 0 
        
        (rowInds, colInds) = self.nonzero()
            
        for i in range(rowInds.shape[0]): 
            if i == 0: 
                maxVal = self.thisPtr.coeff(rowInds[i], colInds[i])            
            
            if self.thisPtr.coeff(rowInds[i], colInds[i]) > maxVal: 
                maxVal = self.thisPtr.coeff(rowInds[i], colInds[i])
            
        return maxVal 
        
    def var(self): 
        """
        Return the variance of the elements of this array. 
        """
        cdef double mean = self.mean() 
        cdef numpy.ndarray[long, ndim=1, mode="c"] rowInds
        cdef numpy.ndarray[long, ndim=1, mode="c"] colInds
        cdef unsigned int i
        cdef double result = 0
        
        if self.size == 0: 
            return float("nan")
        
        (rowInds, colInds) = self.nonzero()
            
        for i in range(rowInds.shape[0]): 
            result += (self.thisPtr.coeff(rowInds[i], colInds[i]) - mean)**2
        
        result += (self.size - self.getnnz())*mean**2
        result /= float(self.size)
        
        return result 
    
    def std(self): 
        """
        Return the standard deviation of the array elements. 
        """
        return numpy.sqrt(self.var())
        
    def __abs__(self): 
        """
        Return a matrix whose elements are the absolute values of this array. 
        """
        cdef csarray_long result = csarray_long((self.shape[1], self.shape[0]))
        del result.thisPtr
        result.thisPtr = new SparseMatrixExt[long](self.thisPtr.abs())
        return result 
    
    def __neg__(self): 
        """
        Return the negation of this array. 
        """
        cdef csarray_long result = csarray_long((self.shape[1], self.shape[0]))
        del result.thisPtr
        result.thisPtr = new SparseMatrixExt[long](self.thisPtr.negate())
        return result 
    

    def __add__(csarray_long self, csarray_long A): 
        """
        Add two matrices together. 
        """
        if self.shape != A.shape: 
            raise ValueError("Cannot add matrices of shapes " + str(self.shape) + " and " + str(A.shape))
        
        cdef csarray_long result = csarray_long((self.shape[0], A.shape[1]))
        del result.thisPtr
        result.thisPtr = new SparseMatrixExt[long](self.thisPtr.add(deref(A.thisPtr)))
        return result     
        
    def __sub__(csarray_long self, csarray_long A): 
        """
        Subtract one matrix from another.  
        """
        if self.shape != A.shape: 
            raise ValueError("Cannot subtract matrices of shapes " + str(self.shape) + " and " + str(A.shape))
        
        cdef csarray_long result = csarray_long((self.shape[0], A.shape[1]))
        del result.thisPtr
        result.thisPtr = new SparseMatrixExt[long](self.thisPtr.subtract(deref(A.thisPtr)))
        return result    
     
    def hadamard(self, csarray_long A): 
        """
        Find the element-wise matrix (hadamard) product. 
        """
        if self.shape != A.shape: 
            raise ValueError("Cannot elementwise multiply matrices of shapes " + str(self.shape) + " and " + str(A.shape))
        
        cdef csarray_long result = csarray_long((self.shape[0], A.shape[1]))
        del result.thisPtr
        result.thisPtr = new SparseMatrixExt[long](self.thisPtr.hadamard(deref(A.thisPtr)))
        return result 

    def compress(self): 
        """
        Turn this matrix into compressed sparse format by freeing extra memory 
        space in the buffer. 
        """
        self.thisPtr.makeCompressed()
        
    
    def reserve(self, int n): 
        """
        Reserve n nonzero entries and turns the matrix into uncompressed mode. 
        """
        self.thisPtr.reserve(n)
        
    def dot(self, csarray_long A): 
        if self.shape[1] != A.shape[0]: 
            raise ValueError("Cannot multiply matrices of shapes " + str(self.shape) + " and " + str(A.shape))
        
        cdef csarray_long result = csarray_long((self.shape[0], A.shape[1]))
        del result.thisPtr
        result.thisPtr = new SparseMatrixExt[long](self.thisPtr.dot(deref(A.thisPtr)))
        return result 
        
    def transpose(self): 
        """
        Find the transpose of this matrix. 
        """
        cdef csarray_long result = csarray_long((self.shape[1], self.shape[0]))
        del result.thisPtr
        result.thisPtr = new SparseMatrixExt[long](self.thisPtr.trans())
        return result 
   
    #def norm(self, ord="fro"): 
        """
        Return the norm of this array. Currently only the Frobenius norm is 
        supported. 
        """
        #return self.thisPtr.norm()
   
    def ones(self): 
        """
        Fill the array with ones. 
        """
        self.thisPtr.fill(1)
        
   
    shape = property(__getShape)
    size = property(__getSize)
    ndim = property(__getNDim)
    dtype = property(__getDType)
    

    
cdef class csarray_float:
    cdef SparseMatrixExt[float] *thisPtr     
    def __cinit__(self, shape):
        """
        Create a new column major dynamic array.
        """

        self.thisPtr = new SparseMatrixExt[float](shape[0], shape[1]) 
            
    def __dealloc__(self): 
        """
        Deallocate the SparseMatrixExt object.  
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
            for k, index in enumerate(inds):  
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
        
        cdef csarray_float result = csarray_float((rowInds.shape[0], colInds.shape[0]))     
        
        rowIndsC = numpy.ascontiguousarray(rowInds, dtype=numpy.int32) 
        colIndsC = numpy.ascontiguousarray(colInds, dtype=numpy.int32) 
        
        if rowInds.shape[0] != 0 and colInds.shape[0] != 0: 
            self.thisPtr.slice(&rowIndsC[0], rowIndsC.shape[0], &colIndsC[0], colIndsC.shape[0], result.thisPtr) 
        return result 
        
    def nonzero(self): 
        """
        Return a tuple of arrays corresponding to nonzero elements. 
        """
        cdef numpy.ndarray[long, ndim=1, mode="c"] rowInds = numpy.zeros(self.getnnz(), dtype=numpy.int64) 
        cdef numpy.ndarray[long, ndim=1, mode="c"] colInds = numpy.zeros(self.getnnz(), dtype=numpy.int64)  
        
        if self.getnnz() != 0:
            self.thisPtr.nonZeroInds(&rowInds[0], &colInds[0])
        
        return (rowInds, colInds)
                    
    def __setitem__(self, inds, val):
        """
        Set elements of the array. If i,j = inds are integers then the corresponding 
        value in the array is set. 
        """
        i, j = inds 
        
        if type(i) == numpy.ndarray and type(j) == numpy.ndarray: 
            self.put(val, i, j)
        else:
            i = int(i) 
            j = int(j)
            if i < 0 or i>=self.thisPtr.rows(): 
                raise ValueError("Invalid row index " + str(i)) 
            if j < 0 or j>=self.thisPtr.cols(): 
                raise ValueError("Invalid col index " + str(j))        
            
            self.thisPtr.insertVal(i, j, val) 

    
    def put(self, val, numpy.ndarray[numpy.int_t, ndim=1] rowInds not None, numpy.ndarray[numpy.int_t, ndim=1] colInds not None): 
        """
        Select rowInds and colInds
        """
        cdef unsigned int ix 
        self.reserve(len(rowInds))
        
        if type(val) == numpy.ndarray: 
            for ix in range(len(rowInds)): 
                self.thisPtr.insertVal(rowInds[ix], colInds[ix], val[ix])
        else:
            for ix in range(len(rowInds)): 
                self.thisPtr.insertVal(rowInds[ix], colInds[ix], val)
            

    def sum(self, axis=None): 
        """
        Sum all of the elements in this array. If one specifies an axis 
        then we sum along the axis. 
        """
        cdef numpy.ndarray[double, ndim=1, mode="c"] result    
        cdef numpy.ndarray[long, ndim=1, mode="c"] rowInds
        cdef numpy.ndarray[long, ndim=1, mode="c"] colInds
        cdef unsigned int i
        
        if axis==None: 
            """
            scalarResult = 0 
            (rowInds, colInds) = self.nonzero()
            
            for i in range(rowInds.shape[0]): 
                scalarResult += self.thisPtr.coeff(rowInds[i], colInds[i])  
            
            return scalarResult
            """
            return self.thisPtr.sumValues()
            #There seems to be a very temporamental problem with thisPtr.sum()
            #return self.thisPtr.sum()
        elif axis==0: 
            result = numpy.zeros(self.shape[1], dtype=numpy.float) 
            (rowInds, colInds) = self.nonzero()
            
            for i in range(rowInds.shape[0]): 
                result[colInds[i]] += self.thisPtr.coeff(rowInds[i], colInds[i])   
        elif axis==1: 
            result = numpy.zeros(self.shape[0], dtype=numpy.float) 
            (rowInds, colInds) = self.nonzero()
            
            for i in range(rowInds.shape[0]): 
                result[rowInds[i]] += self.thisPtr.coeff(rowInds[i], colInds[i])  
        else:
            raise ValueError("Invalid axis: " + str(axis))
            
        return result 
                
        
    def mean(self, axis=None): 
        """
        Find the mean value of this array. 
        """
        if self.thisPtr.size() != 0:
            if axis ==None: 
                return self.sum()/float(self.thisPtr.size())
            elif axis == 0: 
                return self.sum(0)/float(self.shape[0])
            elif axis == 1: 
                return self.sum(1)/float(self.shape[1])
        else: 
            return float("nan")
     
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
        
    def trace(self): 
        """
        Returns the trace of the array which is simply the sum of the diagonal 
        entries. 
        """
        return self.diag().sum()
         
    def __mul__(self, double x):
        """
        Return a new array multiplied by a scalar value x. 
        """
        cdef csarray_float result = self.copy() 
        result.thisPtr.scalarMultiply(x)
        return result 
        
    def copy(self): 
        """
        Return a copied version of this array. 
        """
        cdef csarray_float result = csarray_float(self.shape)
        del result.thisPtr
        result.thisPtr = new SparseMatrixExt[float](deref(self.thisPtr))
        return result 
        
    def toarray(self): 
        """
        Convert this sparse matrix into a numpy array. 
        """
        cdef numpy.ndarray[double, ndim=2, mode="c"] result = numpy.zeros(self.shape, numpy.float)
        cdef numpy.ndarray[long, ndim=1, mode="c"] rowInds
        cdef numpy.ndarray[long, ndim=1, mode="c"] colInds
        cdef unsigned int i
        
        (rowInds, colInds) = self.nonzero()
            
        for i in range(rowInds.shape[0]): 
            result[rowInds[i], colInds[i]] += self.thisPtr.coeff(rowInds[i], colInds[i])   
            
        return result 
        
        
    def min(self): 
        """
        Find the minimum element of this array. 
        """
        cdef numpy.ndarray[long, ndim=1, mode="c"] rowInds
        cdef numpy.ndarray[long, ndim=1, mode="c"] colInds
        cdef unsigned int i
        cdef float minVal 
        
        if self.size == 0: 
            return float("nan")
        elif self.getnnz() != self.size: 
            minVal = 0 
        
        (rowInds, colInds) = self.nonzero()
            
        for i in range(rowInds.shape[0]): 
            if i == 0: 
                minVal = self.thisPtr.coeff(rowInds[i], colInds[i])
            
            if self.thisPtr.coeff(rowInds[i], colInds[i]) < minVal: 
                minVal = self.thisPtr.coeff(rowInds[i], colInds[i])
            
        return minVal 
        
    def max(self): 
        """
        Find the maximum element of this array. 
        """
        cdef numpy.ndarray[long, ndim=1, mode="c"] rowInds
        cdef numpy.ndarray[long, ndim=1, mode="c"] colInds
        cdef unsigned int i
        cdef float maxVal
        
        if self.size == 0: 
            return float("nan")
        elif self.getnnz() != self.size: 
            maxVal = 0 
        
        (rowInds, colInds) = self.nonzero()
            
        for i in range(rowInds.shape[0]): 
            if i == 0: 
                maxVal = self.thisPtr.coeff(rowInds[i], colInds[i])            
            
            if self.thisPtr.coeff(rowInds[i], colInds[i]) > maxVal: 
                maxVal = self.thisPtr.coeff(rowInds[i], colInds[i])
            
        return maxVal 
        
    def var(self): 
        """
        Return the variance of the elements of this array. 
        """
        cdef double mean = self.mean() 
        cdef numpy.ndarray[long, ndim=1, mode="c"] rowInds
        cdef numpy.ndarray[long, ndim=1, mode="c"] colInds
        cdef unsigned int i
        cdef double result = 0
        
        if self.size == 0: 
            return float("nan")
        
        (rowInds, colInds) = self.nonzero()
            
        for i in range(rowInds.shape[0]): 
            result += (self.thisPtr.coeff(rowInds[i], colInds[i]) - mean)**2
        
        result += (self.size - self.getnnz())*mean**2
        result /= float(self.size)
        
        return result 
    
    def std(self): 
        """
        Return the standard deviation of the array elements. 
        """
        return numpy.sqrt(self.var())
        
    def __abs__(self): 
        """
        Return a matrix whose elements are the absolute values of this array. 
        """
        cdef csarray_float result = csarray_float((self.shape[1], self.shape[0]))
        del result.thisPtr
        result.thisPtr = new SparseMatrixExt[float](self.thisPtr.abs())
        return result 
    
    def __neg__(self): 
        """
        Return the negation of this array. 
        """
        cdef csarray_float result = csarray_float((self.shape[1], self.shape[0]))
        del result.thisPtr
        result.thisPtr = new SparseMatrixExt[float](self.thisPtr.negate())
        return result 
    

    def __add__(csarray_float self, csarray_float A): 
        """
        Add two matrices together. 
        """
        if self.shape != A.shape: 
            raise ValueError("Cannot add matrices of shapes " + str(self.shape) + " and " + str(A.shape))
        
        cdef csarray_float result = csarray_float((self.shape[0], A.shape[1]))
        del result.thisPtr
        result.thisPtr = new SparseMatrixExt[float](self.thisPtr.add(deref(A.thisPtr)))
        return result     
        
    def __sub__(csarray_float self, csarray_float A): 
        """
        Subtract one matrix from another.  
        """
        if self.shape != A.shape: 
            raise ValueError("Cannot subtract matrices of shapes " + str(self.shape) + " and " + str(A.shape))
        
        cdef csarray_float result = csarray_float((self.shape[0], A.shape[1]))
        del result.thisPtr
        result.thisPtr = new SparseMatrixExt[float](self.thisPtr.subtract(deref(A.thisPtr)))
        return result    
     
    def hadamard(self, csarray_float A): 
        """
        Find the element-wise matrix (hadamard) product. 
        """
        if self.shape != A.shape: 
            raise ValueError("Cannot elementwise multiply matrices of shapes " + str(self.shape) + " and " + str(A.shape))
        
        cdef csarray_float result = csarray_float((self.shape[0], A.shape[1]))
        del result.thisPtr
        result.thisPtr = new SparseMatrixExt[float](self.thisPtr.hadamard(deref(A.thisPtr)))
        return result 

    def compress(self): 
        """
        Turn this matrix into compressed sparse format by freeing extra memory 
        space in the buffer. 
        """
        self.thisPtr.makeCompressed()
        
    
    def reserve(self, int n): 
        """
        Reserve n nonzero entries and turns the matrix into uncompressed mode. 
        """
        self.thisPtr.reserve(n)
        
    def dot(self, csarray_float A): 
        if self.shape[1] != A.shape[0]: 
            raise ValueError("Cannot multiply matrices of shapes " + str(self.shape) + " and " + str(A.shape))
        
        cdef csarray_float result = csarray_float((self.shape[0], A.shape[1]))
        del result.thisPtr
        result.thisPtr = new SparseMatrixExt[float](self.thisPtr.dot(deref(A.thisPtr)))
        return result 
        
    def transpose(self): 
        """
        Find the transpose of this matrix. 
        """
        cdef csarray_float result = csarray_float((self.shape[1], self.shape[0]))
        del result.thisPtr
        result.thisPtr = new SparseMatrixExt[float](self.thisPtr.trans())
        return result 
   
    #def norm(self, ord="fro"): 
        """
        Return the norm of this array. Currently only the Frobenius norm is 
        supported. 
        """
        #return self.thisPtr.norm()
   
    def ones(self): 
        """
        Fill the array with ones. 
        """
        self.thisPtr.fill(1)
        
   
    shape = property(__getShape)
    size = property(__getSize)
    ndim = property(__getNDim)
    dtype = property(__getDType)
    

    
cdef class csarray_double:
    cdef SparseMatrixExt[double] *thisPtr     
    def __cinit__(self, shape):
        """
        Create a new column major dynamic array.
        """

        self.thisPtr = new SparseMatrixExt[double](shape[0], shape[1]) 
            
    def __dealloc__(self): 
        """
        Deallocate the SparseMatrixExt object.  
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
            for k, index in enumerate(inds):  
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
        
        cdef csarray_double result = csarray_double((rowInds.shape[0], colInds.shape[0]))     
        
        rowIndsC = numpy.ascontiguousarray(rowInds, dtype=numpy.int32) 
        colIndsC = numpy.ascontiguousarray(colInds, dtype=numpy.int32) 
        
        if rowInds.shape[0] != 0 and colInds.shape[0] != 0: 
            self.thisPtr.slice(&rowIndsC[0], rowIndsC.shape[0], &colIndsC[0], colIndsC.shape[0], result.thisPtr) 
        return result 
        
    def nonzero(self): 
        """
        Return a tuple of arrays corresponding to nonzero elements. 
        """
        cdef numpy.ndarray[long, ndim=1, mode="c"] rowInds = numpy.zeros(self.getnnz(), dtype=numpy.int64) 
        cdef numpy.ndarray[long, ndim=1, mode="c"] colInds = numpy.zeros(self.getnnz(), dtype=numpy.int64)  
        
        if self.getnnz() != 0:
            self.thisPtr.nonZeroInds(&rowInds[0], &colInds[0])
        
        return (rowInds, colInds)
                    
    def __setitem__(self, inds, val):
        """
        Set elements of the array. If i,j = inds are integers then the corresponding 
        value in the array is set. 
        """
        i, j = inds 
        
        if type(i) == numpy.ndarray and type(j) == numpy.ndarray: 
            self.put(val, i, j)
        else:
            i = int(i) 
            j = int(j)
            if i < 0 or i>=self.thisPtr.rows(): 
                raise ValueError("Invalid row index " + str(i)) 
            if j < 0 or j>=self.thisPtr.cols(): 
                raise ValueError("Invalid col index " + str(j))        
            
            self.thisPtr.insertVal(i, j, val) 

    
    def put(self, val, numpy.ndarray[numpy.int_t, ndim=1] rowInds not None, numpy.ndarray[numpy.int_t, ndim=1] colInds not None): 
        """
        Select rowInds and colInds
        """
        cdef unsigned int ix 
        self.reserve(len(rowInds))
        
        if type(val) == numpy.ndarray: 
            for ix in range(len(rowInds)): 
                self.thisPtr.insertVal(rowInds[ix], colInds[ix], val[ix])
        else:
            for ix in range(len(rowInds)): 
                self.thisPtr.insertVal(rowInds[ix], colInds[ix], val)
            

    def sum(self, axis=None): 
        """
        Sum all of the elements in this array. If one specifies an axis 
        then we sum along the axis. 
        """
        cdef numpy.ndarray[double, ndim=1, mode="c"] result    
        cdef numpy.ndarray[long, ndim=1, mode="c"] rowInds
        cdef numpy.ndarray[long, ndim=1, mode="c"] colInds
        cdef unsigned int i
        
        if axis==None: 
            """
            scalarResult = 0 
            (rowInds, colInds) = self.nonzero()
            
            for i in range(rowInds.shape[0]): 
                scalarResult += self.thisPtr.coeff(rowInds[i], colInds[i])  
            
            return scalarResult
            """
            return self.thisPtr.sumValues()
            #There seems to be a very temporamental problem with thisPtr.sum()
            #return self.thisPtr.sum()
        elif axis==0: 
            result = numpy.zeros(self.shape[1], dtype=numpy.float) 
            (rowInds, colInds) = self.nonzero()
            
            for i in range(rowInds.shape[0]): 
                result[colInds[i]] += self.thisPtr.coeff(rowInds[i], colInds[i])   
        elif axis==1: 
            result = numpy.zeros(self.shape[0], dtype=numpy.float) 
            (rowInds, colInds) = self.nonzero()
            
            for i in range(rowInds.shape[0]): 
                result[rowInds[i]] += self.thisPtr.coeff(rowInds[i], colInds[i])  
        else:
            raise ValueError("Invalid axis: " + str(axis))
            
        return result 
                
        
    def mean(self, axis=None): 
        """
        Find the mean value of this array. 
        """
        if self.thisPtr.size() != 0:
            if axis ==None: 
                return self.sum()/float(self.thisPtr.size())
            elif axis == 0: 
                return self.sum(0)/float(self.shape[0])
            elif axis == 1: 
                return self.sum(1)/float(self.shape[1])
        else: 
            return float("nan")
     
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
        
    def trace(self): 
        """
        Returns the trace of the array which is simply the sum of the diagonal 
        entries. 
        """
        return self.diag().sum()
         
    def __mul__(self, double x):
        """
        Return a new array multiplied by a scalar value x. 
        """
        cdef csarray_double result = self.copy() 
        result.thisPtr.scalarMultiply(x)
        return result 
        
    def copy(self): 
        """
        Return a copied version of this array. 
        """
        cdef csarray_double result = csarray_double(self.shape)
        del result.thisPtr
        result.thisPtr = new SparseMatrixExt[double](deref(self.thisPtr))
        return result 
        
    def toarray(self): 
        """
        Convert this sparse matrix into a numpy array. 
        """
        cdef numpy.ndarray[double, ndim=2, mode="c"] result = numpy.zeros(self.shape, numpy.float)
        cdef numpy.ndarray[long, ndim=1, mode="c"] rowInds
        cdef numpy.ndarray[long, ndim=1, mode="c"] colInds
        cdef unsigned int i
        
        (rowInds, colInds) = self.nonzero()
            
        for i in range(rowInds.shape[0]): 
            result[rowInds[i], colInds[i]] += self.thisPtr.coeff(rowInds[i], colInds[i])   
            
        return result 
        
        
    def min(self): 
        """
        Find the minimum element of this array. 
        """
        cdef numpy.ndarray[long, ndim=1, mode="c"] rowInds
        cdef numpy.ndarray[long, ndim=1, mode="c"] colInds
        cdef unsigned int i
        cdef double minVal 
        
        if self.size == 0: 
            return float("nan")
        elif self.getnnz() != self.size: 
            minVal = 0 
        
        (rowInds, colInds) = self.nonzero()
            
        for i in range(rowInds.shape[0]): 
            if i == 0: 
                minVal = self.thisPtr.coeff(rowInds[i], colInds[i])
            
            if self.thisPtr.coeff(rowInds[i], colInds[i]) < minVal: 
                minVal = self.thisPtr.coeff(rowInds[i], colInds[i])
            
        return minVal 
        
    def max(self): 
        """
        Find the maximum element of this array. 
        """
        cdef numpy.ndarray[long, ndim=1, mode="c"] rowInds
        cdef numpy.ndarray[long, ndim=1, mode="c"] colInds
        cdef unsigned int i
        cdef double maxVal
        
        if self.size == 0: 
            return float("nan")
        elif self.getnnz() != self.size: 
            maxVal = 0 
        
        (rowInds, colInds) = self.nonzero()
            
        for i in range(rowInds.shape[0]): 
            if i == 0: 
                maxVal = self.thisPtr.coeff(rowInds[i], colInds[i])            
            
            if self.thisPtr.coeff(rowInds[i], colInds[i]) > maxVal: 
                maxVal = self.thisPtr.coeff(rowInds[i], colInds[i])
            
        return maxVal 
        
    def var(self): 
        """
        Return the variance of the elements of this array. 
        """
        cdef double mean = self.mean() 
        cdef numpy.ndarray[long, ndim=1, mode="c"] rowInds
        cdef numpy.ndarray[long, ndim=1, mode="c"] colInds
        cdef unsigned int i
        cdef double result = 0
        
        if self.size == 0: 
            return float("nan")
        
        (rowInds, colInds) = self.nonzero()
            
        for i in range(rowInds.shape[0]): 
            result += (self.thisPtr.coeff(rowInds[i], colInds[i]) - mean)**2
        
        result += (self.size - self.getnnz())*mean**2
        result /= float(self.size)
        
        return result 
    
    def std(self): 
        """
        Return the standard deviation of the array elements. 
        """
        return numpy.sqrt(self.var())
        
    def __abs__(self): 
        """
        Return a matrix whose elements are the absolute values of this array. 
        """
        cdef csarray_double result = csarray_double((self.shape[1], self.shape[0]))
        del result.thisPtr
        result.thisPtr = new SparseMatrixExt[double](self.thisPtr.abs())
        return result 
    
    def __neg__(self): 
        """
        Return the negation of this array. 
        """
        cdef csarray_double result = csarray_double((self.shape[1], self.shape[0]))
        del result.thisPtr
        result.thisPtr = new SparseMatrixExt[double](self.thisPtr.negate())
        return result 
    

    def __add__(csarray_double self, csarray_double A): 
        """
        Add two matrices together. 
        """
        if self.shape != A.shape: 
            raise ValueError("Cannot add matrices of shapes " + str(self.shape) + " and " + str(A.shape))
        
        cdef csarray_double result = csarray_double((self.shape[0], A.shape[1]))
        del result.thisPtr
        result.thisPtr = new SparseMatrixExt[double](self.thisPtr.add(deref(A.thisPtr)))
        return result     
        
    def __sub__(csarray_double self, csarray_double A): 
        """
        Subtract one matrix from another.  
        """
        if self.shape != A.shape: 
            raise ValueError("Cannot subtract matrices of shapes " + str(self.shape) + " and " + str(A.shape))
        
        cdef csarray_double result = csarray_double((self.shape[0], A.shape[1]))
        del result.thisPtr
        result.thisPtr = new SparseMatrixExt[double](self.thisPtr.subtract(deref(A.thisPtr)))
        return result    
     
    def hadamard(self, csarray_double A): 
        """
        Find the element-wise matrix (hadamard) product. 
        """
        if self.shape != A.shape: 
            raise ValueError("Cannot elementwise multiply matrices of shapes " + str(self.shape) + " and " + str(A.shape))
        
        cdef csarray_double result = csarray_double((self.shape[0], A.shape[1]))
        del result.thisPtr
        result.thisPtr = new SparseMatrixExt[double](self.thisPtr.hadamard(deref(A.thisPtr)))
        return result 

    def compress(self): 
        """
        Turn this matrix into compressed sparse format by freeing extra memory 
        space in the buffer. 
        """
        self.thisPtr.makeCompressed()
        
    
    def reserve(self, int n): 
        """
        Reserve n nonzero entries and turns the matrix into uncompressed mode. 
        """
        self.thisPtr.reserve(n)
        
    def dot(self, csarray_double A): 
        if self.shape[1] != A.shape[0]: 
            raise ValueError("Cannot multiply matrices of shapes " + str(self.shape) + " and " + str(A.shape))
        
        cdef csarray_double result = csarray_double((self.shape[0], A.shape[1]))
        del result.thisPtr
        result.thisPtr = new SparseMatrixExt[double](self.thisPtr.dot(deref(A.thisPtr)))
        return result 
        
    def transpose(self): 
        """
        Find the transpose of this matrix. 
        """
        cdef csarray_double result = csarray_double((self.shape[1], self.shape[0]))
        del result.thisPtr
        result.thisPtr = new SparseMatrixExt[double](self.thisPtr.trans())
        return result 
   
    #def norm(self, ord="fro"): 
        """
        Return the norm of this array. Currently only the Frobenius norm is 
        supported. 
        """
        #return self.thisPtr.norm()
   
    def ones(self): 
        """
        Fill the array with ones. 
        """
        self.thisPtr.fill(1)
        
   
    shape = property(__getShape)
    size = property(__getSize)
    ndim = property(__getNDim)
    dtype = property(__getDType)
    

    
