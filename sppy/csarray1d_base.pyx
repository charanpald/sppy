# cython: profile=False
from cython.operator cimport dereference as deref, preincrement as inc 
import numpy 
cimport numpy
import cython 
from sppy.dtype import dataTypeDict
from libcpp.vector cimport vector
numpy.import_array()


      
cdef template[DataType] class csarray1d:  
    def __cinit__(self, shape):
        """
        Create a new dynamic array.
        """
        cdef int shapeVal         
        
        if type(shape) == tuple and len(shape) == 1: 
            shapeVal = shape[0]
        elif type(shape) == int: 
            shapeVal = shape
        else: 
            raise ValueError("Invalid input: " + str(shape))
            
        self.thisPtr = new SparseVectorExt[DataType](shapeVal) 
      
    def __abs__(self): 
        """
        Return a matrix whose elements are the absolute values of this array. 
        """
        cdef csarray1d[DataType] result = csarray1d[DataType](self.shape[0])
        del result.thisPtr
        result.thisPtr = new SparseVectorExt[DataType](self.thisPtr.abs())
        return result 
      
    def __add__(csarray1d[DataType] self, csarray1d[DataType] a): 
        """
        Add two matrices together. 
        """
        if self.shape != a.shape: 
            raise ValueError("Cannot add matrices of shapes " + str(self.shape) + " and " + str(a.shape))
        
        cdef int shapeVal = self.shape[0]
        cdef csarray1d[DataType] result = csarray1d[DataType](shapeVal)
        del result.thisPtr
        result.thisPtr = new SparseVectorExt[DataType](self.thisPtr.add(deref(a.thisPtr)))
        return result          
      
    def __dealloc__(self): 
        """
        Deallocate the SparseVectorExt object.  
        """
        del self.thisPtr

    def __getitem__(self, ind):
        """
        Get a value or set of values from the array (denoted A). Currently 3 types of parameters 
        are supported. If i is an integer then the corresponding element of the array 
        is returned. If i is an arrays of ints then we return the corresponding 
        values of A[i[k]] (note: i,j must be sorted in ascending order). If i
         is a slice e.g. a[1:5] then we return the submatrix corresponding to 
        the slice. 
        """        
        if type(ind) == tuple: 
            if len(ind)==1: 
                ind = ind[0]
            else: 
                raise ValueError("Invalid input " + str(ind))
        
        
        if type(ind) == numpy.ndarray or type(ind) == slice:
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
        
    def __mul__(self, double x):
        """
        Return a new array multiplied by a scalar value x. 
        """
        cdef csarray1d[DataType] result = self.copy() 
        result.thisPtr.scalarMultiply(x)
        return result 

    def __neg__(self): 
        """
        Return the negation of this array. 
        """
        cdef csarray1d[DataType] result = csarray1d[DataType](self.shape[0])
        del result.thisPtr
        result.thisPtr = new SparseVectorExt[DataType](self.thisPtr.negate())
        return result 

    def __setitem__(self, ind, val):
        """
        Set elements of the array. If i is integers then the corresponding 
        value in the array is set. 
        """
        
        if type(ind) == numpy.ndarray: 
            self.put(val, ind)
        elif type(ind) == int:
            ind = int(ind) 
            if ind < 0 or ind>=self.thisPtr.rows(): 
                raise ValueError("Invalid index " + str(ind))   
            self.thisPtr.insertVal(ind, val) 
        else:
            raise ValueError("Invalid index " + str(ind))  

    def __sub__(csarray1d[DataType] self, csarray1d[DataType] a): 
        """
        Subtract two matrices together. 
        """
        if self.shape != a.shape: 
            raise ValueError("Cannot subtract matrices of shapes " + str(self.shape) + " and " + str(a.shape))
        
        cdef int shapeVal = self.shape[0]
        cdef csarray1d[DataType] result = csarray1d[DataType](shapeVal)
        del result.thisPtr
        result.thisPtr = new SparseVectorExt[DataType](self.thisPtr.subtract(deref(a.thisPtr)))
        return result 

    def copy(self): 
        """
        Return a copied version of this array. 
        """
        cdef csarray1d[DataType] result = csarray1d[DataType](self.shape)
        del result.thisPtr
        result.thisPtr = new SparseVectorExt[DataType](deref(self.thisPtr))
        return result 
      
    def dotCsarray1d(self, csarray1d[DataType] a): 
        if self.shape != a.shape: 
            raise ValueError("Cannot compute dot product of matrices of shapes " + str(self.shape) + " and " + str(a.shape))
            
        return self.thisPtr.dot(deref(a.thisPtr))
         
    def dtype(self): 
        """
        Return the dtype of the current object. 
        """
        return dataTypeDict["DataType"]    

    def getnnz(self): 
        """
        Return the number of non-zero elements in the array. 
        """
        return self.thisPtr.nonZeros()

    def hadamard(self, csarray1d[DataType] a): 
        """
        Find the element-wise matrix (hadamard) product. 
        """
        if self.shape != a.shape: 
            raise ValueError("Cannot elementwise multiply matrices of shapes " + str(self.shape) + " and " + str(a.shape))
        
        cdef csarray1d[DataType] result = csarray1d[DataType](self.shape[0])
        del result.thisPtr
        result.thisPtr = new SparseVectorExt[DataType](self.thisPtr.hadamard(deref(a.thisPtr)))
        return result 

    def max(self): 
        """
        Find the maximum element of this array. 
        """
        cdef numpy.ndarray[long, ndim=1, mode="c"] inds
        cdef unsigned int i
        cdef DataType maxVal 
        
        if self.size == 0: 
            return float("nan")
        elif self.getnnz() != self.size: 
            maxVal = 0 
        
        (inds,) = self.nonzero()
            
        for i in range(inds.shape[0]): 
            if self.thisPtr.coeff(inds[i]) > maxVal: 
                maxVal = self.thisPtr.coeff(inds[i])
            
        return maxVal         

    def mean(self,): 
        """
        Find the mean value of this array. 
        """
        if self.thisPtr.size() != 0:
            return self.sum()/float(self.thisPtr.size())
        else: 
            return float("nan")

    def min(self): 
        """
        Find the minimum element of this array. 
        """
        cdef numpy.ndarray[long, ndim=1, mode="c"] inds
        cdef unsigned int i
        cdef DataType minVal 
        
        if self.size == 0: 
            return float("nan")
        elif self.getnnz() != self.size: 
            minVal = 0 
        
        (inds,) = self.nonzero()
            
        for i in range(inds.shape[0]): 
            if self.thisPtr.coeff(inds[i]) < minVal: 
                minVal = self.thisPtr.coeff(inds[i])
            
        return minVal 

    def nonzero(self): 
        """
        Return a tuple of arrays corresponding to nonzero elements. 
        """
        cdef numpy.ndarray[long, ndim=1, mode="c"] inds = numpy.zeros(self.getnnz(), dtype=numpy.int64) 
        
        if self.getnnz() != 0:
            self.thisPtr.nonZeroInds(&inds[0])
        
        return (inds, )

    def ones(self): 
        """
        Fill the array with ones. 
        """
        self.thisPtr.fill(1)

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
        
    def std(self): 
        """
        Return the standard deviation of the array elements. 
        """
        return numpy.sqrt(self.var())

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

    def sum(self): 
        """
        Sum all of the elements in this array. 
        """      
        return self.thisPtr.sumValues()
        
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
        
    def values(self): 
        """
        Return the values of this object according to the elements returned 
        using nonzero.  
        """

        cdef numpy.ndarray[DataType, ndim=1, mode="c"] vals = numpy.zeros(self.getnnz(), self.dtype()) 
        
        if self.getnnz() != 0:
            self.thisPtr.nonZeroVals(&vals[0])
        
        return vals           
        
    def var(self): 
        """
        Return the variance of the elements of this array. 
        """
        cdef double mean = self.mean() 
        cdef numpy.ndarray[long, ndim=1, mode="c"] inds
        cdef unsigned int i
        cdef double result = 0
        
        if self.size == 0: 
            return float("nan")
        
        (inds, ) = self.nonzero()
            
        for i in range(inds.shape[0]): 
            result += (self.thisPtr.coeff(inds[i]) - mean)**2
        
        result += (self.size - self.getnnz())*mean**2
        result /= float(self.size)
        
        return result 
    

    shape = property(__getShape)
    size = property(__getSize)
    ndim = property(__getNDim)        


