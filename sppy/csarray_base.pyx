# cython: profile=True
# cython: boundscheck=False
from cython.operator cimport dereference as deref, preincrement as inc 
from libcpp.vector cimport vector
from sppy.dtype import dataTypeDict
from cython.parallel import prange 
import numpy 
import multiprocessing 
cimport numpy
import cython 
numpy.import_array()
from libc.math cimport sqrt 

cdef template[DataType, StorageType] class csarray:
    def __cinit__(self, shape):
        """
        Create a new column or row major dynamic array.
        """
        self.thisPtr = new SparseMatrixExt[DataType, StorageType](shape[0], shape[1]) 
  
    def __abs__(self): 
        """
        Return a matrix whose elements are the absolute values of this array. 
        """
        cdef csarray[DataType, StorageType] result = csarray[DataType, StorageType]((self.shape[0], self.shape[1]))
        del result.thisPtr
        result.thisPtr = new SparseMatrixExt[DataType, StorageType](self.thisPtr.abs())
        return result 
      
    def __adArraySlice(self, numpy.ndarray[numpy.int_t, ndim=1, mode="c"] rowInds, numpy.ndarray[numpy.int_t, ndim=1, mode="c"] colInds): 
        """
        Array slicing where one passes two arrays of the same length and elements are picked 
        according to self[rowInds[i], colInds[i]). 
        """
        cdef int ix 
        cdef numpy.ndarray[DataType, ndim=1, mode="c"] result = numpy.zeros(rowInds.shape[0], self.dtype())
        
        if (rowInds >= self.shape[0]).any() or (colInds >= self.shape[1]).any(): 
            raise ValueError("Indices out of range")
        
        for ix in range(rowInds.shape[0]): 
                result[ix] = self.thisPtr.coeff(rowInds[ix], colInds[ix])
        return result
  
    def __add__(csarray[DataType, StorageType] self, csarray[DataType, StorageType] A): 
        """
        Add two matrices together. 
        """
        if self.shape != A.shape: 
            raise ValueError("Cannot add matrices of shapes " + str(self.shape) + " and " + str(A.shape))
        
        cdef csarray[DataType, StorageType] result = csarray[DataType, StorageType]((self.shape[0], A.shape[1]))
        del result.thisPtr
        result.thisPtr = new SparseMatrixExt[DataType, StorageType](self.thisPtr.add(deref(A.thisPtr)))
        return result   
          
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

    def __getStorage(self): 
        return "StorageType"     
    
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
        
        if isinstance(i, list): 
            i = numpy.array(i, numpy.int)
        if isinstance(j, list): 
            j = numpy.array(j, numpy.int)
        
        inds = i,j
        
        if type(i) == numpy.ndarray and type(j) == numpy.ndarray: 
            return self.__adArraySlice(numpy.ascontiguousarray(i, dtype=numpy.int) , numpy.ascontiguousarray(j, dtype=numpy.int) )
        elif (type(i) == numpy.ndarray or isinstance(i, slice)) and (isinstance(j, slice) or type(j) == numpy.ndarray):
            indList = []            
            for k, index in enumerate(inds):  
                if isinstance(index, numpy.ndarray): 
                    indList.append(index) 
                elif isinstance(index, slice): 
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
        elif (isinstance(i, int) and isinstance(j, slice)) or (isinstance(i, slice) and isinstance(j, int)):                
            if isinstance(i, int): 
                inds = self.rowInds(i)
                slc = j 
                if slc.start == None: 
                    start = 0
                else: 
                    start = slc.start 
                if slc.stop == None: 
                    stop = self.shape[1]
                else: 
                    stop = slc.stop 
                    
                result = csarray[DataType, StorageType]((self.shape[1], 1))   
                
                for ind in inds: 
                    if start <= ind < stop: 
                        result[ind, 0] = self.thisPtr.coeff(i, ind)                 
            elif isinstance(j, int): 
                inds = self.colInds(j)
                slc = i 
                if slc.start == None: 
                    start = 0
                else: 
                    start = slc.start 
                if slc.stop == None: 
                    stop = self.shape[1]
                else: 
                    stop = slc.stop 
                    
                result = csarray[DataType, StorageType]((self.shape[0], 1))   
                
                for ind in inds: 
                    if start <= ind < stop: 
                        result[ind, 0] = self.thisPtr.coeff(ind, j)  
            return result 
        else:     
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

    def __mul__(self, double x):
        """
        Return a new array multiplied by a scalar value x. 
        """
        cdef csarray[DataType, StorageType] result = self.copy() 
        result.thisPtr.scalarMultiply(x)
        return result 

    def __neg__(self): 
        """
        Return the negation of this array. 
        """
        cdef csarray[DataType, StorageType] result = csarray[DataType, StorageType]((self.shape[1], self.shape[0]))
        del result.thisPtr
        result.thisPtr = new SparseMatrixExt[DataType, StorageType](self.thisPtr.negate())
        return result       

    def __putUsingTriplets(self, numpy.ndarray[DataType, ndim=1, mode="c"] vals not None, numpy.ndarray[int, ndim=1] rowInds not None, numpy.ndarray[int, ndim=1] colInds not None): 
        """
        The row indices must be sorted in descending order if in column major order. 
        """
        cdef int n = rowInds.shape[0]         
        self.thisPtr.putUsingTriplets(&rowInds[0], &colInds[0], &vals[0], n) 
    
        
    def __putUsingTriplets2(self, DataType val, numpy.ndarray[int, ndim=1] rowInds not None, numpy.ndarray[int, ndim=1] colInds not None): 
        """
        The row indices must be sorted in ascending order if in column major order.  
        """
        cdef int n = rowInds.shape[0]        
        self.thisPtr.putUsingTriplets2(&rowInds[0], &colInds[0], val, n)  

    def __setitem__(self, inds, val):
        """
        Set elements of the array. If i,j = inds are integers then the corresponding 
        value in the array is set. 
        """
        i, j = inds 
        
        if type(i) == numpy.ndarray and type(j) == numpy.ndarray: 
            if i.dtype != numpy.dtype("i"): 
                i = numpy.array(i, numpy.int32)
            if j.dtype != numpy.dtype("i"): 
                j = numpy.array(j, numpy.int32)            
            
            self.put(val, i, j)
        else:
            i = int(i) 
            j = int(j)
            if i < 0 or i>=self.thisPtr.rows(): 
                raise ValueError("Invalid row index " + str(i)) 
            if j < 0 or j>=self.thisPtr.cols(): 
                raise ValueError("Invalid col index " + str(j))        
            
            self.thisPtr.insertVal(i, j, val) 

    def __sub__(csarray[DataType, StorageType] self, csarray[DataType, StorageType] A): 
        """
        Subtract one matrix from another.  
        """
        if self.shape != A.shape: 
            raise ValueError("Cannot subtract matrices of shapes " + str(self.shape) + " and " + str(A.shape))
        
        cdef csarray[DataType, StorageType] result = csarray[DataType, StorageType]((self.shape[0], A.shape[1]))
        del result.thisPtr
        result.thisPtr = new SparseMatrixExt[DataType, StorageType](self.thisPtr.subtract(deref(A.thisPtr)))
        return result    

    def biCGSTAB(self, numpy.ndarray[DataType, ndim=1, mode="c"] v, int maxIter=1000, double tol=10**-6):         
        cdef int outputCode = 0  
        cdef numpy.ndarray[DataType, ndim=1, mode="c"] result = numpy.zeros(v.shape[0])

        if self.shape[0] != self.shape[1]: 
            raise ValueError("Can only operate on square matrices")
        if v.shape[0] != self.shape[0]: 
            raise ValueError("Length of v must match columns of A")
        
        maxIterations = min(maxIter, v.shape[0])
        
        outputCode = self.thisPtr.biCGSTAB(&v[0], v.shape[0], &result[0], maxIterations, tol)
        return result, outputCode 


    def colInds(self, int i): 
        cdef unsigned int j
        cdef vector[int] vect = self.thisPtr.getIndsCol(i)
        cdef numpy.ndarray[int, ndim=1, mode="c"] inds = numpy.zeros(vect.size(), numpy.int32)
        
        #Must be a better way to do this 
        for j in range(vect.size()): 
            inds[j] = vect[j]

        return inds  

    def compress(self): 
        """
        Turn this matrix into compressed sparse format by freeing extra memory 
        space in the buffer. 
        """
        self.thisPtr.makeCompressed()

    def copy(self): 
        """
        Return a copied version of this array. 
        """
        cdef csarray[DataType, StorageType] result = csarray[DataType, StorageType](self.shape)
        del result.thisPtr
        result.thisPtr = new SparseMatrixExt[DataType, StorageType](deref(self.thisPtr))
        return result 
    
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
    
    def dotCsarray2d(self, csarray[DataType, StorageType] A): 
        if self.shape[1] != A.shape[0]: 
            raise ValueError("Cannot multiply matrices of shapes " + str(self.shape) + " and " + str(A.shape))
        
        cdef csarray[DataType, StorageType] result = csarray[DataType, StorageType]((self.shape[0], A.shape[1]))
        del result.thisPtr
        result.thisPtr = new SparseMatrixExt[DataType, StorageType](self.thisPtr.dot(deref(A.thisPtr)))
        return result 

    def dotNumpy1d(self, numpy.ndarray[double, ndim=1, mode="c"] v): 
        """
        Take this array and multiply it with a numpy array. 
        """
        if self.shape[1] != v.shape[0]: 
            raise ValueError("Cannot multiply matrices of shapes " + str(self.shape) + " and " + str(v.shape[0], v.shape[1]))
        
        cdef numpy.ndarray[double, ndim=1, mode="c"] result = numpy.zeros(self.shape[0])
        self.thisPtr.dot1d(&v[0], &result[0])
            
        return result    
    
    def dotNumpy2d(self, numpy.ndarray[double, ndim=2, mode="c"] A): 
        """
        Take this array and multiply it with a numpy array. 
        """
        if self.shape[1] != A.shape[0]: 
            raise ValueError("Cannot multiply matrices of shapes " + str(self.shape) + " and " + str(A.shape[0], A.shape[1]))
        
        cdef numpy.ndarray[double, ndim=2, mode="c"] result = numpy.zeros((self.shape[0], A.shape[1]))
        self.thisPtr.dot2d(&A[0, 0], A.shape[1], &result[0, 0])
            
        return result     

    def dtype(self): 
        """
        Return the dtype of the current object. 
        """
        return dataTypeDict["DataType"]    
    
    def getnnz(self): 
        """
        Return the number of non-zero elements in the array 
        """
        return self.thisPtr.nonZeros()    
   
    def hadamard(self, csarray[DataType, StorageType] A): 
        """
        Find the element-wise matrix (hadamard) product. 
        """
        if self.shape != A.shape: 
            raise ValueError("Cannot elementwise multiply matrices of shapes " + str(self.shape) + " and " + str(A.shape))
        
        cdef csarray[DataType, StorageType] result = csarray[DataType, StorageType]((self.shape[0], A.shape[1]))
        del result.thisPtr
        result.thisPtr = new SparseMatrixExt[DataType, StorageType](self.thisPtr.hadamard(deref(A.thisPtr)))
        return result    
   
    def max(self): 
        """
        Find the maximum element of this array. 
        """
        cdef numpy.ndarray[int, ndim=1, mode="c"] rowInds
        cdef numpy.ndarray[int, ndim=1, mode="c"] colInds
        cdef unsigned int i
        cdef DataType maxVal
        
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

    def mean(self, axis=None): 
        """
        Find the mean value of this array. 
        
        :param axis: The axis of the array to compute the mean. 
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
        
    def min(self): 
        """
        Find the minimum element of this array. 
        """
        cdef numpy.ndarray[int, ndim=1, mode="c"] rowInds
        cdef numpy.ndarray[int, ndim=1, mode="c"] colInds
        cdef unsigned int i
        cdef DataType minVal 
        
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
        
    def nonzero(self): 
        """
        Return a tuple of arrays corresponding to nonzero elements. 
        """
        cdef numpy.ndarray[int, ndim=1, mode="c"] rowInds = numpy.zeros(self.getnnz(), dtype=numpy.int32) 
        cdef numpy.ndarray[int, ndim=1, mode="c"] colInds = numpy.zeros(self.getnnz(), dtype=numpy.int32)
        
        if self.getnnz() != 0:
            self.thisPtr.nonZeroInds(&rowInds[0], &colInds[0])
        
        return (rowInds, colInds)

    def norm(self): 
        """
        Return the Frobenius norm of this matrix. 
        """
        
        return self.thisPtr.norm()

    def ones(self): 
        """
        Fill the array with ones. 
        """
        self.thisPtr.fill(1)

    def pdot1d(self, numpy.ndarray[double, ndim=1, mode="c"] v not None): 
        """
        Take this array and multiply it with a numpy array using multithreading. 
        """
        if self.shape[1] != v.shape[0]: 
            raise ValueError("Cannot multiply using shapes " + str(self.shape) + " and " + str(v.shape[0]))
            
        cdef numpy.ndarray[double, ndim=1, mode="c"] result = numpy.zeros(self.shape[0])  
        cdef int numCpus = multiprocessing.cpu_count()                          
        cdef int numJobs = numCpus
        cdef int i  
        cdef numpy.ndarray[numpy.int_t, ndim=1] rowInds = numpy.array(numpy.linspace(0, self.shape[0], numJobs+1), numpy.int)
        
        for i in prange(numJobs, nogil=True, num_threads=numCpus, schedule="static"):
            self.thisPtr.dotSub1d(&v[0], rowInds[i], rowInds[i+1], &result[0])
            
        return result          

    def pdot2d(self, numpy.ndarray[double, ndim=2, mode="c"] A not None): 
        """
        Take this array and multiply it with a numpy array using multithreading. 
        """
        if self.shape[1] != A.shape[0]: 
            raise ValueError("Cannot multiply matrices of shapes " + str(self.shape) + " and " + str(A.shape[0], A.shape[1]))
        if self.storage != "rowMajor": 
            raise ValueError("Only thread-safe on row major matrices")
            
        cdef numpy.ndarray[double, ndim=2, mode="c"] result = numpy.zeros((self.shape[0], A.shape[1]))  
        cdef int numCpus = multiprocessing.cpu_count()                          
        cdef int numJobs = numCpus
        cdef int i  
        cdef numpy.ndarray[numpy.int_t, ndim=1] rowInds = numpy.array(numpy.linspace(0, self.shape[0], numJobs+1), numpy.int)
        
        for i in prange(numJobs, nogil=True, num_threads=numCpus, schedule="static"):
            self.thisPtr.dotSub2d(&A[0, 0], result.shape[1],  rowInds[i], rowInds[i+1], &result[0, 0])
            
        return result     

    def put(self, val, numpy.ndarray[int, ndim=1] rowInds not None, numpy.ndarray[int, ndim=1] colInds not None, init=False): 
        """
        Put some values into this matrix. Notice, that this is faster if init=True and 
        the matrix has just been created. 
        """  
        cdef unsigned int ix         
        
        if init: 
            if type(val) == numpy.ndarray: 
                self.__putUsingTriplets(val, rowInds, colInds)
            else:
                self.__putUsingTriplets2(val, rowInds, colInds)
        else: 
            self.reserve(int(len(rowInds)*1.2))
            
            if type(val) == numpy.ndarray:
                for ix in range(len(rowInds)):
                    self.thisPtr.insertVal(rowInds[ix], colInds[ix], val[ix])
            else:
                for ix in range(len(rowInds)):
                    self.thisPtr.insertVal(rowInds[ix], colInds[ix], val)   

    def reserve(self, int n): 
        """
        Reserve n nonzero entries and turns the matrix into uncompressed mode. 
        """
        self.thisPtr.reserve(n)

    def rowInds(self, int i):
        """
        Returns the non zero indices for the ith row. 
        """
        cdef unsigned int j
        cdef vector[int] vect = self.thisPtr.getIndsRow(i)
        cdef numpy.ndarray[int, ndim=1, mode="c"] inds = numpy.zeros(vect.size(), numpy.int32)
        
        #Must be a better way to do this 
        for j in range(vect.size()): 
            inds[j] = vect[j]

        return inds  

    def setZero(self):
        self.thisPtr.setZero()

    def subArray(self, numpy.ndarray[numpy.int_t, ndim=1, mode="c"] rowInds, numpy.ndarray[numpy.int_t, ndim=1, mode="c"] colInds): 
        """
        Explicitly perform an array slice to return a submatrix with the given
        indices. Only works with ascending ordered indices. This is similar 
        to using numpy.ix_. 
        """
        cdef numpy.ndarray[int, ndim=1, mode="c"] rowIndsC 
        cdef numpy.ndarray[int, ndim=1, mode="c"] colIndsC 
        
        cdef csarray[DataType, StorageType] result = csarray[DataType, StorageType]((rowInds.shape[0], colInds.shape[0]))     
        
        rowIndsC = numpy.ascontiguousarray(rowInds, dtype=numpy.int32) 
        colIndsC = numpy.ascontiguousarray(colInds, dtype=numpy.int32) 
        
        if rowInds.shape[0] != 0 and colInds.shape[0] != 0: 
            self.thisPtr.slice(&rowIndsC[0], rowIndsC.shape[0], &colIndsC[0], colIndsC.shape[0], result.thisPtr) 
        return result 

    def std(self): 
        """
        Return the standard deviation of the array elements. 
        """
        return numpy.sqrt(self.var())

    def sum(self, axis=None): 
        """
        Sum all of the elements in this array. If one specifies an axis 
        then we sum along the axis. 
        """
        cdef numpy.ndarray[double, ndim=1, mode="c"] result    
        cdef numpy.ndarray[int, ndim=1, mode="c"] rowInds
        cdef numpy.ndarray[int, ndim=1, mode="c"] colInds
        cdef unsigned int i
        
        if axis==None: 
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
  
    def toarray(self): 
        """
        Convert this sparse matrix into a numpy array. 
        """
        cdef numpy.ndarray[double, ndim=2, mode="c"] result = numpy.zeros(self.shape, numpy.float)
        cdef numpy.ndarray[int, ndim=1, mode="c"] rowInds
        cdef numpy.ndarray[int, ndim=1, mode="c"] colInds
        cdef unsigned int i
        
        (rowInds, colInds) = self.nonzero()
            
        for i in range(rowInds.shape[0]): 
            result[rowInds[i], colInds[i]] += self.thisPtr.coeff(rowInds[i], colInds[i])   
            
        return result 
              
    def trace(self): 
        """
        Returns the trace of the array which is simply the sum of the diagonal 
        entries. 
        """
        return self.diag().sum()

    def transpose(self): 
        """
        Find the transpose of this matrix. 
        """
        cdef csarray[DataType, StorageType] result = csarray[DataType, StorageType]((self.shape[1], self.shape[0]))
        del result.thisPtr
        result.thisPtr = new SparseMatrixExt[DataType, StorageType](self.thisPtr.trans())
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
        cdef numpy.ndarray[int, ndim=1, mode="c"] rowInds
        cdef numpy.ndarray[int, ndim=1, mode="c"] colInds
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
   
    shape = property(__getShape)
    size = property(__getSize)
    ndim = property(__getNDim)
    storage = property(__getStorage)
    nnz = property(getnnz)
    

    
