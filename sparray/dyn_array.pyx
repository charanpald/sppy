from cython.operator cimport dereference as deref, preincrement as inc 
import numpy 
cimport numpy 


cdef extern from "include/DynamicSparseMatrixExt.h": 
   cdef cppclass DynamicSparseMatrixExt[T]:  
      DynamicSparseMatrixExt()
      DynamicSparseMatrixExt(int, int)
      int rows()
      int cols() 
      int size() 
      void insertVal(int, int, T)
      int nonZeros()
      T coeff(int, int)
      T sum()
      DynamicSparseMatrixExt[T]* slice(int*, int, int*, int) 

cdef class dyn_array:
    cdef DynamicSparseMatrixExt[double] *thisPtr     
    def __cinit__(self, shape, dtype=numpy.float):
        if dtype==numpy.float: 
            self.thisPtr = new DynamicSparseMatrixExt[double](shape[0], shape[1])
    def __dealloc__(self):
        del self.thisPtr
    def getNDim(self): 
        return 2 
    def getShape(self):
        return (self.thisPtr.rows(), self.thisPtr.cols())
    def getSize(self): 
        return self.thisPtr.size()    
    def getnnz(self): 
        return self.thisPtr.nonZeros()
    def __getitem__(self, inds):
        i, j = inds 
        if type(i) == int and type(j) == int: 
            if i < 0 or i>=self.thisPtr.rows(): 
                raise ValueError("Invalid row index " + str(i)) 
            if j < 0 or j>=self.thisPtr.cols(): 
                raise ValueError("Invalid col index " + str(j))      
            return self.thisPtr.coeff(i, j)
        elif type(i) == numpy.ndarray and type(j) == numpy.ndarray: 
            result = numpy.zeros(i.shape[0])
            for ix in range(i.shape[0]): 
                    result[ix] = self.thisPtr.coeff(i[ix], j[ix])
            return result
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
                        stop = index.start  
                    indArr = numpy.arange(start, stop)
                    indList.append(indArr)
            
            result = dyn_array((indList[0].shape[0], indList[1].shape[0]))
            for ind1 in range(indList[0].shape[0]): 
                for ind2 in range(indList[1].shape[0]): 
                    result[ind1, ind2] = self.thisPtr.coeff(indList[0][ind1], indList[1][ind2])
                    
            return result
                        
    def __setitem__(self, inds, val):
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
    
    def put(self, double val, numpy.ndarray[numpy.int32_t, ndim=1] rowInds not None , numpy.ndarray[numpy.int32_t, ndim=1] colInds not None): 
        cdef unsigned int ix 
        for ix in range(len(rowInds)): 
            self.thisPtr.insertVal(rowInds[ix], colInds[ix], val)

    def sum(self): 
        return self.thisPtr.sum()
     
    def __str__(self): 
        outputStr = "dyn_array " + str(self.shape) + " " + str(self.getnnz()) 
        return outputStr 
    
    shape = property(getShape)
    ndim = property(getNDim)

    
