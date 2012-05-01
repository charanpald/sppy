from cython.operator cimport dereference as deref, preincrement as inc 
import numpy 
cimport numpy 


cdef extern from "SparseMatrixExt.h": 
   cdef cppclass SparseMatrixExt[T]:  
      SparseMatrixExt()
      SparseMatrixExt(int, int)
      int rows()
      int cols() 
      int size() 
      void insert(int, int)

cdef class cmpr_array:
    cdef SparseMatrixExt[double] *thisptr     
    def __cinit__(self, shape):
        self.thisptr = new SparseMatrixExt[double](shape[0], shape[1])
    def __dealloc__(self):
        del self.thisptr
    def getNDim(self): 
        return 2 
    def getShape(self):
        return (self.thisptr.rows(), self.thisptr.cols())
        
    def getSize(self): 
        return self.thisptr.size()
        
    """
    def __getitem__(self, inds):
        i, j = inds 
        if type(i) == int and type(j) == int: 
            if i < 0 or i>=self.thisptr.size1(): 
                raise ValueError("Invalid row index " + str(i)) 
            if j < 0 or j>=self.thisptr.size2(): 
                raise ValueError("Invalid col index " + str(j))      
            return self.thisptr.get_item(i, j)
        elif type(i) == numpy.ndarray and type(j) == numpy.ndarray: 
            result = numpy.zeros(i.shape[0])
            for ind in range(i.shape[0]): 
                    result[ind] = self.thisptr.get_item(i[ind], j[ind])
            return result
        elif type(i) == numpy.ndarray and type(j) == slice:
            if j.start == None: 
                start = 0
            else: 
                start = j.start 
            if j.stop == None: 
                stop = self.shape[0]
            else:
                stop = j.start  
            sliceSize = stop - start 
            result = map_array((i.shape[0], sliceSize), 10)
            for ind1 in range(i.shape[0]): 
                for ind2 in range(start, stop): 
                    result[ind1, ind2] = self.thisptr.get_item(i[ind1], ind2)
            return result
                        

    def __setitem__(self, inds, val):
        cdef unsigned int i, j 
        i, j = inds 
        if i < 0 or i>=self.thisptr.size1(): 
            raise ValueError("Invalid row index " + str(i)) 
        if j < 0 or j>=self.thisptr.size2(): 
            raise ValueError("Invalid col index " + str(j))      
            
        self.thisptr.set_item(i, j, val)
    
    def add(self, map_array A): 
        C = map_array(self.shape, 10)
        C.thisptr = &self.thisptr.add(deref(A.thisptr))
        return  C
        
    def minus(self, map_array A): 
        C = map_array(self.shape, 10)
        C.thisptr = &self.thisptr.minus(deref(A.thisptr))
        return  C
    
     
    def __str__(self): 
        outputStr = "map_array([" 
        for i in range(self.shape[0]): 
            for j in range(self.shape[1]):
                if j==0 and i!=0:     
                    outputStr += " "*10 + "[" + str(self[i, j]) + ", "    
                elif j!=self.shape[1]-1:  
                    outputStr += str(self[i, j]) + ", "
                else: 
                    outputStr += str(self[i, j])
            if i!=self.shape[0]-1: 
                outputStr += "]\n"
            else: 
                outputStr += "])\n"    
        return outputStr 
    """

    shape = property(getShape)
    ndim = property(getNDim)

    
