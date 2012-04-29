from cython.operator cimport dereference as deref, preincrement as inc #dereference and increment operators
import numpy 
cimport numpy 

cdef extern from "mapped_matrix_ext.h": 
   cdef cppclass mapped_matrix_ext[T]:  
      mapped_matrix_ext()
      mapped_matrix_ext(int, int, int)
      int size1()
      int size2() 
      T get_item(int, int)
      void set_item(int, int, T) 
      mapped_matrix_ext[T]& add(mapped_matrix_ext[T]&)
      mapped_matrix_ext[T]& minus(mapped_matrix_ext[T]&)
      mapped_matrix_ext[T]& multiply(T)


      
#TODO: Allow for different dtypes 
#Sum method 
#Slicing 
#Optimise 
#Operators: add, multiply, divide, minus
#Other: dot product, max, min, put, take, trace, transpose 
#mean, std, var 

cdef mapped_matrix_ext[double] *A = new mapped_matrix_ext[double](3, 3, 5) 
cdef mapped_matrix_ext[double] *B = new mapped_matrix_ext[double](3, 3, 5)    
cdef mapped_matrix_ext[double] C
cdef mapped_matrix_ext[double] *D
C = A.add(deref(B))
D = &C

cdef class map_array:
    cdef mapped_matrix_ext[double] *thisptr     
    def __cinit__(self, shape, nnz):
        self.thisptr = new mapped_matrix_ext[double](shape[0], shape[1], nnz)
    def __dealloc__(self):
        del self.thisptr
    def getNDim(self): 
        return 2 
    def getShape(self):
        return (self.thisptr.size1(), self.thisptr.size2())
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

    shape = property(getShape)
    ndim = property(getNDim)


