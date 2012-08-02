# cython: profile=False
from cython.operator cimport dereference as deref, preincrement as inc 
from sparray.csarray_sub import csarray_int, csarray_double 
import numpy 
cimport numpy
 
numpy.import_array()


class csarray: 
    def __init__(self, shape, dtype=numpy.float): 
        if dtype == numpy.float: 
            self._array = csarray_double(shape)
        elif dtype == numpy.int: 
            self._array = csarray_int(shape)
        else: 
            raise ValueError("Unknown dtype: " + str(dtype))
            
        self._dtype = dtype
            
    def __getattr__(self, name):
        try: 
            return getattr(self, name)
        except: 
            return getattr(self._array, name)

    def __getitem__(self, inds):
        return self._array.__getitem__(inds) 
        
    def __setitem__(self, inds, val):
        self._array.__setitem__(inds, val) 
        
    def __abs__(self): 
        return self._array.__abs__()
        
    def __neg__(self): 
        return self._array.__neg__()

    def __add__(self, A): 
        return self._array.__add__(A._array)
        
    def __sub__(self, A): 
        return self._array.__sub__(A._array)
        
    def hadamard(self, A): 
        return self._array.hadamard(A._array)
        
    def __mul__(self, x):
        newArray = self.copy() 
        newArray._array = newArray._array*x
        return newArray
        
    def copy(self): 
        newArray = csarray(self.shape, self.dtype)
        newArray._array = self._array.copy()
        return newArray 
     
    def __str__(self): 
        """
        Return a string representation of the non-zero elements of the array. 
        """
        outputStr = "csarray dtype:" + str(numpy.dtype(self.dtype)) + " shape:" + str(self.shape) + " non-zeros:" + str(self.getnnz()) + "\n"
        (rowInds, colInds) = self.nonzero()
        vals = self[rowInds, colInds]
        
        for i in range(self.getnnz()): 
            outputStr += "(" + str(rowInds[i]) + ", " + str(colInds[i]) + ")" + " " + str(vals[i]) 
            if i != self.getnnz()-1: 
                outputStr += "\n"
        
        return outputStr 
    
    def __getDType(self): 
        return self._dtype
    
    dtype = property(__getDType)