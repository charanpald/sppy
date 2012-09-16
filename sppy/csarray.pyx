# cython: profile=False
from cython.operator cimport dereference as deref, preincrement as inc 
from sppy.csarray_sub import csarray_int, csarray_double, csarray_float, csarray_long, csarray_short, csarray_signed_char  
import struct
import numpy 
cimport numpy
 
numpy.import_array()


class csarray(object): 
    def __init__(self, S, dtype=numpy.float): 
        """
        Create a new csarray using the given shape and dtype. If the dtype is 
        float or int we assume 64 bits. 
        """
        if type(S) == tuple: 
            shape = S
        elif type(S) == numpy.ndarray or type(S) == csarray:
            shape = S.shape
        else: 
            raise ValueError("Invalid parameter: " + str(S))
            
        
        if dtype == numpy.float32: 
            self._array = csarray_float(shape)        
        elif dtype == numpy.float64 or dtype==numpy.float: 
            self._array = csarray_double(shape)
        elif dtype == numpy.int8: 
            self._array = csarray_signed_char(shape)    
        elif dtype == numpy.int16: 
            self._array = csarray_short(shape)
        elif dtype == numpy.dtype(int): 
            self._array = csarray_int(shape)
        elif dtype == numpy.dtype(long) or dtype == numpy.int: 
            self._array = csarray_long(shape)
        else: 
            raise ValueError("Unknown dtype: " + str(dtype))
            
        if type(S) == numpy.ndarray or type(S) == csarray:
            nonzeros = S.nonzero()
            self._array[nonzeros] = S[nonzeros]
            
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
        resultArray = self._array.__abs__()
        result = csarray(resultArray.shape, resultArray.dtype)
        result._array = resultArray
        return result
        
    def __neg__(self): 
        resultArray = self._array.__neg__()
        result = csarray(resultArray.shape, resultArray.dtype)
        result._array = resultArray
        return result

    def __add__(self, A): 
        result = csarray(self.shape, self.dtype)
        result._array = self._array.__add__(A._array)
        return result
        
    def __sub__(self, A): 
        result = csarray(self.shape, self.dtype)
        result._array = self._array.__sub__(A._array)
        return result
        
    def hadamard(self, A): 
        result = csarray(self.shape, self.dtype)
        result._array = self._array.hadamard(A._array)
        return result
        
    def dot(self, A): 
        resultArray = self._array.dot(A._array)
        result = csarray(resultArray.shape, resultArray.dtype)
        result._array = resultArray
        return result
        
    def transpose(self): 
        resultArray = self._array.transpose()
        result = csarray(resultArray.shape, resultArray.dtype)
        result._array = resultArray
        return result
        
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
    T = property(transpose)