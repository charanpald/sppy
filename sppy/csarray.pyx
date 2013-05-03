# cython: profile=False
from cython.operator cimport dereference as deref, preincrement as inc 
from sppy.csarray_sub import csarray_int_colMajor, csarray_double_colMajor, csarray_float_colMajor, csarray_long_colMajor, csarray_short_colMajor, csarray_signed_char_colMajor  
from sppy.csarray_sub import csarray_int_rowMajor, csarray_double_rowMajor, csarray_float_rowMajor, csarray_long_rowMajor, csarray_short_rowMajor, csarray_signed_char_rowMajor
from sppy.csarray1d_sub import csarray1d_int, csarray1d_double, csarray1d_float, csarray1d_long, csarray1d_short, csarray1d_signed_char  
import struct
import numpy 
cimport numpy
 
numpy.import_array()


class csarray(object): 
    def __init__(self, S, dtype=numpy.float, storageType="colMajor"): 
        """
        Create a new csarray using the given shape and dtype. If the dtype is 
        float or int we assume 64 bits. 
        """
        if type(S) == tuple: 
            shape = S
        elif type(S) == int: 
            shape = S, 
        elif type(S) == numpy.ndarray or type(S) == csarray:
            shape = S.shape
        else: 
            raise ValueError("Invalid parameter: " + str(S))
            
        if len(shape) == 1: 
            if dtype == numpy.float32: 
                self._array = csarray1d_float(shape)        
            elif dtype == numpy.float64 or dtype==numpy.float: 
                self._array = csarray1d_double(shape)
            elif dtype == numpy.int8: 
                self._array = csarray1d_signed_char(shape)    
            elif dtype == numpy.int16: 
                self._array = csarray1d_short(shape)
            elif dtype == numpy.dtype(int): 
                self._array = csarray1d_int(shape)
            elif dtype == numpy.dtype(long) or dtype == numpy.int: 
                self._array = csarray1d_long(shape)
            else: 
                raise ValueError("Unknown dtype: " + str(dtype))  
        elif len(shape) == 2: 
            if storageType=="colMajor": 
                if dtype == numpy.float32: 
                    self._array = csarray_float_colMajor(shape)        
                elif dtype == numpy.float64 or dtype==numpy.float: 
                    self._array = csarray_double_colMajor(shape)
                elif dtype == numpy.int8: 
                    self._array = csarray_signed_char_colMajor(shape)    
                elif dtype == numpy.int16: 
                    self._array = csarray_short_colMajor(shape)
                elif dtype == numpy.dtype(int): 
                    self._array = csarray_int_colMajor(shape)
                elif dtype == numpy.dtype(long) or dtype == numpy.int: 
                    self._array = csarray_long_colMajor(shape)
                else: 
                    raise ValueError("Unknown dtype: " + str(dtype))
            elif storageType == "rowMajor": 
                if dtype == numpy.float32: 
                    self._array = csarray_float_rowMajor(shape)        
                elif dtype == numpy.float64 or dtype==numpy.float: 
                    self._array = csarray_double_rowMajor(shape)
                elif dtype == numpy.int8: 
                    self._array = csarray_signed_char_rowMajor(shape)    
                elif dtype == numpy.int16: 
                    self._array = csarray_short_rowMajor(shape)
                elif dtype == numpy.dtype(int): 
                    self._array = csarray_int_rowMajor(shape)
                elif dtype == numpy.dtype(long) or dtype == numpy.int: 
                    self._array = csarray_long_rowMajor(shape)
                else: 
                    raise ValueError("Unknown dtype: " + str(dtype))
            else: 
                raise ValueError("Unknown storage type: " + str(storageType))
        else:
            raise ValueError("Only 1 and 2d arrays supported")
            
        if type(S) == numpy.ndarray or type(S) == csarray:
            nonzeros = S.nonzero()
            if len(shape) == 2: 
                self._array[nonzeros] = S[nonzeros]
            elif len(shape) == 1: 
                self._array[nonzeros[0]] = S[nonzeros[0]]
            
        self._dtype = dtype
            
    def __getattr__(self, name):
        try: 
            return getattr(self, name)
        except: 
            return getattr(self._array, name)

    def __getitem__(self, inds):
        result = self._array.__getitem__(inds) 
        
        if type(result) in self.baseTypes: 
            newArray = csarray(result.shape, self.dtype)
            newArray.convertBase(result, self.dtype)
            result = newArray
            
        return result 
        
    def __setitem__(self, inds, val):
        self._array.__setitem__(inds, val) 
        
    def __abs__(self): 
        resultArray = self._array.__abs__()
        result = csarray(resultArray.shape, self.dtype)
        result._array = resultArray
        return result
        
    def __neg__(self): 
        resultArray = self._array.__neg__()
        result = csarray(resultArray.shape, self.dtype)
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
        
        try: 
            result = csarray(resultArray.shape, A.dtype)
            result._array = resultArray
        except AttributeError:
            result = resultArray
            
        return result
        
    def transpose(self): 
        resultArray = self._array.transpose()
        result = csarray(resultArray.shape, self.dtype)
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
        
        if self.ndim == 2: 
            (rowInds, colInds) = self.nonzero() 
            vals = self.values()
            
            for i in range(self.getnnz()): 
                outputStr += "(" + str(rowInds[i]) + ", " + str(colInds[i]) + ")" + " " + str(vals[i]) 
                if i != self.getnnz()-1: 
                    outputStr += "\n"
        else: 
            inds = self.nonzero()[0]
            vals = self[inds]
            
            for i in range(self.getnnz()): 
                outputStr += "(" + str(inds[i]) + ")" + " " + str(vals[i]) 
                if i != self.getnnz()-1: 
                    outputStr += "\n"
                    
        return outputStr 
    
    def __getDType(self): 
        return self._dtype
        
    def convertBase(self, array, dtype): 
        """
        Convert a base class to csarray. 
        """
        del self._array
        self._array = array       
        self._dtype = dtype 
       
    def toScipyCsc(self): 
        """
        Convert this matrix to scipy. Returns a copy of the data in csc_matrix 
        form. 
        """  
        try: 
            import scipy.sparse
        except ImportError: 
            raise 
            
        if self.storage != "colMajor": 
            raise ValueError("Method only supports ColMajor matrices")
    
        rowInds, colInds = self.nonzero()  
        indPtrTemp = numpy.cumsum(numpy.bincount(colInds, minlength=self.shape[0]))
        indPtr = numpy.zeros(self.shape[1]+1, numpy.int32)
        indPtr[1:self.shape[1]+1] = indPtrTemp         

        A = scipy.sparse.csc_matrix(self.shape, dtype=self.dtype)
        A.indices = numpy.array(rowInds, numpy.int32) 
        A.data = self.values()  
        A.indptr = indPtr
        
        return A 
        
    def toScipyCsr(self): 
        """
        Convert this matrix to scipy. Returns a copy of the data in csr_matrix 
        form. 
        """  
        try: 
            import scipy.sparse
        except ImportError: 
            raise 
            
        if self.storage != "rowMajor": 
            raise ValueError("Method only supports RowMajor matrices")
    
        rowInds, colInds = self.nonzero()  
        indPtrTemp = numpy.cumsum(numpy.bincount(rowInds, minlength=self.shape[0]))
        indPtr = numpy.zeros(self.shape[0]+1, numpy.int32)
        indPtr[1:self.shape[0]+1] = indPtrTemp         

        A = scipy.sparse.csr_matrix(self.shape, dtype=self.dtype)
        A.indices = numpy.array(colInds, numpy.int32) 
        A.data = self.values()  
        A.indptr = indPtr
        
        return A 
     
    dtype = property(__getDType)
    T = property(transpose)
    baseTypes = [csarray_int_colMajor, csarray_double_colMajor, csarray_float_colMajor, csarray_long_colMajor, csarray_short_colMajor, csarray_signed_char_colMajor]
    baseTypes.extend([csarray_int_rowMajor, csarray_double_rowMajor, csarray_float_rowMajor, csarray_long_rowMajor, csarray_short_rowMajor, csarray_signed_char_rowMajor])