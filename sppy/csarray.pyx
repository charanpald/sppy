# cython: profile=False
from cython.operator cimport dereference as deref, preincrement as inc 
from sppy.csarray_sub import csarray_int_colMajor, csarray_double_colMajor, csarray_float_colMajor, csarray_long_colMajor, csarray_short_colMajor, csarray_signed_char_colMajor  
from sppy.csarray_sub cimport csarray_int_rowMajor, csarray_double_rowMajor, csarray_float_rowMajor, csarray_long_rowMajor, csarray_short_rowMajor, csarray_signed_char_rowMajor
from sppy.csarray1d_sub import csarray1d_int, csarray1d_double, csarray1d_float, csarray1d_long, csarray1d_short, csarray1d_signed_char  
from sppy.csarray1d_sub cimport csarray1d_int, csarray1d_double, csarray1d_float, csarray1d_long, csarray1d_short, csarray1d_signed_char 
import struct
import numpy 
cimport numpy
 
numpy.import_array()


class csarray(object):  
    def __init__(self, S, dtype=numpy.float, storagetype="col"): 
        """
        Create a new empty 2d or 1d csarray with the given shape. 
        
        :param S: A tuple representing the shape of the matrix or a numpy array or scipy matrix 
        
        :param dtype: A numpy dtype for the elements
        
        :param storagetype: One of "row" or "col". 
        """
        if type(S) == tuple: 
            shape = S
        elif type(S) == int:  
            shape = S, 
        else: 
            try: 
                shape = S.shape
            except AttributeError:
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
            if storagetype=="col": 
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
            elif storagetype == "row": 
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
                raise ValueError("Unknown storage type: " + str(storagetype))
        else:
            raise ValueError("Only 1 and 2d arrays supported")
            
        try: 
            nonzeros = S.nonzero()
            if len(shape) == 2: 
                self._array[nonzeros] = numpy.array(S[S.nonzero()]).flatten()
            elif len(shape) == 1: 
                self._array[nonzeros[0]] = S[nonzeros[0]]
        except AttributeError: 
            pass
            
        self._dtype = dtype
        self.storagetype = storagetype
    
    def __abs__(self): 
        """
        Compute the absolute value of the elements of this matrix. 
        """
        resultArray = self._array.__abs__()
        result = csarray(resultArray.shape, self.dtype)
        result._array = resultArray
        return result    

    def __add__(self, A): 
        """
        Add this matrix to another one with identical dimentions. If A is a numpy 
        array it will be converted to a csarray. 
        
        :param A: The matrix to add. 
        """
        if isinstance(A, numpy.ndarray):
            A = csarray(A)        
        
        result = csarray(self.shape, self.dtype)
        result._array = self._array.__add__(A._array)
        return result

    def __convertBase(self, array, dtype): 
        """
        Convert a base class to csarray. 
        """
        del self._array
        self._array = array       
        self._dtype = dtype 

    def __getattr__(self, name):
        try: 
            return getattr(self, name)
        except: 
            return getattr(self._array, name)

    def __getDType(self): 
        return numpy.dtype(self._dtype)
        
    def __getitem__(self, inds):
        result = self._array.__getitem__(inds) 
        
        if type(result) in self.baseTypes: 
            newArray = csarray(result.shape, self.dtype)
            newArray.__convertBase(result, self.dtype)
            result = newArray
            
        return result 

    def __mul__(self, x):
        """
        Multiply this matrix with another one with identical dimentions.
        
        :param A: The matrix to multiply. 
        """
        newArray = self.copy() 
        newArray._array = newArray._array*x
        return newArray

    def __neg__(self): 
        """
        Negate all the elements of this matrix. 
        """
        resultArray = self._array.__neg__()
        result = csarray(resultArray.shape, self.dtype)
        result._array = resultArray
        return result
        
    def __setitem__(self, inds, val):
        """
        Set some elements of this matrix using given indices and values. 
        """
        self._array.__setitem__(inds, val) 
        
    def __str__(self): 
        """
        Return a string representation of the non-zero elements of the array. 
        """
        if self.ndim == 2: 
            outputStr = "csarray dtype:" + str(numpy.dtype(self.dtype)) + " shape:" + str(self.shape) + " non-zeros:" + str(self.getnnz()) + " storage:" + str(self.storagetype) + "\n"
            (rowInds, colInds) = self.nonzero() 
            vals = self.values()
            
            for i in range(self.getnnz()): 
                outputStr += "(" + str(rowInds[i]) + ", " + str(colInds[i]) + ")" + " " + str(vals[i]) 
                if i != self.getnnz()-1: 
                    outputStr += "\n"
        else: 
            outputStr = "csarray dtype:" + str(numpy.dtype(self.dtype)) + " shape:" + str(self.shape) + " non-zeros:" + str(self.getnnz()) + "\n"
            inds = self.nonzero()[0]
            vals = self[inds]
            
            for i in range(self.getnnz()): 
                outputStr += "(" + str(inds[i]) + ")" + " " + str(vals[i]) 
                if i != self.getnnz()-1: 
                    outputStr += "\n"
                    
        return outputStr 

    def __sub__(self, A):
        """
        Subtract this matrix from another one with identical dimentions.
        If A is a numpy array it will be converted to a csarray. 
        """
        if isinstance(A, numpy.ndarray):
            A = csarray(A)
            
        result = csarray(self.shape, self.dtype)
        result._array = self._array.__sub__(A._array)
        return result

    def compress(self): 
        """
        Turn this matrix into compressed sparse format by freeing extra memory 
        space in the buffer. 
        """
        self._array.compress()

    def copy(self): 
        """
        Return a copy of this array. 
        """
        newArray = csarray(self.shape, self.dtype, self.storagetype)
        newArray._array = self._array.copy()
        return newArray 
        
    def diag(self): 
        """
        Return a numpy array containing the diagonal entries of this matrix. If 
        the matrix is non-square then the diagonal array is the same size as the 
        smallest dimension. 
        """
        return self._array.diag()       
    
    def dot(self, A): 
        """
        Compute the dot product between this and either a csarray or numpy array A.
        
        :param A: The input numpy array or csarray. 
        """
        if isinstance(A, numpy.ndarray):  
            if A.ndim == 2: 
                A = numpy.ascontiguousarray(A)
                result = self._array.dotNumpy2d(A)
            else: 
                result = self._array.dotNumpy1d(A)
        else: 
            if A.ndim == 2: 
                resultArray = self._array.dotCsarray2d(A._array)
            else: 
                resultArray = self._array.dotCsarray1d(A._array)
            
            try: 
                result = csarray(resultArray.shape, A.dtype)
                result._array = resultArray
            except AttributeError:
                result = resultArray
            
        return result

    @staticmethod 
    def fromScipySparse(A, storagetype="col"): 
        """
        Take a scipy sparse matrix A and return a csarray, copying the data. 
        
        :param A: A scipy.sparse matrix 
        
        :param storagetype: The storage order of the elements of the output csarray. 
        """
        try: 
            import scipy.sparse
        except ImportError: 
            raise         
        
        rowInds, colInds = A.nonzero() 
        B = csarray(A.shape, dtype=A.dtype, storagetype=storagetype)
        B.put(A.data, rowInds, colInds, init=True)
        
        return B 

    def getnnz(self): 
        """
        Return the number of non-zero elements in the array 
        """
        return self._array.getnnz()

    def hadamard(self, A): 
        """
        Compute the hadamard (element-wise) product between this array and A. 
        """
        result = csarray(self.shape, self.dtype)
        result._array = self._array.hadamard(A._array)
        return result

    def max(self): 
        """
        Find the maximum element of this array. 
        """
        return self._array.max()

    def mean(self, axis=None): 
        """
        Find the mean value of this array. 
        
        :param axis: The axis of the array to compute the mean. 
        """
        if self.ndim == 2: 
            return self._array.mean(axis)
        else: 
            return self._array.mean()

    def min(self): 
        """
        Find the minimum element of this array. 
        """
        return self._array.min()

    def nonzero(self): 
        """
        Return a tuple of arrays corresponding to nonzero elements. 
        """
        return self._array.nonzero()


    def pdot(self, A): 
        """
        Compute the dot product between this and either a csarray or numpy array
        using multithreading.
        
        :param A: The input numpy array or csarray.  
        """
        if isinstance(A, numpy.ndarray):  
            if A.ndim == 2: 
                result = self._array.pdot2d(A)
            else: 
                result = self._array.pdot1d(A)
        else: 
            raise ValueError("Cannot pdot with A of type " + str(type(A)))
            
        return result

    def prune(self, double eps=10**-10, double precision=10**-20): 
        """
        Suppresses all nonzeros which are much smaller in magnitude than eps under 
        the tolerence precision. 
        """
        self._array.prune(eps, precision) 
        
    def reserve(self, int n): 
        """
        Reserve n nonzero entries and turns the matrix into uncompressed mode. 
        """
        self._array.reserve(n)
  
    def rowInds(self, int i):
        """
        Returns the non zero indices for the ith row. 
        
        :param i: The index of the row of the array. 
        """
        return self._array.rowInds(i)
      
    def std(self): 
        """
        Return the standard deviation of the array elements. 
        """
        return self._array.std()

    def sum(self, axis=None): 
        """
        Sum all of the elements in this array. If one specifies an axis 
        then we sum along the axis. 
        
        :param axis: The axis to sum along. 
        """
        if self.ndim == 2: 
            return self._array.sum(axis)
        else: 
            return self._array.sum()

    def toarray(self): 
        """
        Convert this sparse array into a numpy array. 
        """
        return self._array.toarray()
                     
    def toScipyCsc(self): 
        """
        Convert this matrix to a scipy sparse matrix. Returns a copy of the data in 
        csc_matrix form.  
        """  
        try: 
            import scipy.sparse
        except ImportError: 
            raise 
            
        if self.storagetype != "col": 
            raise ValueError("Method only supports col matrices")
    
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
        Convert this matrix to a scipy sparse matrix. Returns a copy of the data in 
        csr_matrix form. 
        """  
        try: 
            import scipy.sparse
        except ImportError: 
            raise 
            
        if self.storagetype != "row": 
            raise ValueError("Method only supports row matrices")
    
        rowInds, colInds = self.nonzero()  
        indPtrTemp = numpy.cumsum(numpy.bincount(rowInds, minlength=self.shape[0]))
        indPtr = numpy.zeros(self.shape[0]+1, numpy.int32)
        indPtr[1:self.shape[0]+1] = indPtrTemp         

        A = scipy.sparse.csr_matrix(self.shape, dtype=self.dtype)
        A.indices = numpy.array(colInds, numpy.int32) 
        A.data = self.values()  
        A.indptr = indPtr
        
        return A 

    def trace(self): 
        """
        Returns the trace of the array which is simply the sum of the diagonal 
        entries. 
        """
        return self._array.trace()
   
    def transpose(self): 
        """
        Swap the rows and columns of this matrix, i.e. perform a transpose operation. 
        """
        resultArray = self._array.transpose()
        result = csarray(resultArray.shape, self.dtype)
        result._array = resultArray
        return result

    def values(self): 
        """
        Return the values of this object according to the elements returned 
        using nonzero.  
        """
        return self._array.values()
    
    def var(self): 
        """
        Return the variance of the elements of this array. 
        """
        return self._array.var()
    
    dtype = property(__getDType)
    T = property(transpose)
    baseTypes = [csarray_int_colMajor, csarray_double_colMajor, csarray_float_colMajor, csarray_long_colMajor, csarray_short_colMajor, csarray_signed_char_colMajor]
    baseTypes.extend([csarray_int_rowMajor, csarray_double_rowMajor, csarray_float_rowMajor, csarray_long_rowMajor, csarray_short_rowMajor, csarray_signed_char_rowMajor])
