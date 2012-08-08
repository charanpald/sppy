import numpy 
from sparray import csarray 

"""
Some utility function to create sparse arrays. 
"""

def diag(x): 
    """
    From a 1D numpy array x create a diagonal sparse array. 
    """
    result = csarray((x.shape[0], x.shape[0]), x.dtype)
    result[(numpy.arange(x.shape[0]), numpy.arange(x.shape[0]))] = x
    
    return result 
    
    
def eye(n, dtype=numpy.float):
    """
    Create the identity matrix of size n. 
    """
    
    result = diag(numpy.ones(n, dtype=dtype))
    return result 
    
def rand(m, n, density, dtype=numpy.float): 
    """
    Generate a random sparse matrix with m rows and n cols with given density 
    and dtype. 
    """
    result = csarray((m, n), dtype)
    numEntries = int((m*n)*density)
    
    inds = numpy.random.randint(0, m*n, numEntries)
    rowInds, colInds = numpy.unravel_index(inds, (m, n))    
    result[rowInds, colInds] = numpy.random.rand(numEntries)
    
    return result 
    

def zeros(shape, dtype=numpy.float): 
    """
    Create a zeros matrix of the given shape and dtype. 
    """
    result = csarray(shape, dtype)
    return result
    
def ones(shape, dtype=numpy.float): 
    """
    Create a ones matrix of the given shape and dtype. Generally a bad idea 
    for large matrices. 
    """
    result = csarray(shape, dtype)
    result.ones()
    return result
    