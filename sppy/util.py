import numpy 
from sppy import csarray 

"""
Some utility function to create sparse arrays. 
"""


def diag(a, storagetype="col"): 
    """
    Takes a 1d numpy array and creates a csarray with the corresponding diagonal 
    elements. 
    """
    n = a.shape[0]
    A = csarray((n, n), dtype=a.dtype, storagetype=storagetype)
    
    A[numpy.arange(n), numpy.arange(n)] = a
    
    return A
    
def eye(n, dtype=numpy.float):
    """
    Create the identity matrix of size n. 
    """
    
    result = diag(numpy.ones(n, dtype=dtype))
    return result 
    
def rand(shape, density, dtype=numpy.float): 
    """
    Generate a random sparse matrix with m rows and n cols with given density 
    and dtype. 
    """
    result = csarray(shape, dtype)
    size = result.size
    numEntries = int(size*density)
    
    inds = numpy.random.randint(0, size, numEntries)
    
    if result.ndim == 2: 
        rowInds, colInds = numpy.unravel_index(inds, shape)    
        result[rowInds, colInds] = numpy.random.rand(numEntries)
    elif result.ndim == 1: 
        result[inds] = numpy.random.rand(numEntries)
    
    return result 
    

def zeros(shape, dtype=numpy.float, storageType="col"): 
    """
    Create a zeros matrix of the given shape and dtype. 
    """
    result = csarray(shape, dtype, storageType)
    return result
    
def ones(shape, dtype=numpy.float, storageType="col"): 
    """
    Create a ones matrix of the given shape and dtype. Generally a bad idea 
    for large matrices. 
    """
    result = csarray(shape, dtype, storageType)
    result.ones()
    return result

#def solve(A, b): 
    """
    Solve a system of linear equations given by Ax = b.  
    """
    
    
