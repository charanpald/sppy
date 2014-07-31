import numpy 
from sppy import csarray 

"""
Some utility function to create sparse arrays. 
"""


def diag(a, storagetype="col"): 
    """
    Takes a 1d numpy array and creates a csarray with the corresponding diagonal 
    elements.
    
    :param a: A 1d numpy array 
    :type a: `numpy.ndarray`
    
    :param storagetype: The storage type of the csarray ("row" or "col")
    :type storagetype: `str`
    """
    n = a.shape[0]
    A = csarray((n, n), dtype=a.dtype, storagetype=storagetype)
    
    A[numpy.arange(n), numpy.arange(n)] = a
    
    return A
    
def eye(n, dtype=numpy.float):
    """
    Create the identity matrix of size n by n. 
    
    :param n: The size of the output array 
    :type n: `int`
    
    :param dtype: The data type of the output array (e.g. numpy.int)  
    """
    
    result = diag(numpy.ones(n, dtype=dtype))
    return result 
    
def rand(shape, density, dtype=numpy.float, storagetype="col"): 
    """
    Generate a random sparse matrix with m rows and n cols with given density 
    and dtype. 
    
    :param shape: The shape of the output array (m, n)    
    
    :param density: The proportion of non zero elements to create
    
    :param dtype: The data type of the output array (only supports floats at the moment)  
    
    :param storagetype: The storage type of the csarray ("row" or "col")
    :type storagetype: `str`
    """
    result = csarray(shape, dtype, storagetype=storagetype)
    size = result.size
    numEntries = int(size*density)
    
    inds = numpy.random.randint(0, size, numEntries)
    
    if result.ndim == 2: 
        rowInds, colInds = numpy.unravel_index(inds, shape) 
        rowInds = numpy.array(rowInds, numpy.int32)
        colInds = numpy.array(colInds, numpy.int32)
        result.put(numpy.array(numpy.random.rand(numEntries), dtype), rowInds, colInds, init=True)
    elif result.ndim == 1: 
        result[inds] = numpy.array(numpy.random.rand(numEntries), dtype)
    
    return result 
    

def zeros(shape, dtype=numpy.float, storageType="col"): 
    """
    Create a zeros matrix of the given shape and dtype. 
    
    :param shape: The shape of the output array 

    :param dtype: The data type of the output array (e.g. numpy.int)    
    
    :param storagetype: The storage type of the csarray ("row" or "col")
    :type storagetype: `str`
    """
    result = csarray(shape, dtype, storageType)
    return result
    
def ones(shape, dtype=numpy.float, storageType="col"): 
    """
    Create a ones matrix of the given shape and dtype. Generally a bad idea 
    for large matrices.
    
    :param shape: The shape of the output array 
    
    :param dtype: The data type of the output array (e.g. numpy.int)  
    
    :param storagetype: The storage type of the csarray ("row" or "col")
    :type storagetype: `str`
    """
    result = csarray(shape, dtype, storageType)
    result.ones()
    return result


