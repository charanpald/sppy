csarray
=======

The csarray class represents a compressed sparse matrix in column major format. To see more details about the underlying storage mechanism, refer to the `Eigen SparseMatrix page <http://eigen.tuxfamily.org/dox/TutorialSparse.html>`_. As an example the follow code creates a new float array with 5 rows and 5 columns and then assigns -0.2 to the element at index (3, 3).  

:: 

    >>> from sparray import csarray 
    >>> B = csarray((5, 5)) 
    >>> B[3, 3] = -0.2
    
One can also create arrays of int types for example, by using 

:: 

    >>> import numpy
    >>> from sparray import csarray 
    >>> B = csarray((5, 5), dtype=numpy.int) 
    
Many of the same properties can be found in csarray as numpy arrays: B.shape gives the dimensions of the array, B.ndim is the number of dimensions, B.size is the total number of elements. 

When creating an array it is more efficient to add elements when the number of elements is reserved in advance, and similarly when one is finished adding elements on can compress the resulting matrix. 

    >>> B = csarray((1000, 1000))
    >>> B.reserve(500) 
    >>> #Elements are added here 
    >>> #Now compress the final matrix.   
    >>> B.compress()
    
Other methods currently implemented include: min, max, mean, sum, std, var, trace, diag. One can also add, subtract, negate and find the absolute value of csarrays. To convert to a numpy.array use the toarray method. 


