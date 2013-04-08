Sparse Matrices and Vectors
===========================

The csarray class represents a 1 dimensional sparse vector or a 2-dimensional compressed sparse matrix in row or column major format. To see more details about the underlying storage mechanism, refer to the `Eigen SparseMatrix page <http://eigen.tuxfamily.org/dox/TutorialSparse.html>`_. 

Construction
------------

As an example the follow code creates a new float array in column major format with 5 rows and 5 columns and then assigns -0.2 to the element at index (3, 3). 

:: 

    >>> from sppy import csarray 
    >>> B = csarray((5, 5)) 
    >>> B[3, 3] = -0.2
    
One can also create arrays of int types for example, by using 

:: 

    >>> import numpy
    >>> from sppy import csarray 
    >>> B = csarray((5, 5), dtype=numpy.int) 
    >>> C = csarray((5, 5), dtype=numpy.int, storageType="rowMajor") 
    >>> b = csarray(5, dtype=numpy.int) 

Notice that b is a 1-dimensional sparse vector, B is an array in column major format, and C is in row major format.     
    
The currently supported dtypes are: int8, int16, int32 int64, float32 and float64. An alternative way to construct a csarray is by using an existing numpy array or csarray,

:: 

    >>> #Construct using numpy array 
    >>> A = numpy.array([[3, 0, 0], [1.2, 2, 0], [0, 0, 0.4]])
    >>> B = csarray(A, dtype=numpy.float32) 
    >>>
    >>> #Construct using existing sppy 
    >>> D = csarray(B, dtype=numpy.int)
   
Other ways of creating arrays include the functions zeros, ones, diag, eye: and rand: 

:: 

    >>> import sppy
    >>> A = sppy.eye(10) 
    >>> B = sppy.diag(numpy.array([1, 2, 3]) 
    >>> C = sppy.rand((5, 7), 0.1)
    >>> D = sppy.zeros((10, 10)) 
    >>> E = sppy.ones((10, 10))  

Here, A is a 10 by 10 identity matrix, B is a 3 by 3 matrix with [1, 2, 3] along its diagonal entries, C is a 5 by 7 matrix with uniformly random elements inserted in approximately a proportion of 0.1 of the matrix, D is an all zeros matrix, and E is all ones. 

Assigning Elements
------------------

When assembling an array it is more efficient to add elements when the number of elements is reserved in advance, and similarly when one is finished adding elements we can compress the resulting matrix. 

:: 

    >>> B = csarray((1000, 1000))
    >>> B.reserve(500) 
    >>> #Elements are added here 
    >>> #Now compress the final matrix.   
    >>> B.compress()
    
An efficient way to assign elements is using the indexing notation, 

::

    >>> B = csarray((10, 10))
    >>> B.reserve(3) 
    >>> B[numpy.array([1, 2, 3]), numpy.array([4, 5, 6])] = numpy.array([0.1, 0.2, 0.4]) 
    >>> B.compress()

Operations
----------

Many of the same properties can be found in csarray as numpy arrays: B.shape gives the dimensions of the array, B.ndim is the number of dimensions, B.size is the total number of elements. Other methods currently implemented include: min, max, mean, sum, std, var, trace, diag, tranpose. One can also add, subtract, negate, matrix multiply, find the Hadamard product and find the absolute value of csarrays. To convert to a numpy.array use the toarray method. 

:: 

    >>> A = sppy.rand((5, 7), 0.1) 
    >>> B = sppy.rand((7, 5), 0.1)
    >>> C = sppy.rand((5, 7), 0.1)
    >>> D = csarray(numpy.random.randn(7, 7))
    >>> E = A + C 
    >>> F = A.dot(B) 
    >>> g = A.sum() 
    >>> H = abs(A)
    >>> I = B.T
    >>> j = D.trace()
    >>> K = D.toarray()
    >>> 
    >>> #Vectors 
    >>> a = sppy.rand(5, 0.1) 
    >>> b = sppy.rand(5, 0.1)
    >>> c = a + b 
    >>> d = a * b 
    >>> e = a.dot(b)
    >>> f = a.toarray()
    


