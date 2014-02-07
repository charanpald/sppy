Compressed Sparse Array 
=======================

The csarray class represents a Compressed Sparse Array object which is essentially a 2d matrix or 1d vector with few nonzero elements. The underlying implementation uses the Eigen sparse matrix code. A csarray is initially specified using a size, data type (int, float etc.) and storage type (row or column major format). 

As an example, a csarray of shape (5, 5) can be created in the following manner: 

::

    >>> import numpy 
    >>> from sppy import csarray 
    >>> #Create a new column major dynamic array of float type
    >>> B = csarray((5, 5), storagetype="col", dtype=numpy.float) 
    >>> B[3, 3] = -0.2
    >>> B[0, 4] = -1.23
    
Alternatively, one can create a csarray using a numpy array, a scipy sparse matrix, or a csarray: 

:: 

    >>> A = numpy.random.rand(5, 7) 
    >>> B = csarray(A, storagetype="col")
    >>> C = csarray(B, storagetype="row") 
  
In addition, one can generate arrays using a predefined structure, for example diagonal matrices or randomly generated. See the documentation below for details and also for information on the operations on these objects. 

Generating Arrays
-----------------
.. automodule:: sppy
   :members: diag, eye, ones, zeros, rand

Methods 
-------
.. autoclass:: sppy.csarray
    :inherited-members:
    :members:
    
   
   
   

