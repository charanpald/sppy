Compressed Sparse Array 
=======================

The csarray class represents a Compressed Sparse Array object which is essentially a 2d matrix or 1d vector with few nonzero elements. It can be created in the following manner: 

::

    >>> import numpy 
    >>> from sppy import csarray 
    >>> #Create a new column major dynamic array of float type
    >>> B = csarray((5, 5), storagetype="col") 
    >>> B[3, 3] = -0.2
    >>> B[0, 4] = -1.23
    
which creates a 2d array with 5 rows and 5 columns. Alternatively, one can create a csarray using a numpy array, or another csarray: 

:: 

    >>> A = numpy.random.rand(5, 7) 
    >>> B = csarray(A, storagetype="col")
    >>> C = csarray(B, storagetype="row") 
  


Methods 
-------
.. autoclass:: sppy.csarray
   :members:
   :inherited-members:

