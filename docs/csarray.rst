csarray
=========

The csarray class represents a Compressed Sparse Array object which is essentially a 2d matrix or 1d vector with few nonzero elements. 

::

    >>> import numpy 
    >>> from sppy import csarray 
    >>> #Create a new column major dynamic array of float type
    >>> B = csarray((5, 5), storagetype="col") 
    >>> B[3, 3] = -0.2
    >>> B[0, 4] = -1.23
    
    
Methods 
-------
.. autoclass:: sppy.csarray
   :members:
   :inherited-members:

