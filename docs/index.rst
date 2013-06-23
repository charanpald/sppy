.. SpPy documentation master file, created by
   sphinx-quickstart on Sun May 27 11:59:41 2012.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to SpPy's documentation!
===================================

SpPy is a fast sparse matrix/array library written in Python and based on the C++ matrix library Eigen. A sparse matrix is one in which many of the elements are zeros, and by storing only non-zero elements, one can often make memory and computational savings over dense matrices which store all elements. The library supports (compressed) sparse matrices, sparse vectors and a number of linear algebra operations (such as the randomised SVD and matrix norm). Furthermore, SpPy has a similar interface to numpy so that existing code requires minimal change to work with sparse matrices and vectors.

Update: this project was formerly called sparray. 


Getting Started 
---------------
First, read the installation guide in the reference documentation linked below. More complete documentation is forthcoming, but for now here are some examples. 

:: 

    >>> import numpy 
    >>> from sppy import csarray 
    >>> #Create a new column major dynamic array of float type
    >>> B = csarray((5, 5), storagetype="col") 
    >>> B[3, 3] = -0.2
    >>> B[0, 4] = -1.23
    >>> print(B.shape)
    (5, 5)
    >>> print(B.size)
    25
    >>> print(B.getnnz())
    2
    >>> print(B)
    csarray dtype:float64 shape:(5, 5) non-zeros:2 storage:col
    (3, 3) -0.2
    (0, 4) -1.23
    >>> B[numpy.array([0, 1]), numpy.array([0,1])] = 27
    >>> print(B)
    csarray dtype:float64 shape:(5, 5) non-zeros:4 storage:col
    (0, 0) 27.0
    (1, 1) 27.0
    (3, 3) -0.2
    (0, 4) -1.23
    >>> print(B.sum())
    52.57

User guide:

.. toctree::
   :maxdepth: 2
  
   install
   sparseops 
   
Reference guide:

.. toctree::
   :maxdepth: 2
   
   csarray
   linalg 

Support
--------

For any questions or comments please email me at <my first name> at gmail dot com. 


.. toctree::
   :maxdepth: 2


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

