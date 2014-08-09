sppy
====

The aim of this project is to implement a fast sparse array library by wrapping the corresponding parts of Eigen in Python. The interface is similar to numpy so that existing code requires minimal change to work with sparse arrays. See http://packages.python.org/sppy/index.html for the documentation. 

Note: this code has been tested with Python 2.7/3.2, numpy 1.7.1 and scipy 0.12.0 on Ubuntu 14.04 64-bit. 

Changelog
---------

Changes in version 0.6.6: 

* Fix bugs in slicing 
* Turn off bounds checking 
* Element-wise sin, cos, floor, ceil, sign for 2D arrays 

Changes in version 0.6.5: 

* Add methods to support pickling 
* nonzeroRowsPtr method 
* Several bug fixes 

Changes in version 0.6.4: 

* Constructor of csarray accepts scipy.sparse 
* Added io module to read and write matrix market files 
* In csarray: clip, and submatrix methods 
* Various optimisations and bug fixes 
* Better documentation 

Changes in version 0.6.3: 

* Added prune function to remove nnz elements 
* Bug fix for numpy.int32 in csarray.__getitem__ 

Changes in version 0.6.2: 

* Added sppy.linalg.biCGSTAB which solves linear equations of the form Ax = b with x unknown. 

Changes in version 0.6.1: 

* Better documentation 
* Added sppy.linalg.rsvd (randomised Singular Value Decomposition) and sppy.linalg.norm 

Changes in version 0.6: 

* Dot product with numpy arrays, and parallel version (pdot)
* sppy.linalg.GeneralLinearOperator to work with some scipy.sparse.linalg functions 
* Optimisations to put method
* Convert from scipy.sparse matrices (csarray.fromScipySparse)
* Automatic generation of specialised templates in setup.py (credit: Björn Dahlgren)

