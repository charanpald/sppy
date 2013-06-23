sppy
====

The aim of this project is to implement a fast sparse array library by wrapping the corresponding parts of Eigen in Python. The interface is similar to numpy so that existing code requires minimal change to work with sparse arrays. See http://packages.python.org/sppy/index.html for the documentation. 

Note: this code has been tested with Python 2.7/3.2, numpy 1.7.1 and scipy 0.12.0 on Ubuntu 13.04 64-bit. 

Changelog
---------

Changes in version 0.6.1: 

* Better documentation 
* Added sppy.linalg.rsvd (randomised Singular Value Decomposition) and sppy.linalg.norm 

Changes in version 0.6: 

* Dot product with numpy arrays, and parallel version (pdot)
* sppy.linalg.GeneralLinearOperator to work with some scipy.sparse.linalg functions 
* Optimisations to put method
* Convert from scipy.sparse matrices (csarray.fromScipySparse)
* Automatic generation of specialised templates in setup.py (credit: Björn Dahlgren)

