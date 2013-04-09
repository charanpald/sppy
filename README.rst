sppy
====

The aim of this project is to implement a fast sparse array library by wrapping the corresponding parts of Eigen in Python. The interface is similar to numpy so that existing code requires minimal change to work with sparse arrays. See http://packages.python.org/sppy/index.html for the documentation. 

Note: this code has been tested with Python 2.7/3.2, numpy 1.7.0 and scipy 0.11.0 on Ubuntu 12.10 64-bit. 

Changelog
---------

Changes in version 0.5: 

* Get nonzero indices for a particular row or col 
* Method to return all values 
* Convert to scipy csc_matrix 
* Support for RowMajor matrices 

Changes in version 0.4: 

* Added support for 1d arrays
