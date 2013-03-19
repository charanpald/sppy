sppy
====

The aim of this project is to implement a fast sparse array library by wrapping the corresponding parts of Eigen in Python. The interface is similar to numpy so that existing code requires minimal change to work with sparse arrays. See http://packages.python.org/sppy/index.html for the documentation. 

Changelog
---------

Changes in version 0.5: 
* Get nonzero indices for a particular row or col 
* Method to return all values 
* Convert to scipy csc_matrix 
* Support for RowMajor matrices 

Changes in version 0.4: 

* Added support for 1d arrays
