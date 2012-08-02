.. sparray documentation master file, created by
   sphinx-quickstart on Sun May 27 11:59:41 2012.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to sparray's documentation!
===================================

The aim of the sparray project is to implement a fast sparse array library by wrapping the corresponding parts of Eigen in Python. The interface is similar to numpy so that existing code requires minimal change to work with sparse arrays.

Downloading
-----------
Download for Windows, Linux or Mac OS using: 

-  `The Python Package Index (PyPI) <http://pypi.python.org/pypi/sparray/>`_ 

To use this library, you must have `Python <http://www.python.org/>`_, `NumPy <http://numpy.scipy.org/>`_ and `Eigen 3.1 or higher <http://eigen.tuxfamily.org/>`_. The code has been verified on Python 2.7.2, Numpy 1.6.1 and Eigen 3.1.1. The automatic testing routine requires Python 2.7 or later. The source code repository is available at `github <https://github.com/charanpald/sparray>`_ for those that want the bleeding edge, or are interested in development.  

Installation 
-------------
Ensure that `pip <http://pypi.python.org/pypi/pip>`_ is installed, and then install sparray in the following way: 

::

	pip install sparray

If installing from source unzip the sparray-x.y.z.tar.gz file and then run setup.py as follows: 

::

	python setup.py install 

In order to test the library (recommended), using the following commands in python 

::

	import sparray 
	sparray.test() 

and check that all tested pass. 

Examples 
--------
A good way to learn about the features of the library is to look at the source code, in particular :doc:`csarray`. More complete documentation is forthcoming, but for now here are some examples

:: 

    >>> import numpy 
    >>> from sparray import csarray 
    >>> #Create a new column major dynamic array of float type
    >>> B = csarray((5, 5)) 
    >>> B[3, 3] = -0.2
    >>> B[0, 4] = -1.23
    >>> print(B.shape)
    (5, 5)
    >>> print(B.size)
    25
    >>> print(B.getnnz())
    2
    >>> print(B)
    csarray dtype:float64 shape:(5, 5) non-zeros:2
    (3, 3) -0.2
    (0, 4) -1.23
    >>> B[numpy.array([0, 1]), numpy.array([0,1])] = 27
    >>> print(B)
    csarray dtype:float64 shape:(5, 5) non-zeros:4
    (0, 0) 27.0
    (1, 1) 27.0
    (3, 3) -0.2
    (0, 4) -1.23
    >>> print(B.sum())
    52.57

Reference pages:

.. toctree::
   :maxdepth: 1

   csarray

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

