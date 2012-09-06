Downloading
-----------
Download for Windows, Linux or Mac OS using: 

-  `The Python Package Index (PyPI) <http://pypi.python.org/pypi/SpPy/>`_ 

To use this library, you must have `Python <http://www.python.org/>`_, `NumPy <http://numpy.scipy.org/>`_ and `Eigen 3.1 or higher <http://eigen.tuxfamily.org/>`_. The code has been verified on Python 2.7.2, Numpy 1.6.1 and Eigen 3.1.1 on Ubuntu 12.04. The automatic testing routine requires Python 2.7 or later. The source code repository is available at `github <https://github.com/charanpald/SpPy>`_ for those that want the bleeding edge, or are interested in development.  

Installation 
-------------
Ensure that `pip <http://pypi.python.org/pypi/pip>`_ is installed, and then install SpPy in the following way: 

::

	pip install sppy

If installing from source unzip the SpPy-x.y.z.tar.gz file and then run setup.py as follows: 

::

	python setup.py install 

In order to test the library (recommended), using the following commands in python 

::

	import sppy 
	sppy.test() 

and check that all tested pass. 
