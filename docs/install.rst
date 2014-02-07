Downloading
===========
Download for Windows, Linux or Mac OS using: 

-  `The Python Package Index (PyPI) <http://pypi.python.org/pypi/sppy/>`_ 

To use this library, you must have `Python <http://www.python.org/>`_, `NumPy <http://numpy.scipy.org/>`_, `Eigen 3.1 or higher <http://eigen.tuxfamily.org/>`_ and `OpenMP <http://openmp.org/wp/>`_. If you are using Python 3.3 or higher you will need `distribute <https://pypi.python.org/pypi/distribute>`_. The code has been verified on Python 2.7.4, Numpy 1.7.1 and Eigen 3.1.1 on Ubuntu 13.04. The automatic testing routine requires Python 2.7 or later. The source code repository is available at `github <https://github.com/charanpald/SpPy>`_ for those that want the bleeding edge, or are interested in development (see also the `wiki <https://github.com/charanpald/sppy/wiki>`_ in the later case).  

Installation 
============
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

Windows 
-------

To install on Windows one needs to have Python installed along with Numpy and a C++ compiler such as `MinGW <http://www.mingw.org/>`_. Here we assume that you are using MinGW. Download Eigen and copy it to the MinGW include directory, which is typically C:\\MinGW32\\include. One can then build and install sppy using the following two commands: 

::

    python setup.py build --compiler=mingw32
    python setup.py install 

Note that sppy is currently untested on Windows. 

