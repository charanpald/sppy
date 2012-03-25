#from distutils.core import setup
from setuptools.extension import Extension
from setuptools import setup
from Cython.Distutils import build_ext

setup(name = 'sparray',
      version = 0.1,
      author = 'Charanpal Dhanjal',
      author_email = 'charanpal@gmail.com',
      summary = 'A sparse matrix package based on Boost uBLAS',
      ext_modules=[Extension("sparray.map", ["sparray/map.pyx"], language="c++")],  
      cmdclass={'build_ext': build_ext})
