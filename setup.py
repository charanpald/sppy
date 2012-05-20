#from distutils.core import setup
from setuptools.extension import Extension
from setuptools import setup
from Cython.Distutils import build_ext

include  = ["/usr/include/eigen3/"]

setup(name = 'sparray',
      version = 0.1,
      author = 'Charanpal Dhanjal',
      author_email = 'charanpal@gmail.com',
      ext_modules=[Extension("sparray.csr_array", ["sparray/csr_array.pyx"], language="c++", include_dirs=include), 
                   Extension("sparray.dyn_array", ["sparray/dyn_array.pyx"], language="c++", include_dirs=include)],  
      cmdclass={'build_ext': build_ext})
