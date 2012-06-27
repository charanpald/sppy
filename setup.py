#from distutils.core import setup
from setuptools import setup
from distutils.extension import Extension

try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True

cmdclass = { }

if use_cython:
    ext_modules=[Extension("sparray.csr_array", ["sparray/dyn_array.pyx"], language="c++")]  
    cmdclass.update({ 'build_ext': build_ext })
else:
    ext_modules = [Extension("sparray.csr_array", [ "sparray/dyn_array.cpp" ])] 

setup(name = 'sparray',
      version = "0.1",
      author = 'Charanpal Dhanjal',
      author_email = 'charanpal@gmail.com',
      packages = ['sparray', 'sparray.test'],
      install_requires=['numpy>=1.5.0'],
      url = 'http://packages.python.org/sparray/',
      license = 'GNU Library or Lesser General Public License (LGPL)',
      platforms=["OS Independent"],
      keywords=['numpy', 'sparse matrix'],
      long_description= 'A sparse matrix library based on Eigen.',
      classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        ],
      ext_modules=ext_modules, 
      cmdclass=cmdclass)
