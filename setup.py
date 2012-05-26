from distutils.core import setup
from distutils.extension import Extension

#Set this to the path of the Eigen library on your system 
include  = ["/usr/include/eigen3/"]

try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True

cmdclass = { }

if use_cython:
    ext_modules=[Extension("sparray.csr_array", ["sparray/csr_array.pyx"], language="c++", include_dirs=include),
                 Extension("sparray.dyn_array", ["sparray/dyn_array.pyx"], language="c++", include_dirs=include)]  
    cmdclass.update({ 'build_ext': build_ext })
else:
    ext_modules = [Extension("sparray.csr_array", [ "sparray/csr_array.cpp" ], include_dirs=include),
                    Extension("sparray.dyn_array", [ "sparray/dyn_array.cpp" ], include_dirs=include)] 

setup(name = 'sparray',
      version = "0.1",
      author = 'Charanpal Dhanjal',
      author_email = 'charanpal@gmail.com',
      packages = ['sparray', 'sparray.test'],
      ext_modules=ext_modules, 
      cmdclass=cmdclass)
