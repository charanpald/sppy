#from distutils.core import setup
import numpy 
from setuptools import setup
from distutils.extension import Extension

try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True

def execfile3(file, globals=globals(), locals=locals()):
    with open(file, "r") as fh:
        exec(fh.read()+"\n", globals, locals)

execfile3('sppy/version.py')

cmdclass = { }

if use_cython:
    # Expand cdef template and generate _sub.pyx files:
    import sys
    sys.path.append('./sppy/') # Circumvents loading __init__.py before csarray module is built
    import TemplateExpander
    TemplateExpander.expand_base(workdir='./sppy')
    ext_modules=[Extension("sppy.csarray", ["sppy/csarray.pyx"], language="c++", include_dirs=[numpy.get_include()], extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp'])] 
    ext_modules.append(Extension("sppy.csarray_sub", ["sppy/csarray_sub.pyx"], language="c++", include_dirs=[numpy.get_include()], extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp'])) 
    ext_modules.append(Extension("sppy.csarray1d_sub", ["sppy/csarray1d_sub.pyx"], language="c++", include_dirs=[numpy.get_include()])) 
    cmdclass.update({ 'build_ext': build_ext })
else:
    ext_modules = [Extension("sppy.csarray", [ "sppy/csarray.cpp" ], include_dirs=[numpy.get_include()]), Extension("sppy.csarray_sub", ["sppy/csarray_sub.cpp"], include_dirs=[numpy.get_include()]), Extension("sppy.csarray1d_sub", ["sppy/csarray1d_sub.cpp"], include_dirs=[numpy.get_include()])] 

descriptionFile = open("Description.rst")
description = "".join(descriptionFile.readlines()) 
descriptionFile.close()


setup(name = 'sppy',
      version = __version__,
      author = 'Charanpal Dhanjal',
      author_email = 'charanpal@gmail.com',
      packages = ['sppy', 'sppy.test', 'sppy.linalg', 'sppy.linalg.test', "sppy.lib", 'sppy.lib.test'],
      package_data = {'': ['*.pyx', '*.pxd']},
      install_requires=['numpy>=1.5.0'],
      url = 'http://packages.python.org/sppy/',
      license = 'GNU Library or Lesser General Public License (LGPL)',
      platforms=["OS Independent"],
      keywords=['numpy', 'sparse matrix'],
      long_description= description,
      classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        ],
      ext_modules=ext_modules, 
      cmdclass=cmdclass)
