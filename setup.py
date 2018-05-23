from distutils.core import setup
from Cython.Distutils import build_ext 
from distutils.extension import Extension
import numpy

setup(
    cmdclass = {'build_ext': build_ext},
    include_dirs = [numpy.get_include()], 
    ext_modules = [Extension('skipgram_data', ['skipgram_data.pyx'],
                    include_dirs=[numpy.get_include()]),
                Extension('wordnet_data', ['wordnet_data.pyx'],
                    include_dirs=[numpy.get_include()]),
                Extension('sampling', ['sampling.pyx'],
                    include_dirs=[numpy.get_include()]),]
)
