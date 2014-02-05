import logging
import sys
import numpy
import scipy.sparse
from apgl.util import *
from sppy.io import mmread

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
numpy.random.seed(21)


class ioProfile():
    def __init__(self):
        numpy.random.seed(21)
        
    def profileMmread(self): 
        matrixFileName = PathDefaults.getDataDir() + "reference/authorDocumentMatrix.mtx"
            
        ProfileUtils.profile('mmread(matrixFileName)', globals(), locals())
        
profiler = ioProfile() 
profiler.profileMmread()