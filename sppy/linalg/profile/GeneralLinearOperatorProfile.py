

import numpy
import logging
import sys
import scipy.sparse.linalg
from apgl.util.Sampling import Sampling 
from apgl.util.ProfileUtils import ProfileUtils
from exp.util.SparseUtils import SparseUtils
from exp.util.LinOperatorUtils import LinOperatorUtils
from sppy import csarray
from sppy.linalg import GeneralLinearOperator 

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

class GeneralLinearOperatorProfile(object):
    def __init__(self):
        numpy.random.seed(21)        
        
        #Create a low rank matrix  
        n = 500000 
        m = 500000 
        self.r = 200 
        k = 10**6
        
        print("Generating low rank")
        self.X = SparseUtils.generateSparseLowRank((n, m), self.r, k)
        print("Generating csarray")
        self.X = csarray.fromScipySparse(self.X, storageType="rowMajor")
        print("Done")
     
    def profilePowerIteration(self): 
                
        p = 200 
        q = 5
        omega = numpy.random.randn(self.X.shape[1], p)
        L = GeneralLinearOperator.asLinearOperator(self.X)
        
        def run(): 
            Y = L.matmat(omega)

            for i in range(q):
                Y = L.rmatmat(Y)
                Y = L.matmat(Y)
                
        ProfileUtils.profile('run()', globals(), locals())
        
    def profilePowerIteration2(self): 
                
        p = 100 
        q = 5
        omega = numpy.random.randn(self.X.shape[1], p)
        L = GeneralLinearOperator.asLinearOperator(self.X, parallel=True)
        
        def run(): 
            Y = L.matmat(omega)

            for i in range(q):
                Y = L.rmatmat(Y)
                Y = L.matmat(Y)
                
        ProfileUtils.profile('run()', globals(), locals())
     
        
if __name__ == '__main__':     
    profiler = GeneralLinearOperatorProfile()
    profiler.profilePowerIteration() 
    profiler.profilePowerIteration2() 