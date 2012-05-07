import logging
import sys
import numpy
from apgl.util import *
from sparray.csr_array import csr_array 
from pysparse import spmatrix

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
numpy.random.seed(21)


class csr_arrayProfile():
    def __init__(self):
        #take some random indices 
        self.N = 5000 
        M = 100 
        
        self.rowInds = numpy.random.randint(0, M, self.N)
        self.colInds = numpy.random.randint(0, M, self.N)
        self.val = numpy.random.rand() 
        
        self.k = 1000
    
    def profilePut(self):
        def runPut(): 
            A = csr_array((self.N, self.N))
            for i in range(self.k):         
                
                #A[(self.rowInds, self.colInds)] = self.val 
                A.put(self.val, self.rowInds, self.colInds)
        
        ProfileUtils.profile('runPut()', globals(), locals())
        
    def profilePutPySparse(self): 
        
        def runPut(): 
            A = spmatrix.ll_mat(self.N, self.N)
            for i in range(self.k):         
                A.put(self.val, self.rowInds, self.colInds)
        
        ProfileUtils.profile('runPut()', globals(), locals())
    
profiler = csr_arrayProfile()
profiler.profilePut()
#profiler.profilePutPySparse()