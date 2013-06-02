import logging
import sys
import numpy
import scipy.sparse
from apgl.util import *
from sppy import csarray
from pysparse import spmatrix
from apgl.util.PySparseUtils import PySparseUtils 

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
numpy.random.seed(21)


class csarrayProfile():
    def __init__(self):
        #take some random indices 
        self.N = 5000 
        self.M = 100 
        
        self.rowInds = numpy.array(numpy.random.randint(0, self.M, self.N), dtype=numpy.int)
        self.colInds = numpy.array(numpy.random.randint(0, self.M, self.N), dtype=numpy.int)
        
        self.val = numpy.random.rand() 
        
        self.k = 1000
    
    def profilePut(self):
        self.rowInds = self.rowInds.tolist()
        self.colInds = self.colInds.tolist()        
        
        def runPut(): 
            
            for i in range(self.k):         
                A = csarray((self.N, self.N))
                #A[(self.rowInds, self.colInds)] = self.val 
                A.put(self.val, self.rowInds, self.colInds)
        
        ProfileUtils.profile('runPut()', globals(), locals())
        
    def profilePut2(self):
        def runPut(): 
            
            for i in range(self.k):         
                A = csarray((self.N, self.N))
                #A[(self.rowInds, self.colInds)] = self.val 
                A.put(self.val, self.rowInds, self.colInds)
        
        ProfileUtils.profile('runPut()', globals(), locals())
        
    def profilePutPySparse(self): 
        
        def runPut(): 
            A = spmatrix.ll_mat(self.N, self.N)
            for i in range(self.k):         
                A.put(self.val, self.rowInds, self.colInds)
        
        ProfileUtils.profile('runPut()', globals(), locals())
        
    def profileSlicePys(self): 
        A = spmatrix.ll_mat(self.N, self.N)  
        A.put(self.val, self.rowInds, self.colInds)
        
        def runSlice():     
            for i in range(10):  
                sliceInds = numpy.array(numpy.random.randint(0, self.M, self.N), dtype=numpy.int32)
                B = A[:, sliceInds]
            
        ProfileUtils.profile('runSlice()', globals(), locals())
        
    def profileSliceSpa(self): 
        A = csarray((self.N, self.N))
        A.put(self.val, self.rowInds, self.colInds)
        
        def runSlice():     
            for i in range(10):  
                sliceInds = numpy.array(numpy.random.randint(0, self.M, self.N), dtype=numpy.int)
                B = A[:, sliceInds]
            
        ProfileUtils.profile('runSlice()', globals(), locals())
        
    def profileSumPys(self): 
        A = spmatrix.ll_mat(self.N, self.N)  
        A.put(self.val, self.rowInds, self.colInds)
        
        def runSum():     
            for i in range(1000):  
                 i = PySparseUtils.sum(A)
            print(i)
            
        ProfileUtils.profile('runSum()', globals(), locals())
        
    def profileSumSpa(self): 
        A = csarray((self.N, self.N))
        A.put(self.val, self.rowInds, self.colInds)
        
        def runSum():     
            for i in range(1000):  
                 i = A.sum()
            print(i)
            
        ProfileUtils.profile('runSum()', globals(), locals())
        
    def profileGetNonZerosPys(self): 
        A = spmatrix.ll_mat(self.N, self.N)  
        A.put(self.val, self.rowInds, self.colInds)
        
        def runNonZeros(): 
            for i in range(1000):
                (rows, cols) = PySparseUtils.nonzero(A)
                nzVals = numpy.zeros(len(rows))
                A.take(nzVals, rows, cols)
            
        ProfileUtils.profile('runNonZeros()', globals(), locals())

    def profileGetNonZerosSpa(self): 
        A = csarray((self.N, self.N)) 
        A.put(self.val, self.rowInds, self.colInds)
        
        def runNonZeros(): 
            for i in range(1000):
                rows, cols = A.nonzero()
                vals = A[rows, cols]
            print(numpy.sum(vals))
            
        ProfileUtils.profile('runNonZeros()', globals(), locals())
        
    def profilePutSorted(self): 
        #Test speed of array creation 
        numpy.random.seed(21)
        m = 10000
        n = 20000      
        numInds = 100000
        
        inds = numpy.random.randint(0, m*n, numInds)
        inds = numpy.unique(inds)
        vals = numpy.random.randn(inds.shape[0])
        
        rowInds, colInds = numpy.unravel_index(inds, (m, n), order="FORTRAN")
                
        A = csarray((m, n), storageType="colMajor")
        
        vectorNnz = numpy.bincount(colInds, minlength=A.shape[1])

        #ProfileUtils.profile('A.put(vals, rowInds, colInds)', globals(), locals())
        #ProfileUtils.profile('A.putSorted(vals, rowInds, colInds)', globals(), locals())

        ProfileUtils.profile("scipy.sparse.csc_matrix((vals, (rowInds, colInds)), A.shape )", globals(), locals())


profiler = csarrayProfile()
#profiler.profilePut()
#profiler.profileSlicePys()
#profiler.profileSliceSpa()
#profiler.profileSumPys()
#profiler.profileSumSpa()
#profiler.profileGetNonZerosPys()
#profiler.profileGetNonZerosSpa()
profiler.profilePutSorted()
