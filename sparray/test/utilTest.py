import logging
import sys
import unittest
import numpy
import numpy.testing as nptst 
import math 

import sparray 

class utilTest(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    def testDiag(self): 
        
        a = numpy.array([1, 2, 3])
        
        X = sparray.diag(a)
        self.assertEquals(X.getnnz(), 3)
        self.assertEquals(X[0, 0], 1)
        self.assertEquals(X[1, 1], 2)
        self.assertEquals(X[2, 2], 3)


    def testEye(self): 
        X = sparray.eye(10)
        
        for i in range(10): 
            self.assertEquals(X[i, i], 1)
            
        self.assertEquals(X.dtype, numpy.float)
        
        X = sparray.eye(10, numpy.int)
        for i in range(10): 
            self.assertEquals(X[i, i], 1)
            
        self.assertEquals(X.dtype, numpy.int)
        
    def testRand(self): 
        X = sparray.rand(10, 10, 0.1)

        self.assertEquals(X.shape, (10, 10))
        self.assertAlmostEqual(X.getnnz()/float(100), 0.1, 1)        
        

if __name__ == "__main__":
    unittest.main()