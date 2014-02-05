import logging
import sys
import tempfile 
import unittest
import numpy
import numpy.testing as nptst 
import sppy
import sppy.io

class utilTest(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        numpy.random.seed(21)

    def testWrite(self): 
        m = 10 
        n = 5 
        density = 0.5
        A = sppy.rand((m, n), density)
        
        sppy.io.mmwrite(tempfile.tempdir + "/test.mtx", A)
        
        #print(A)
        #print("File written as " + tempfile.tempdir + "/test.mtx")
   

    def testRead(self): 
        m = 10 
        n = 5 
        density = 0.5
        A = sppy.rand((m, n), density)
        
        fileName = tempfile.tempdir + "/test.mtx"
        sppy.io.mmwrite(fileName, A)
        
        B = sppy.io.mmread(fileName)        
        
        nptst.assert_array_almost_equal(A.toarray(), B.toarray(), 5)
        self.assertEquals(A.dtype, B.dtype)
        
        #Now try an integer array 
        A = sppy.csarray((m,n), dtype=numpy.int)
        A[0, 1] = 5 
        A[2, 2] = -2
        A[6, 3] = 1
        
        sppy.io.mmwrite(fileName, A)
        B = sppy.io.mmread(fileName) 
        nptst.assert_array_almost_equal(A.toarray(), B.toarray(), 5)
        self.assertEquals(A.dtype, B.dtype)
   

if __name__ == "__main__":
    unittest.main()