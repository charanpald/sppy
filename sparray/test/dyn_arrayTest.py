import logging
import sys
import unittest
import numpy


from sparray.dyn_array import dyn_array 

class dyn_arrayTest(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        
        self.A = dyn_array((5, 5))
        #self.A[0, 1] = 1 
        #self.A[1, 1] = 2 
        #self.A[2, 1] = 5 
        
        
        nrow = 5 
        ncol = 7
        self.B = dyn_array((nrow, ncol))
        self.B[0, 1] = 1
        self.B[1, 3] = 5.2
        self.B[3, 3] = -0.2
        self.B[0, 6] = -1.23
        self.B[4, 4] = 12.2        
        
    def testInit(self): 
        A = dyn_array((5, 7))
        #A[0, 1] = 1 
        
        self.assertEquals(A.shape, (5, 7))
        
        A = dyn_array((1, 1))
        self.assertEquals(A.shape, (1, 1))
        
        A = dyn_array((1, 0))
        self.assertEquals(A.shape, (1, 0))
        
        A = dyn_array((0, 0))
        self.assertEquals(A.shape, (0, 0))
        
        #Test dtypes 
        A = dyn_array((5, 5))
        A[0, 0] = 1.5 
        
        
    def testNDim(self): 
        A = dyn_array((5, 7))
        self.assertEquals(A.ndim, 2)
        
    def testGetnnz(self): 
       A = dyn_array((5, 7))
       self.assertEquals(A.getnnz(), 0)
       A[0, 0] = 1.0
       
       self.assertEquals(A.getnnz(), 1)
       
       A[2, 1] = 1.0
       self.assertEquals(A.getnnz(), 2)
       
       A[2, 5] = 1.0
       A[3, 5] = 1.0
       self.assertEquals(A.getnnz(), 4)
       
       A[4, 4] = 0.0
       self.assertEquals(A.getnnz(), 5)
       
       B = dyn_array((5, 7))
       B[(numpy.array([1, 2, 3]), numpy.array([4, 5, 6]))] = 1
       self.assertEquals(B.getnnz(), 3)
    
    def testSetItem(self):
        nrow = 5 
        ncol = 7
        A = dyn_array((nrow, ncol))
        A[0, 1] = 1
        A[1, 3] = 5.2
        A[3, 3] = -0.2
        
        self.assertEquals(A[0, 1], 1)
        self.assertAlmostEquals(A[1, 3], 5.2, 5)
        self.assertAlmostEquals(A[3, 3], -0.2)
        
        for i in range(nrow): 
            for j in range(ncol): 
                if (i, j) != (0, 1) and (i, j) != (1, 3) and (i, j) != (3, 3): 
                    self.assertEquals(A[i, j], 0)
        
        try: 
            A[20, 0] = 1
            self.fail()
        except ValueError: 
            pass 
        
        try: 
            A[0, 20] = 1
            self.fail()
        except ValueError: 
            pass 
        
        result = A[(numpy.array([0, 1, 3]), numpy.array([1, 3, 3]))] 
        self.assertEquals(result[0], 1)
        self.assertEquals(result[1], 5.2)
        self.assertEquals(result[2], -0.2)
        
        #Replace value of A 
        A[0, 1] = 2
        self.assertEquals(A[0, 1], 2)
        self.assertAlmostEquals(A[1, 3], 5.2, 5)
        self.assertAlmostEquals(A[3, 3], -0.2)
        
        for i in range(nrow): 
            for j in range(ncol): 
                if (i, j) != (0, 1) and (i, j) != (1, 3) and (i, j) != (3, 3): 
                    self.assertEquals(A[i, j], 0)
       
       
    def testStr(self): 
        nrow = 5 
        ncol = 7
        A = dyn_array((nrow, ncol))
        A[0, 1] = 1
        A[1, 3] = 5.2
        A[3, 3] = -0.2
        
        #print(A)

    def testSum(self): 
        nrow = 5 
        ncol = 7
        A = dyn_array((nrow, ncol))
        A[0, 1] = 1
        A[1, 3] = 5.2
        A[3, 3] = -0.2
        
        self.assertEquals(A.sum(), 6.0)
        
        A[3, 4] = -1.2
        self.assertEquals(A.sum(), 4.8)
        
        A[0, 0] = 1.34
        self.assertEquals(A.sum(), 6.14)
        
        A[0, 0] = 0 
        self.assertEquals(A.sum(), 4.8)

    def testGet(self): 
        self.assertEquals(self.B[0, 1], 1)
        self.assertEquals(self.B[1, 3], 5.2)
        self.assertEquals(self.B[3, 3], -0.2)
        self.assertEquals(self.B.getnnz(), 5)
        
        #self.B
        C = self.B[numpy.array([0, 1, 3]), numpy.array([1, 3, 3])]
        self.assertEquals(C.shape[0], 3)
        self.assertEquals(C[0], 1)
        self.assertEquals(C[1], 5.2)
        self.assertEquals(C[2], -0.2)
        
        C = self.B[numpy.array([0, 1, 3]), :]
        self.assertEquals(C.shape, (3, 7))
        self.assertEquals(C.getnnz(), 4)
        self.assertEquals(C[0, 1], 1)
        self.assertEquals(C[1, 3], 5.2)
        self.assertEquals(C[2, 3], -0.2)
        self.assertEquals(C[0, 6], -1.23)
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    
    