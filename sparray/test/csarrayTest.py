import logging
import sys
import unittest
import numpy
import numpy.testing as nptst 
import math 

from sparray import csarray 

class csarrayTest(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        self.A = csarray((5, 5))

        nrow = 5 
        ncol = 7
        self.B = csarray((nrow, ncol))
        self.B[0, 1] = 1
        self.B[1, 3] = 5.2
        self.B[3, 3] = -0.2
        self.B[0, 6] = -1.23
        self.B[4, 4] = 12.2        
        
        nrow = 100 
        ncol = 100
        self.C = csarray((nrow, ncol))
        self.C[0, 1] = 1
        self.C[10, 3] = 5.2
        self.C[30, 34] = -0.2
        self.C[0, 62] = -1.23
        self.C[4, 41] = 12.2      
        
        self.D = csarray((5, 5))
        self.D[0, 0] = 23.1
        self.D[2, 0] = -3.1
        self.D[3, 0] = -10.0 
        self.D[2, 1] = -5 
        self.D[3, 1] = 5
        
        self.E = csarray((0, 0))
        
        self.F = csarray((6, 6), dtype=numpy.int)
        self.F[0, 0] = 23
        self.F[2, 0] = -3
        self.F[3, 0] = -10 
        self.F[2, 1] = -5 
        self.F[3, 1] = 5
        
    def testInit(self): 
        A = csarray((5, 7))
        self.assertEquals(A.shape, (5, 7))
        
        A = csarray((1, 1))
        self.assertEquals(A.shape, (1, 1))
        
        A = csarray((1, 0))
        self.assertEquals(A.shape, (1, 0))
        
        A = csarray((0, 0))
        self.assertEquals(A.shape, (0, 0))
        
        #Test bad input params 
        self.assertRaises(IndexError, csarray, (0,))
        self.assertRaises(ValueError, csarray, 0)
        
        #TODO: Test other dtypes 
        A = csarray((5, 5))
        self.assertEquals(A.dtype, numpy.float)
        
        self.assertEquals(self.F[0, 0], 23)
        self.F[1, 1] = 51.2        
        self.assertEquals(self.F[1, 1], 51)
        
        #Test assignment with a numpy array 
        A = numpy.array([[3.1, 0, 100], [1.11, 0, 4], [0, 0, 5.2]])
        B = csarray(A)
        
        nptst.assert_array_equal(B.toarray(), A)
        
        B = csarray(A, dtype=numpy.int8)
        nptst.assert_array_equal(B.toarray(), numpy.array(A, numpy.int8))
        
        #Assignment to other csarray 
        B = csarray(self.B, numpy.int)
        
        for i in range(B.shape[0]): 
            for j in range(B.shape[1]): 
                self.assertEquals(B[i, j], int(self.B[i, j]))
                
        F = csarray(self.F, numpy.float)
        F[0, 0] += 0.1
        
        self.assertEquals(F[0, 0], 23.1)
        
        
    def testNDim(self): 
        A = csarray((5, 7))
        self.assertEquals(A.ndim, 2)
        
        A = csarray((0, 0))
        self.assertEquals(A.ndim, 2)
    
    def testSize(self): 
        self.assertEquals(self.A.size, 25)
        self.assertEquals(self.B.size, 35)
        self.assertEquals(self.C.size, 10000)
        self.assertEquals(self.F.size, 36)
        
    def testGetnnz(self): 
       A = csarray((5, 7))
       self.assertEquals(A.getnnz(), 0)
       A[0, 0] = 1.0
       
       self.assertEquals(A.getnnz(), 1)
       
       A[2, 1] = 1.0
       self.assertEquals(A.getnnz(), 2)
       
       A[2, 5] = 1.0
       A[3, 5] = 1.0
       self.assertEquals(A.getnnz(), 4)
       
       #If we insert a zero it is not registered as zero 
       A[4, 4] = 0.0
       self.assertEquals(A.getnnz(), 4)
       
       #But erasing an item keeps it (can call prune)
       A[3, 5] = 0.0
       self.assertEquals(A.getnnz(), 4)
       
       B = csarray((5, 7))
       B[(numpy.array([1, 2, 3]), numpy.array([4, 5, 6]))] = 1
       self.assertEquals(B.getnnz(), 3)
       
       for i in range(5): 
           for j in range(7): 
               B[i, j] = 1
               
       self.assertEquals(B.getnnz(), 35)
       
       self.assertEquals(self.A.getnnz(), 0)
       self.assertEquals(self.B.getnnz(), 5)
       self.assertEquals(self.C.getnnz(), 5)
       self.assertEquals(self.F.getnnz(), 5)
    
    def testSetItem(self):
        nrow = 5 
        ncol = 7
        A = csarray((nrow, ncol))
        A[0, 1] = 1
        A[1, 3] = 5.2
        A[3, 3] = -0.2
        
        self.assertEquals(A[0, 1], 1)
        self.assertAlmostEquals(A[1, 3], 5.2)
        self.assertAlmostEquals(A[3, 3], -0.2)
        
        for i in range(nrow): 
            for j in range(ncol): 
                if (i, j) != (0, 1) and (i, j) != (1, 3) and (i, j) != (3, 3): 
                    self.assertEquals(A[i, j], 0)
        
        self.assertRaises(ValueError, A.__setitem__, (20, 1), 1)  
        self.assertRaises(TypeError, A.__setitem__, (1, 1), "a")   
        self.assertRaises(ValueError, A.__setitem__, (1, 100), 1)   
        self.assertRaises(ValueError, A.__setitem__, (-1, 1), 1)   
        self.assertRaises(ValueError, A.__setitem__, (0, -1), 1) 
        
        result = A[(numpy.array([0, 1, 3]), numpy.array([1, 3, 3]))] 
        self.assertEquals(result[0], 1)
        self.assertEquals(result[1], 5.2)
        self.assertEquals(result[2], -0.2)
        
        #Replace value of A 
        A[0, 1] = 2
        self.assertEquals(A[0, 1], 2)
        self.assertAlmostEquals(A[1, 3], 5.2)
        self.assertAlmostEquals(A[3, 3], -0.2)
        
        for i in range(nrow): 
            for j in range(ncol): 
                if (i, j) != (0, 1) and (i, j) != (1, 3) and (i, j) != (3, 3): 
                    self.assertEquals(A[i, j], 0)
                    
        #Try setting items with arrays 
        A = csarray((nrow, ncol))
        A[numpy.array([0, 1]), numpy.array([2, 3])] = numpy.array([1.2, 2.4])
        
        self.assertEquals(A.getnnz(), 2)
        self.assertEquals(A[0, 2], 1.2)
        self.assertEquals(A[1, 3], 2.4)
        
        A[numpy.array([2, 4]), numpy.array([2, 3])] = 5
        
        self.assertEquals(A[2, 2], 5)
        self.assertEquals(A[4, 3], 5)
       
       
    def testStr(self): 
        nrow = 5 
        ncol = 7
        A = csarray((nrow, ncol))
        A[0, 1] = 1
        A[1, 3] = 5.2
        A[3, 3] = -0.2
        
        outputStr = "csarray dtype:float64 shape:(5, 7) non-zeros:3\n" 
        outputStr += "(0, 1) 1.0\n"
        outputStr += "(1, 3) 5.2\n"
        outputStr += "(3, 3) -0.2"
        self.assertEquals(str(A), outputStr) 
        
        B = csarray((5, 5))
        outputStr = "csarray dtype:float64 shape:(5, 5) non-zeros:0\n" 
        self.assertEquals(str(B), outputStr) 

    def testSum(self): 
        nrow = 5 
        ncol = 7
        A = csarray((nrow, ncol))
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
        
        self.assertEquals(self.A.sum(), 0.0)
        self.assertEquals(self.B.sum(), 16.97)
        self.assertEquals(self.C.sum(), 16.97)
        self.assertAlmostEquals(self.D.sum(), 10)
        
        self.assertEquals(self.F.sum(), 10)
        
        #Test sum along axes 
        nptst.assert_array_equal(self.A.sum(0), numpy.zeros(5))
        nptst.assert_array_equal(self.B.sum(0), numpy.array([0, 1, 0, 5, 12.2, 0, -1.23])) 
        nptst.assert_array_equal(self.D.sum(0), numpy.array([10, 0, 0, 0, 0])) 
        
        nptst.assert_array_equal(self.A.sum(1), numpy.zeros(5))
        nptst.assert_array_almost_equal(self.B.sum(1), numpy.array([-0.23, 5.2, 0, -0.2, 12.2])) 
        nptst.assert_array_equal(self.D.sum(1), numpy.array([23.1, 0, -8.1, -5, 0])) 

    def testGet(self): 
        self.assertEquals(self.B[0, 1], 1)
        self.assertEquals(self.B[1, 3], 5.2)
        self.assertEquals(self.B[3, 3], -0.2)
        self.assertEquals(self.B.getnnz(), 5)
        
        #Test negative indices 
        self.assertEquals(self.B[-5, -6], 1)
        self.assertEquals(self.B[-1, -3], 12.2)
        
        self.assertRaises(ValueError, self.B.__getitem__, (20, 1))  
        self.assertRaises(ValueError, self.B.__getitem__, (1, 20))  
        self.assertRaises(ValueError, self.B.__getitem__, (-6, 1))  
        self.assertRaises(ValueError, self.B.__getitem__, (1, -8))  
        self.assertRaises(TypeError, self.B.__getitem__, (1))
        self.assertRaises(ValueError, self.B.__getitem__, "a")
        self.assertRaises(ValueError, self.B.__getitem__, ("a", "c"))
        
        #Test array indexing using arrays  
        C = self.B[numpy.array([0, 1, 3]), numpy.array([1, 3, 3])]
        self.assertEquals(C.shape[0], 3)
        self.assertEquals(C[0], 1)
        self.assertEquals(C[1], 5.2)
        self.assertEquals(C[2], -0.2)
        
        C = self.A[numpy.array([0, 1, 3]), numpy.array([1, 3, 3])]
        self.assertEquals(C[0], 0)
        self.assertEquals(C[1], 0)
        self.assertEquals(C[2], 0)
        
        A = csarray((2, 2))
        self.assertRaises(ValueError, A.__getitem__, (numpy.array([0, 1]), numpy.array([1, 3])))
        
        A = csarray((2, 2))
        self.assertRaises(ValueError, A.__getitem__, (numpy.array([0, 2]), numpy.array([1, 1])))
        
        A = csarray((0, 0))
        self.assertRaises(ValueError, A.__getitem__, (numpy.array([0, 1, 3]), numpy.array([1, 3, 3])))
        
        
        #Test submatrix indexing 
        C = self.B[:, :]
        
        for i in range(C.shape[0]): 
            for j in range(C.shape[1]): 
                C[i, j] = self.B[i, j]
                
        C = self.B[0:5, 0:7]
        
        for i in range(C.shape[0]): 
            for j in range(C.shape[1]): 
                C[i, j] = self.B[i, j]
        
        C = self.B[numpy.array([0, 1, 3]), :]
        self.assertEquals(C.shape, (3, 7))
        self.assertEquals(C.getnnz(), 4)
        self.assertEquals(C[0, 1], 1)
        self.assertEquals(C[1, 3], 5.2)
        self.assertEquals(C[2, 3], -0.2)
        self.assertEquals(C[0, 6], -1.23)
        
        C = self.B[numpy.array([0, 1, 3]), 0:7]
        self.assertEquals(C.shape, (3, 7))
        self.assertEquals(C.getnnz(), 4)
        self.assertEquals(C[0, 1], 1)
        self.assertEquals(C[1, 3], 5.2)
        self.assertEquals(C[2, 3], -0.2)
        self.assertEquals(C[0, 6], -1.23)
        
        C = self.B[:, numpy.array([3])]
        self.assertEquals(C.shape, (5, 1))
        self.assertEquals(C.getnnz(), 2)
        self.assertEquals(C[1, 0], 5.2)
        self.assertEquals(C[3, 0], -0.2)
        
        self.assertEquals(self.F[0, 0], 23)
                
    def testSubArray(self): 
        rowInds = numpy.array([0, 1], numpy.int)
        colInds = numpy.array([1, 3, 6], numpy.int)
        A = self.B.subArray(rowInds, colInds)
        
        for i in range(A.shape[0]): 
            for j in range(A.shape[1]): 
                self.assertEquals(A[i, j], self.B[rowInds[i], colInds[j]])
                
        #Try all rows/cols 
        rowInds = numpy.arange(5)
        colInds = numpy.arange(7)
        
        A = self.B.subArray(rowInds, colInds)
        
        for i in range(A.shape[0]): 
            for j in range(A.shape[1]): 
                self.assertEquals(A[i, j], self.B[rowInds[i], colInds[j]])
                
        #No rows/cols 
        rowInds = numpy.array([], numpy.int)
        colInds = numpy.array([], numpy.int)
        
        A = self.B.subArray(rowInds, colInds)
        self.assertEquals(A.shape, (0, 0))
        
        A = self.A.subArray(rowInds, colInds)
        self.assertEquals(A.shape, (0, 0))
    

    #@unittest.skip("")          
    def testNonZeroInds(self): 
        
        (rowInds, colInds) = self.B.nonzero()
        
        for i in range(rowInds.shape[0]): 
            self.assertNotEqual(self.B[rowInds[i], colInds[i]], 0)
        
        self.assertEquals(self.B.getnnz(), rowInds.shape[0])
        self.assertEquals(self.B.sum(), self.B[rowInds, colInds].sum())
        
        (rowInds, colInds) = self.C.nonzero()
        
        for i in range(rowInds.shape[0]): 
            self.assertNotEqual(self.C[rowInds[i], colInds[i]], 0)   
            
        self.assertEquals(self.C.getnnz(), rowInds.shape[0])
        self.assertEquals(self.C.sum(), self.C[rowInds, colInds].sum())
        
        (rowInds, colInds) = self.F.nonzero()
        
        for i in range(rowInds.shape[0]): 
            self.assertNotEqual(self.F[rowInds[i], colInds[i]], 0)   
            
        self.assertEquals(self.F.getnnz(), rowInds.shape[0])
        self.assertEquals(self.F.sum(), self.F[rowInds, colInds].sum())
        
        #Try an array with no non zeros 
        nrow = 5 
        ncol = 7
        A = csarray((nrow, ncol))
        (rowInds, colInds) = A.nonzero()
        
        self.assertEquals(A.getnnz(), rowInds.shape[0])
        self.assertEquals(rowInds.shape[0], 0)
        self.assertEquals(colInds.shape[0], 0)
        
        #Zero size array 
        nrow = 0 
        ncol = 0
        A = csarray((nrow, ncol))
        (rowInds, colInds) = A.nonzero()
        self.assertEquals(A.getnnz(), rowInds.shape[0])
        self.assertEquals(rowInds.shape[0], 0)
        self.assertEquals(colInds.shape[0], 0)

    def testDiag(self): 
        nptst.assert_array_equal(self.A.diag(), numpy.zeros(5))
        nptst.assert_array_equal(self.B.diag(), numpy.array([  0,    0,    0,   -0.2,  12.2]))
        nptst.assert_array_equal(self.C.diag(), numpy.zeros(100))
        
        D = csarray((3, 3))
        D[0, 0] = -1
        D[1, 1] = 3.2 
        D[2, 2] = 34 
        
        nptst.assert_array_equal(D.diag(), numpy.array([-1, 3.2, 34]))
        
        E = csarray((0, 0)) 
        nptst.assert_array_equal(E.diag(), numpy.array([]))
        
        nptst.assert_array_equal(self.F.diag(), numpy.array([23, 0,  0,  0,  0, 0]) )

    #@unittest.skip("")
    def testMean(self): 
        self.assertEquals(self.A.mean(), 0)
        
        self.assertAlmostEquals(self.B.mean(), 0.4848571428571428)
        self.assertAlmostEquals(self.C.mean(), 0.001697)
        
        D = csarray((0, 0)) 
        self.assertTrue(math.isnan(D.mean()))
        
        self.assertEquals(self.F.mean(), 10/float(36))
        
        nptst.assert_array_equal(self.A.mean(0), self.A.sum(0)/self.A.shape[0])
        nptst.assert_array_equal(self.B.mean(0), self.B.sum(0)/self.B.shape[0]) 
        nptst.assert_array_equal(self.C.mean(0), self.C.sum(0)/self.C.shape[0])
        nptst.assert_array_equal(self.D.mean(0), self.D.sum(0)/self.D.shape[0])
        nptst.assert_array_equal(self.F.mean(0), self.F.sum(0)/float(self.F.shape[0]))
        
        nptst.assert_array_equal(self.A.mean(1), self.A.sum(1)/self.A.shape[1])
        nptst.assert_array_equal(self.B.mean(1), self.B.sum(1)/self.B.shape[1])
        nptst.assert_array_equal(self.C.mean(1), self.C.sum(1)/self.C.shape[1])
        nptst.assert_array_equal(self.D.mean(1), self.D.sum(1)/self.D.shape[1])
        nptst.assert_array_equal(self.F.mean(1), self.F.sum(1)/float(self.F.shape[1]))
    
    def testCopy(self): 
        A = csarray((5, 5)) 
        A[0, 0] = 1
        A[1, 0] = 2
        A[4, 2] = 3
        self.assertEquals(A[0, 0], 1)
        self.assertEquals(A[1, 0], 2)
        self.assertEquals(A[4, 2], 3)
        
        B = A.copy() 
        A[0, 0] = 2
        A[1, 0] = 3
        A[4, 2] = 4
        A[4, 4] = 5
        
        self.assertEquals(A[0, 0], 2)
        self.assertEquals(A[1, 0], 3)
        self.assertEquals(A[4, 2], 4)   
        self.assertEquals(A[4, 4], 5) 
        self.assertEquals(A.getnnz(), 4)
        
        self.assertEquals(B[0, 0], 1)
        self.assertEquals(B[1, 0], 2)
        self.assertEquals(B[4, 2], 3)
        self.assertEquals(B.getnnz(), 3)
        
        F = self.F.copy() 
        F[0, 0] = -15
        self.assertEquals(F[0, 0], -15)
        self.assertEquals(self.F[0, 0], 23)
        
    def testMultiply(self): 
        val = 2.0 
        C = self.B * val
        
        self.assertEquals(self.B[0, 1], 1)
        self.assertEquals(self.B[1, 3], 5.2)
        self.assertEquals(self.B[3, 3], -0.2)
        self.assertEquals(self.B[0, 6], -1.23)
        self.assertEquals(self.B[4, 4], 12.2)
        
        self.assertEquals(C[0, 1], self.B[0, 1]*val)
        self.assertEquals(C[1, 3], self.B[1, 3]*val)
        self.assertEquals(C[3, 3], self.B[3, 3]*val)
        self.assertEquals(C[0, 6], self.B[0, 6]*val)
        self.assertEquals(C[4, 4], self.B[4, 4]*val)
        
        G = self.F*val 
        self.assertEquals(self.F[0, 0], 23)
        self.assertEquals(G[0, 0], 46)

    def testTrace(self): 
        self.assertEquals(self.A.trace(), 0)
        self.assertEquals(self.B.trace(), 12)
        self.assertEquals(self.C.trace(), 0)
        self.assertEquals(self.D.trace(), 23.1)
        self.assertEquals(self.F.trace(), 23)

    #@unittest.skip("")
    def testToarray(self): 
        A = self.A.toarray()
        self.assertEquals(type(A), numpy.ndarray)
        self.assertEquals(A.shape, self.A.shape)
        self.assertEquals(A.sum(), 0)
        
        B = self.B.toarray()
        self.assertEquals(type(B), numpy.ndarray)
        self.assertEquals(B.shape, self.B.shape)
        self.assertEquals(B[0, 1], 1)
        self.assertEquals(B[1, 3], 5.2)
        self.assertEquals(B[3, 3], -0.2)
        self.assertEquals(B[0, 6], -1.23)
        self.assertEquals(B[4, 4], 12.2)
        self.assertEquals(B.sum(), self.B.sum())
        
        D = self.D.toarray()
        self.assertEquals(type(D), numpy.ndarray)
        self.assertEquals(D.shape, self.D.shape)
        self.assertEquals(D[0, 0], 23.1)
        self.assertEquals(D[2, 0], -3.1)
        self.assertEquals(D[3, 0], -10.0)
        self.assertEquals(D[2, 1], -5)
        self.assertEquals(D[3, 1], 5)
        self.assertAlmostEquals(D.sum(), self.D.sum())
        
        F = self.F.toarray()
        self.assertEquals(type(F), numpy.ndarray)
        self.assertEquals(F.shape, self.F.shape)
        self.assertEquals(F[0, 0], 23)
        self.assertEquals(F[2, 0], -3)
        self.assertEquals(F[3, 0], -10)
        self.assertEquals(F[2, 1], -5)
        self.assertEquals(F[3, 1], 5)
        self.assertAlmostEquals(F.sum(), self.F.sum())
        
       
    def testMin(self):
       self.assertEquals(self.A.min(), 0)
       self.assertEquals(self.B.min(), -1.23)
       self.assertEquals(self.C.min(), -1.23)
       self.assertEquals(self.D.min(), -10)
       self.assertTrue(math.isnan(self.E.min()))
       self.assertEquals(self.F.min(), -10)
       
    def testMax(self):
       self.assertEquals(self.A.max(), 0)
       self.assertEquals(self.B.max(), 12.2)
       self.assertEquals(self.C.max(), 12.2)
       self.assertEquals(self.D.max(), 23.1)
       self.assertTrue(math.isnan(self.E.max()))
       self.assertEquals(self.F.max(), 23)
     
    #@unittest.skip("")
    def testVar(self):
       self.assertEquals(self.A.var(), self.A.toarray().var())
       self.assertAlmostEquals(self.B.var(), self.B.toarray().var())
       self.assertAlmostEquals(self.C.var(), self.C.toarray().var())
       self.assertAlmostEquals(self.D.var(), self.D.toarray().var())
       
       self.assertAlmostEquals(self.F.var(), self.F.toarray().var())
       
    #@unittest.skip("")
    def testStd(self):
       self.assertEquals(self.A.std(), self.A.toarray().std())
       self.assertAlmostEquals(self.B.std(), self.B.toarray().std())
       self.assertAlmostEquals(self.C.std(), self.C.toarray().std())
       self.assertAlmostEquals(self.D.std(), self.D.toarray().std())
       
       self.assertAlmostEquals(self.F.std(), self.F.toarray().std())
          
    def testAbs(self): 
       nptst.assert_array_equal(abs(self.A).toarray(), abs(self.A.toarray()))
       nptst.assert_array_equal(abs(self.B).toarray(), abs(self.B.toarray()))
       nptst.assert_array_equal(abs(self.C).toarray(), abs(self.C.toarray()))
       nptst.assert_array_equal(abs(self.D).toarray(), abs(self.D.toarray()))
       
       nptst.assert_array_equal(abs(self.F).toarray(), abs(self.F.toarray()))
       
    def testNeg(self): 
       nptst.assert_array_equal((-self.A).toarray(), -self.A.toarray())
       nptst.assert_array_equal((-self.B).toarray(), -self.B.toarray())
       nptst.assert_array_equal((-self.C).toarray(), -self.C.toarray())
       nptst.assert_array_equal((-self.D).toarray(), -self.D.toarray())
       
       nptst.assert_array_equal((-self.F).toarray(), -self.F.toarray())
       
    def testAdd(self): 
       #print(self.A.__add__(self.A._array))
       nptst.assert_array_equal((self.A + self.A).toarray(), self.A.toarray()*2)
       nptst.assert_array_equal((self.B + self.B).toarray(), self.B.toarray()*2)
       nptst.assert_array_equal((self.C + self.C).toarray(), self.C.toarray()*2)
       nptst.assert_array_equal((self.D + self.D).toarray(), self.D.toarray()*2)
       
       nptst.assert_array_equal((self.F + self.F).toarray(), self.F.toarray()*2)
       
       A = csarray((5, 5))
       A[0, 1] = 4
       A[1, 3] = 2
       A[3, 3] = 1
       
       B = csarray((5, 5))
       B[0, 2] = 9.2
       B[2, 3] = -5
       B[3, 4] = 12
       
       nptst.assert_array_equal((A + B).toarray(), A.toarray()+B.toarray())
       

    def testSub(self): 
       nptst.assert_array_equal((self.A - self.A).toarray(), self.A.toarray()*0)
       nptst.assert_array_equal((self.B - self.B).toarray(), self.B.toarray()*0)
       nptst.assert_array_equal((self.C - self.C).toarray(), self.C.toarray()*0)
       nptst.assert_array_equal((self.D - self.D).toarray(), self.D.toarray()*0)
       nptst.assert_array_equal((self.F - self.F).toarray(), self.F.toarray()*0)
       
       nptst.assert_array_equal((self.B*2 - self.B).toarray(), self.B.toarray())
       
       A = csarray((5, 5))
       A[0, 1] = 4
       A[1, 3] = 2
       A[3, 3] = 1
       
       B = csarray((5, 5))
       B[0, 2] = 9.2
       B[2, 3] = -5
       B[3, 4] = 12
       
       nptst.assert_array_equal((A - B).toarray(), A.toarray()-B.toarray())
       

    def testHadamard(self): 
       nptst.assert_array_equal((self.A.hadamard(self.A)).toarray(), (self.A.toarray())**2)
       nptst.assert_array_equal((self.B.hadamard(self.B)).toarray(), self.B.toarray()**2)
       nptst.assert_array_equal((self.C.hadamard(self.C)).toarray(), self.C.toarray()**2)
       nptst.assert_array_equal((self.D.hadamard(self.D)).toarray(), self.D.toarray()**2)
       
       nptst.assert_array_equal((self.F.hadamard(self.F)).toarray(), self.F.toarray()**2)
       
       A = csarray((5, 5))
       A[0, 1] = 4
       A[2, 3] = -1.2
       A[1, 3] = 2
       A[3, 3] = 1
       
       B = csarray((5, 5))
       B[0, 2] = 9.2
       B[2, 3] = -5
       B[3, 4] = 12
       B[3, 3] = 12
       
       C = csarray((5, 5))
       
       nptst.assert_array_equal((A.hadamard(B)).toarray(), A.toarray()*B.toarray())
       nptst.assert_array_equal((A.hadamard(C)).toarray(), C.toarray())
 

    def testReserve(self): 
       A = csarray((5, 5))
       A.reserve(5)
       A[0, 1] = 4
       A[2, 3] = -1.2
       A[1, 3] = 2
       A[3, 3] = 1
      
    def testCompress(self): 
       A = csarray((5, 5))
       A[0, 1] = 4
       A[2, 3] = -1.2
       A[1, 3] = 2
       A[3, 3] = 1
       A.compress()
       
    def testDot(self): 
       A = csarray((5, 5))
       A[0, 1] = 4
       A[2, 3] = -1.2
       A[1, 3] = 2
       A[3, 3] = 1 
       
       B = A.dot(A)
       nptst.assert_array_equal(B.toarray(), A.toarray().dot(A.toarray()))
       
       B = self.D.dot(self.D)
       nptst.assert_array_equal(B.toarray(), self.D.toarray().dot(self.D.toarray()))
       
       C = csarray((5, 2))
       for i in range(5): 
           for j in range(2): 
               C[i, j] = 1
               
       self.assertRaises(ValueError, C.dot, C)
       B = A.dot(C)
       nptst.assert_array_equal(B.toarray(), A.toarray().dot(C.toarray()))        
       

    def testTranspose(self): 
        
       A = csarray((5, 5))
       A[0, 1] = 4
       A[2, 3] = -1.2
       A[1, 3] = 2
       A[3, 3] = 1 
       
       self.assertEquals(type(A.T), csarray)
       
       nptst.assert_array_equal(A.transpose().toarray(), A.toarray().T) 
       nptst.assert_array_equal(self.A.T.toarray(), self.A.toarray().T) 
       nptst.assert_array_equal(self.B.T.toarray(), self.B.toarray().T)
       nptst.assert_array_equal(self.C.transpose().toarray(), self.C.toarray().T)
       nptst.assert_array_equal(self.D.transpose().toarray(), self.D.toarray().T)
       nptst.assert_array_equal(self.E.transpose().toarray(), self.E.toarray().T)
       nptst.assert_array_equal(self.F.transpose().toarray(), self.F.toarray().T)

if __name__ == "__main__":
    unittest.main()
    
    