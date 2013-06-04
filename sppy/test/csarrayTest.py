import logging
import sys
import unittest
import numpy
import numpy.testing as nptst 
import math 

from sppy import csarray 

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
        
        self.G = csarray((6, 6), storageType="rowMajor")
        self.G[0, 0] = 23
        self.G[2, 0] = -3
        self.G[3, 0] = -10 
        self.G[2, 1] = -5 
        self.G[3, 1] = 5
        
        self.H = csarray((5, 7), storageType="rowMajor")
        self.H[0, 1] = 1
        self.H[1, 3] = 5.2
        self.H[3, 3] = -0.2
        self.H[0, 6] = -1.23
        self.H[4, 4] = 12.2   
        
        self.I = csarray((5, 5), storageType="rowMajor") 
        
        self.a = csarray(10, dtype=numpy.float)
        self.a[0] = 23 
        self.a[3] = 1.2
        self.a[4] = -8
        
        self.b = csarray(10, dtype=numpy.int)
        self.b[0] = 23 
        self.b[5] = 1
        self.b[8] = -8
        
        self.c = csarray((3, ), dtype=numpy.float)
        
        self.d = csarray((0, ), dtype=numpy.float)
        
        self.storageTypes = ["rowMajor", "colMajor"]
        
    def testInit(self): 
        A = csarray((5, 7))
        self.assertEquals(A.shape, (5, 7))
        
        A = csarray((1, 1))
        self.assertEquals(A.shape, (1, 1))
        
        A = csarray((1, 0))
        self.assertEquals(A.shape, (1, 0))
        
        A = csarray((0, 0))
        self.assertEquals(A.shape, (0, 0))
        
        a = csarray((5))
        self.assertEquals(a.shape, (5,))
        
        a = csarray(0)
        self.assertEquals(a.shape, (0,))
        
        #Test bad input params 
        self.assertRaises(ValueError, csarray, (0,1,2))
        self.assertRaises(ValueError, csarray, "a")
        self.assertRaises(ValueError, csarray, (5, 5), numpy.float, "abc")
        
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
        
        A = numpy.array([3.1, 0, 100])
        B = csarray(A)
        
        nptst.assert_array_equal(B.toarray(), A)
        
        B = csarray(A, dtype=numpy.int8)
        nptst.assert_array_equal(B.toarray(), numpy.array(A, numpy.int8))
        
        B = csarray(A, dtype=numpy.int8, storageType="rowMajor")
        nptst.assert_array_equal(B.toarray(), numpy.array(A, numpy.int8))
        
        #Assignment to other csarray 
        B = csarray(self.B, numpy.int)
        
        for i in range(B.shape[0]): 
            for j in range(B.shape[1]): 
                self.assertEquals(B[i, j], int(self.B[i, j]))
         
        B = csarray(self.B, numpy.int, storageType="rowMajor")
        
        for i in range(B.shape[0]): 
            for j in range(B.shape[1]): 
                self.assertEquals(B[i, j], int(self.B[i, j]))         
         
        F = csarray(self.F, numpy.float)
        F[0, 0] += 0.1
        
        self.assertEquals(F[0, 0], 23.1)
        
        #This doesn't work as we can't instantiate using an array 
        #b = csarray(self.b, numpy.int)
        
        #for i in range(b.shape[0]): 
        #    self.assertEquals(b[i], int(self.b[i]))
        
    def testNDim(self): 
        A = csarray((5, 7))
        self.assertEquals(A.ndim, 2)
        
        A = csarray((5, 7), storageType="rowMajor")
        self.assertEquals(A.ndim, 2)        
        
        A = csarray((0, 0))
        self.assertEquals(A.ndim, 2)
        
        self.assertEquals(self.a.ndim, 1)
        self.assertEquals(self.b.ndim, 1)
    
    def testSize(self): 
        self.assertEquals(self.A.size, 25)
        self.assertEquals(self.B.size, 35)
        self.assertEquals(self.C.size, 10000)
        self.assertEquals(self.F.size, 36)
        self.assertEquals(self.G.size, 36)
        
        self.assertEquals(self.a.size, 10)
        self.assertEquals(self.b.size, 10)
        self.assertEquals(self.c.size, 3)
        
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
       self.assertEquals(self.G.getnnz(), 5)
       
       self.assertEquals(self.a.getnnz(), 3)
       self.assertEquals(self.b.getnnz(), 3)
       self.assertEquals(self.c.getnnz(), 0)
    
    def testSetItem(self):
        nrow = 5 
        ncol = 7
    
        storageTypes = ["colMajor", "rowMajor"]        
        
        for storageType in storageTypes: 
            A = csarray((nrow, ncol), storageType=storageType)
            A[0, 1] = 1
            A[1, 3] = 5.2
            A[3, 3] = -0.2
            
            self.assertEquals(A[0, 1], 1)
            self.assertAlmostEquals(A[1, 3], 5.2)
            self.assertAlmostEquals(A[3, 3], -0.2)
            
            a = csarray(nrow)
            a[0] = 1
            a[1] = 5.2
            a[3] = -0.2
            
            self.assertEquals(a[0], 1)
            self.assertAlmostEquals(a[1], 5.2)
            self.assertAlmostEquals(a[3], -0.2)
            
            for i in range(nrow): 
                for j in range(ncol): 
                    if (i, j) != (0, 1) and (i, j) != (1, 3) and (i, j) != (3, 3): 
                        self.assertEquals(A[i, j], 0)
            
            self.assertRaises(ValueError, A.__setitem__, (20, 1), 1)  
            self.assertRaises(TypeError, A.__setitem__, (1, 1), "a")   
            self.assertRaises(ValueError, A.__setitem__, (1, 100), 1)   
            self.assertRaises(ValueError, A.__setitem__, (-1, 1), 1)   
            self.assertRaises(ValueError, A.__setitem__, (0, -1), 1) 
            self.assertRaises(ValueError, a.__setitem__, (0, 0), 1) 
            
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
                        
            a[0] = 10
            self.assertEquals(a[0], 10)
                        
            #Try setting items with arrays 
            A = csarray((nrow, ncol), storageType=storageType)
            A[numpy.array([0, 1]), numpy.array([2, 3])] = numpy.array([1.2, 2.4])
            
            self.assertEquals(A.getnnz(), 2)
            self.assertEquals(A[0, 2], 1.2)
            self.assertEquals(A[1, 3], 2.4)
            
            A[numpy.array([2, 4]), numpy.array([2, 3])] = 5
            
            self.assertEquals(A[2, 2], 5)
            self.assertEquals(A[4, 3], 5)
            
            a = csarray(nrow, storageType=storageType)
            a[numpy.array([0, 2])] = numpy.array([1.2, 2.4])
            self.assertEquals(a.getnnz(), 2)
            self.assertEquals(a[0], 1.2)
            self.assertEquals(a[2], 2.4)
            
    def testStr(self): 
        nrow = 5 
        ncol = 7
        
        storageTypes = ["colMajor", "rowMajor"]        
        
        for storageType in storageTypes:         
            A = csarray((nrow, ncol), storageType=storageType)
            A[0, 1] = 1
            A[1, 3] = 5.2
            A[3, 3] = -0.2
            
            outputStr = "csarray dtype:float64 shape:(5, 7) non-zeros:3\n" 
            outputStr += "(0, 1) 1.0\n"
            outputStr += "(1, 3) 5.2\n"
            outputStr += "(3, 3) -0.2"
            self.assertEquals(str(A), outputStr) 
            
            B = csarray((5, 5), storageType=storageType)
            outputStr = "csarray dtype:float64 shape:(5, 5) non-zeros:0\n" 
            self.assertEquals(str(B), outputStr) 
            
            outputStr = "csarray dtype:float64 shape:(10,) non-zeros:3\n"
            outputStr +="(0) 23.0\n"
            outputStr +="(3) 1.2\n"
            outputStr +="(4) -8.0"
            self.assertEquals(str(self.a), outputStr) 
            
            outputStr = "csarray dtype:float64 shape:(3,) non-zeros:0\n" 
            self.assertEquals(str(self.c), outputStr) 

    def testSum(self): 
        nrow = 5 
        ncol = 7

        storageTypes = ["colMajor", "rowMajor"]         
        
        for storageType in storageTypes:   
            A = csarray((nrow, ncol), storageType=storageType)
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
        self.assertEquals(self.G.sum(), 10)
        
        self.assertEquals(self.a.sum(), 16.2)
        self.assertEquals(self.b.sum(), 16)
        self.assertEquals(self.c.sum(), 0)
        
        #Test sum along axes 
        nptst.assert_array_equal(self.A.sum(0), numpy.zeros(5))
        nptst.assert_array_equal(self.B.sum(0), numpy.array([0, 1, 0, 5, 12.2, 0, -1.23])) 
        nptst.assert_array_equal(self.D.sum(0), numpy.array([10, 0, 0, 0, 0])) 
        nptst.assert_array_equal(self.F.sum(0), self.G.sum(0)) 
        
        nptst.assert_array_equal(self.A.sum(1), numpy.zeros(5))
        nptst.assert_array_almost_equal(self.B.sum(1), numpy.array([-0.23, 5.2, 0, -0.2, 12.2])) 
        nptst.assert_array_equal(self.D.sum(1), numpy.array([23.1, 0, -8.1, -5, 0])) 
        nptst.assert_array_equal(self.F.sum(1), self.G.sum(1)) 

    def testGet(self): 
        self.assertEquals(self.B[0, 1], 1)
        self.assertEquals(self.B[1, 3], 5.2)
        self.assertEquals(self.B[3, 3], -0.2)
        self.assertEquals(self.B.getnnz(), 5)
        
        self.assertEquals(self.G[0, 0], 23)
        self.assertEquals(self.G[2, 0], -3)
        
        self.assertEquals(self.a[0], 23)
        self.assertEquals(self.a[3], 1.2)
        self.assertEquals(self.a[4], -8)
        self.assertEquals(self.a.getnnz(), 3)
        
        #Test negative indices 
        self.assertEquals(self.B[-5, -6], 1)
        self.assertEquals(self.B[-1, -3], 12.2)
        self.assertEquals(self.G[-3, -5], 5)
        
        self.assertEquals(self.b[-2], -8)
        
        self.assertRaises(ValueError, self.B.__getitem__, (20, 1))  
        self.assertRaises(ValueError, self.B.__getitem__, (1, 20))  
        self.assertRaises(ValueError, self.B.__getitem__, (-6, 1))  
        self.assertRaises(ValueError, self.B.__getitem__, (1, -8))  
        self.assertRaises(TypeError, self.B.__getitem__, (1))
        self.assertRaises(ValueError, self.B.__getitem__, "a")
        self.assertRaises(Exception, self.B.__getitem__, ("a", "c"))
        
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
        
        C = self.G[numpy.array([0, 2, 3]), numpy.array([0, 0, 0])]
        self.assertEquals(C[0], 23)
        self.assertEquals(C[1], -3)
        self.assertEquals(C[2], -10)
        
        d = self.a[numpy.array([0, 3, 4])]
        self.assertEquals(d[0], 23)
        self.assertEquals(d[1], 1.2)
        self.assertEquals(d[2], -8)
        
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
                
        C = self.G[0:3, 0:4]
        
        for i in range(C.shape[0]): 
            for j in range(C.shape[1]): 
                C[i, j] = self.G[i, j]
        
        C = self.B[numpy.array([0, 1, 3]), :]
        self.assertEquals(C.shape, (3, 7))
        self.assertEquals(C.getnnz(), 4)
        self.assertEquals(C[0, 1], 1)
        self.assertEquals(C[1, 3], 5.2)
        self.assertEquals(C[2, 3], -0.2)
        self.assertEquals(C[0, 6], -1.23)
        
        C = self.H[numpy.array([0, 1, 3]), :]
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
        
        C = self.H[numpy.array([0, 1, 3]), :]
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
        
        C = self.H[:, numpy.array([3])]
        self.assertEquals(C.shape, (5, 1))
        self.assertEquals(C.getnnz(), 2)
        self.assertEquals(C[1, 0], 5.2)
        self.assertEquals(C[3, 0], -0.2)
        
        self.assertEquals(self.F[0, 0], 23)
        
        d = self.a[0:4]
        self.assertEquals(d.shape, (4, ))
        self.assertEquals(d[0], 23)
        self.assertEquals(d[3], 1.2)
                
    def testSubArray(self): 
        rowInds = numpy.array([0, 1], numpy.int)
        colInds = numpy.array([1, 3, 6], numpy.int)
        A = self.B.subArray(rowInds, colInds)
        
        for i in range(A.shape[0]): 
            for j in range(A.shape[1]): 
                self.assertEquals(A[i, j], self.B[rowInds[i], colInds[j]])
                
        A = self.H.subArray(rowInds, colInds)
        
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
                
        A = self.H.subArray(rowInds, colInds)
        
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
        
        (rowInds, colInds) = self.G.nonzero()
        
        for i in range(rowInds.shape[0]): 
            self.assertNotEqual(self.G[rowInds[i], colInds[i]], 0)   
            
        self.assertEquals(self.G.getnnz(), rowInds.shape[0])
        self.assertEquals(self.G.sum(), self.G[rowInds, colInds].sum())        
        
        (inds, ) = self.a.nonzero()
        for i in range(inds.shape[0]): 
            self.assertNotEqual(self.a[inds[i]], 0)  
        
        #Try an array with no non zeros 
        nrow = 5 
        ncol = 7
        storageTypes = ["colMajor", "rowMajor"] 
        for storageType in storageTypes: 
            A = csarray((nrow, ncol), storageType=storageType)
            (rowInds, colInds) = A.nonzero()
            
            self.assertEquals(A.getnnz(), rowInds.shape[0])
            self.assertEquals(rowInds.shape[0], 0)
            self.assertEquals(colInds.shape[0], 0)
        
        (inds, ) = self.c.nonzero()   
        self.assertEquals(inds.shape[0], 0)
        
        #Zero size array 
        nrow = 0 
        ncol = 0
        A = csarray((nrow, ncol))
        (rowInds, colInds) = A.nonzero()
        self.assertEquals(A.getnnz(), rowInds.shape[0])
        self.assertEquals(rowInds.shape[0], 0)
        self.assertEquals(colInds.shape[0], 0)
        
        (inds, ) = self.d.nonzero()
        self.assertEquals(inds.shape[0], 0)

    def testDiag(self): 
        nptst.assert_array_equal(self.A.diag(), numpy.zeros(5))
        nptst.assert_array_equal(self.B.diag(), numpy.array([  0,    0,    0,   -0.2,  12.2]))
        nptst.assert_array_equal(self.C.diag(), numpy.zeros(100))
        nptst.assert_array_equal(self.H.diag(), numpy.array([  0,    0,    0,   -0.2,  12.2]))
        
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
        self.assertAlmostEquals(self.H.mean(), 0.4848571428571428)
        
        D = csarray((0, 0)) 
        self.assertTrue(math.isnan(D.mean()))
        
        self.assertEquals(self.F.mean(), 10/float(36))
        
        nptst.assert_array_equal(self.A.mean(0), self.A.sum(0)/self.A.shape[0])
        nptst.assert_array_equal(self.B.mean(0), self.B.sum(0)/self.B.shape[0]) 
        nptst.assert_array_equal(self.C.mean(0), self.C.sum(0)/self.C.shape[0])
        nptst.assert_array_equal(self.D.mean(0), self.D.sum(0)/self.D.shape[0])
        nptst.assert_array_equal(self.F.mean(0), self.F.sum(0)/float(self.F.shape[0]))
        nptst.assert_array_equal(self.G.mean(0), self.G.sum(0)/self.G.shape[0])
        nptst.assert_array_equal(self.H.mean(0), self.H.sum(0)/self.H.shape[0])
        
        nptst.assert_array_equal(self.A.mean(1), self.A.sum(1)/self.A.shape[1])
        nptst.assert_array_equal(self.B.mean(1), self.B.sum(1)/self.B.shape[1])
        nptst.assert_array_equal(self.C.mean(1), self.C.sum(1)/self.C.shape[1])
        nptst.assert_array_equal(self.D.mean(1), self.D.sum(1)/self.D.shape[1])
        nptst.assert_array_equal(self.F.mean(1), self.F.sum(1)/float(self.F.shape[1]))
        nptst.assert_array_equal(self.G.mean(1), self.G.sum(1)/self.G.shape[1])
        nptst.assert_array_equal(self.H.mean(1), self.H.sum(1)/self.H.shape[1])
        
        self.assertEquals(self.a.mean(), 1.6199999999999999)
        self.assertEquals(self.b.mean(), 1.6)
        self.assertEquals(self.c.mean(), 0.0)
        self.assertTrue(math.isnan(self.d.mean()))
    
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
        
        G = self.G.copy() 
        G[0, 0] = -15
        self.assertEquals(G[0, 0], -15)
        self.assertEquals(self.G[0, 0], 23)
        
        #Now try with 1d arrays 
        a2 = self.a.copy()
        self.a[0] = 10
        self.a[3] = 1
        self.a[4] = 2
        
        self.assertEquals(a2[0], 23)
        self.assertEquals(a2[3], 1.2)
        self.assertEquals(a2[4], -8)
        
        
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
        
        C = self.H * val
        
        self.assertEquals(self.H[0, 1], 1)
        self.assertEquals(self.H[1, 3], 5.2)
        self.assertEquals(self.H[3, 3], -0.2)
        self.assertEquals(self.H[0, 6], -1.23)
        self.assertEquals(self.H[4, 4], 12.2)
        
        self.assertEquals(C[0, 1], self.H[0, 1]*val)
        self.assertEquals(C[1, 3], self.H[1, 3]*val)
        self.assertEquals(C[3, 3], self.H[3, 3]*val)
        self.assertEquals(C[0, 6], self.H[0, 6]*val)
        self.assertEquals(C[4, 4], self.H[4, 4]*val)
        
        G = self.F*val 
        self.assertEquals(self.F[0, 0], 23)
        self.assertEquals(G[0, 0], 46)
        
        #Now with vectors 
        a2 = self.a*val 
        self.assertEquals(a2[0], self.a[0]*val)
        self.assertEquals(a2[3], self.a[3]*val)
        self.assertEquals(a2[4], self.a[4]*val)

    def testTrace(self): 
        self.assertEquals(self.A.trace(), 0)
        self.assertEquals(self.B.trace(), 12)
        self.assertEquals(self.C.trace(), 0)
        self.assertEquals(self.D.trace(), 23.1)
        self.assertEquals(self.F.trace(), 23)
        self.assertEquals(self.G.trace(), 23)
        self.assertEquals(self.H.trace(), 12)

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
        
        G = self.G.toarray()
        self.assertEquals(type(G), numpy.ndarray)
        self.assertEquals(G.shape, self.G.shape)
        self.assertEquals(G[0, 0], 23)
        self.assertEquals(G[2, 0], -3)
        self.assertEquals(G[3, 0], -10)
        self.assertEquals(G[2, 1], -5)
        self.assertEquals(G[3, 1], 5)
        self.assertAlmostEquals(G.sum(), self.G.sum())
        
        #Vectors 
        a2 = self.a.toarray()
        self.assertEquals(type(a2), numpy.ndarray)
        self.assertEquals(a2.shape, self.a.shape)
        self.assertEquals(a2[0], 23)
        self.assertEquals(a2[3], 1.2)
        self.assertEquals(a2[4], -8)
        self.assertAlmostEquals(a2.sum(), self.a.sum())
        
        b2 = self.b.toarray()
        self.assertEquals(type(b2), numpy.ndarray)
        self.assertEquals(b2.shape, self.b.shape)
        self.assertEquals(b2[0], 23)
        self.assertEquals(b2[5], 1)
        self.assertEquals(b2[8], -8)
        self.assertAlmostEquals(b2.sum(), self.b.sum())
        
       
    def testMin(self):
       self.assertEquals(self.A.min(), 0)
       self.assertEquals(self.B.min(), -1.23)
       self.assertEquals(self.C.min(), -1.23)
       self.assertEquals(self.D.min(), -10)
       self.assertTrue(math.isnan(self.E.min()))
       self.assertEquals(self.F.min(), -10)
       self.assertEquals(self.G.min(), -10)
       self.assertEquals(self.H.min(), -1.23)      
       
       self.assertEquals(self.a.min(), -8)
       self.assertEquals(self.b.min(), -8)
       self.assertEquals(self.c.min(), 0)
       self.assertTrue(math.isnan(self.d.min()))
       
    def testMax(self):
       self.assertEquals(self.A.max(), 0)
       self.assertEquals(self.B.max(), 12.2)
       self.assertEquals(self.C.max(), 12.2)
       self.assertEquals(self.D.max(), 23.1)
       self.assertTrue(math.isnan(self.E.max()))
       self.assertEquals(self.F.max(), 23)
       self.assertEquals(self.G.max(), 23)
       self.assertEquals(self.H.max(), 12.2)
       
       self.assertEquals(self.a.max(), 23)
       self.assertEquals(self.b.max(), 23)
       self.assertEquals(self.c.max(), 0)
       self.assertTrue(math.isnan(self.d.max()))
     
    #@unittest.skip("")
    def testVar(self):
       self.assertEquals(self.A.var(), self.A.toarray().var())
       self.assertAlmostEquals(self.B.var(), self.B.toarray().var())
       self.assertAlmostEquals(self.C.var(), self.C.toarray().var())
       self.assertAlmostEquals(self.D.var(), self.D.toarray().var())
       
       self.assertAlmostEquals(self.F.var(), self.F.toarray().var())
       self.assertAlmostEquals(self.G.var(), self.G.toarray().var())
       self.assertAlmostEquals(self.H.var(), self.H.toarray().var())
       
       self.assertAlmostEquals(self.a.var(), self.a.toarray().var())
       self.assertAlmostEquals(self.b.var(), self.b.toarray().var())
       
    #@unittest.skip("")
    def testStd(self):
       self.assertEquals(self.A.std(), self.A.toarray().std())
       self.assertAlmostEquals(self.B.std(), self.B.toarray().std())
       self.assertAlmostEquals(self.C.std(), self.C.toarray().std())
       self.assertAlmostEquals(self.D.std(), self.D.toarray().std())
       
       self.assertAlmostEquals(self.F.std(), self.F.toarray().std())
       self.assertAlmostEquals(self.G.std(), self.G.toarray().std())
       self.assertAlmostEquals(self.H.std(), self.H.toarray().std())
       
       self.assertAlmostEquals(self.a.std(), self.a.toarray().std())
       self.assertAlmostEquals(self.b.std(), self.b.toarray().std())
          
    def testAbs(self): 
       nptst.assert_array_equal(abs(self.A).toarray(), abs(self.A.toarray()))
       nptst.assert_array_equal(abs(self.B).toarray(), abs(self.B.toarray()))
       nptst.assert_array_equal(abs(self.C).toarray(), abs(self.C.toarray()))
       nptst.assert_array_equal(abs(self.D).toarray(), abs(self.D.toarray()))
       nptst.assert_array_equal(abs(self.F).toarray(), abs(self.F.toarray()))
       nptst.assert_array_equal(abs(self.G).toarray(), abs(self.G.toarray()))
       nptst.assert_array_equal(abs(self.H).toarray(), abs(self.H.toarray()))
       
       nptst.assert_array_equal(abs(self.a).toarray(), abs(self.a.toarray()))
       nptst.assert_array_equal(abs(self.b).toarray(), abs(self.b.toarray()))
       
    def testNeg(self): 
       nptst.assert_array_equal((-self.A).toarray(), -self.A.toarray())
       nptst.assert_array_equal((-self.B).toarray(), -self.B.toarray())
       nptst.assert_array_equal((-self.C).toarray(), -self.C.toarray())
       nptst.assert_array_equal((-self.D).toarray(), -self.D.toarray())
       
       nptst.assert_array_equal((-self.F).toarray(), -self.F.toarray())
       nptst.assert_array_equal((-self.G).toarray(), -self.G.toarray())
       nptst.assert_array_equal((-self.H).toarray(), -self.H.toarray())
       
       nptst.assert_array_equal((-self.a).toarray(), -self.a.toarray())
       nptst.assert_array_equal((-self.b).toarray(), -self.b.toarray())
       
    def testAdd(self): 
       #print(self.A.__add__(self.A._array))
       nptst.assert_array_equal((self.A + self.A).toarray(), self.A.toarray()*2)
       nptst.assert_array_equal((self.B + self.B).toarray(), self.B.toarray()*2)
       nptst.assert_array_equal((self.C + self.C).toarray(), self.C.toarray()*2)
       nptst.assert_array_equal((self.D + self.D).toarray(), self.D.toarray()*2)
       
       nptst.assert_array_equal((self.F + self.F).toarray(), self.F.toarray()*2)
       nptst.assert_array_equal((self.G + self.G).toarray(), self.G.toarray()*2)
       nptst.assert_array_equal((self.H + self.H).toarray(), self.H.toarray()*2)
       
       A = csarray((5, 5))
       A[0, 1] = 4
       A[1, 3] = 2
       A[3, 3] = 1
       
       B = csarray((5, 5))
       B[0, 2] = 9.2
       B[2, 3] = -5
       B[3, 4] = 12
       
       nptst.assert_array_equal((A + B).toarray(), A.toarray()+B.toarray())
       
       nptst.assert_array_equal((self.a + self.a).toarray(), self.a.toarray()*2)
       nptst.assert_array_equal((self.b + self.b).toarray(), self.b.toarray()*2)
       nptst.assert_array_equal((self.c + self.c).toarray(), self.c.toarray()*2)
       

    def testSub(self): 
       nptst.assert_array_equal((self.A - self.A).toarray(), self.A.toarray()*0)
       nptst.assert_array_equal((self.B - self.B).toarray(), self.B.toarray()*0)
       nptst.assert_array_equal((self.C - self.C).toarray(), self.C.toarray()*0)
       nptst.assert_array_equal((self.D - self.D).toarray(), self.D.toarray()*0)
       nptst.assert_array_equal((self.F - self.F).toarray(), self.F.toarray()*0)
       nptst.assert_array_equal((self.G - self.G).toarray(), self.G.toarray()*0)
       nptst.assert_array_equal((self.H - self.H).toarray(), self.H.toarray()*0)
       
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
       
       nptst.assert_array_equal((self.a - self.a).toarray(), self.a.toarray()*0)
       nptst.assert_array_equal((self.b - self.b).toarray(), self.b.toarray()*0)
       nptst.assert_array_equal((self.c - self.c).toarray(), self.c.toarray()*0)
              

    def testHadamard(self): 
       nptst.assert_array_equal((self.A.hadamard(self.A)).toarray(), (self.A.toarray())**2)
       nptst.assert_array_equal((self.B.hadamard(self.B)).toarray(), self.B.toarray()**2)
       nptst.assert_array_equal((self.C.hadamard(self.C)).toarray(), self.C.toarray()**2)
       nptst.assert_array_equal((self.D.hadamard(self.D)).toarray(), self.D.toarray()**2)
       
       nptst.assert_array_equal((self.F.hadamard(self.F)).toarray(), self.F.toarray()**2)
       nptst.assert_array_equal((self.G.hadamard(self.G)).toarray(), self.G.toarray()**2)
       nptst.assert_array_equal((self.H.hadamard(self.H)).toarray(), self.H.toarray()**2)
       
       for storageType in self.storageTypes: 
           A = csarray((5, 5), storageType=storageType)
           A[0, 1] = 4
           A[2, 3] = -1.2
           A[1, 3] = 2
           A[3, 3] = 1
           
           B = csarray((5, 5), storageType=storageType)
           B[0, 2] = 9.2
           B[2, 3] = -5
           B[3, 4] = 12
           B[3, 3] = 12
           
           C = csarray((5, 5), storageType=storageType)
           
           nptst.assert_array_equal((A.hadamard(B)).toarray(), A.toarray()*B.toarray())
           nptst.assert_array_equal((A.hadamard(C)).toarray(), C.toarray())
       
       nptst.assert_array_equal((self.a.hadamard(self.a)).toarray(), (self.a.toarray())**2)
       nptst.assert_array_equal((self.b.hadamard(self.b)).toarray(), (self.b.toarray())**2)
       nptst.assert_array_equal((self.c.hadamard(self.c)).toarray(), (self.c.toarray())**2)

    def testReserve(self): 
       for storageType in self.storageTypes: 
           A = csarray((5, 5), storageType=storageType)
           A.reserve(5)
           A[0, 1] = 4
           A[2, 3] = -1.2
           A[1, 3] = 2
           A[3, 3] = 1
      
    def testCompress(self): 
        for storageType in self.storageTypes: 
           A = csarray((5, 5), storageType=storageType)
           A[0, 1] = 4
           A[2, 3] = -1.2
           A[1, 3] = 2
           A[3, 3] = 1
           A.compress()
       
    def testDot(self): 
       for storageType in self.storageTypes: 
           A = csarray((5, 5), storageType=storageType)
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
       
       self.assertEquals((self.a.dot(self.a)), (self.a.dot(self.a)))
       self.assertEquals((self.b.dot(self.b)), (self.b.dot(self.b)))
       self.assertEquals((self.c.dot(self.c)), (self.c.dot(self.c)))
       
       #Now test dot product with ndarray
       D = self.D.toarray()
       B = self.D.dot(D)
       
       nptst.assert_array_equal(B, D.dot(D))
       
       A = numpy.random.rand(10, 5)
       B = numpy.random.rand(5, 6)
       
       C = A.dot(B)
       
       Ahat = csarray(A)
       Chat = Ahat.dot(B)

       nptst.assert_array_equal(C, Chat)
       
       #Try some random matrices 
       numRuns = 10 
       for i in range(numRuns): 
           m = numpy.random.randint(1, 50) 
           n = numpy.random.randint(1, 50) 
           p = numpy.random.randint(1, 50) 
           
           A = numpy.random.rand(m, n)
           B = numpy.random.rand(n, p)
           
           C = A.dot(B)
           
           Ahat = csarray(A)
           Chat = Ahat.dot(B)
           
           nptst.assert_array_almost_equal(C, Chat)

    def testPdot(self): 
       """
       D = numpy.ascontiguousarray(self.D.toarray())
       B = self.D.pdot(D)

       nptst.assert_array_equal(B, D.dot(D))
       
       A = numpy.random.rand(10, 5)
       B = numpy.random.rand(5, 6)
       
       C = A.dot(B)
       
       Ahat = csarray(A)
       Chat = Ahat.pdot(B)

       nptst.assert_array_almost_equal(C, Chat)
       """
       
       #Try some random matrices 
       numRuns = 10 
       for i in range(numRuns): 
           m = numpy.random.randint(1, 50) 
           n = numpy.random.randint(1, 50) 
           p = numpy.random.randint(1, 50) 
           
           A = numpy.random.rand(m, n)
           B = numpy.random.rand(n, p)
           
           C = A.dot(B)
           
           Ahat = csarray(A, storageType="rowMajor")
           Chat = Ahat.pdot(B)
           
           nptst.assert_array_almost_equal(C, Chat, 3)

    def testTranspose(self): 
        
       for storageType in self.storageTypes: 
           A = csarray((5, 5), storageType=storageType)
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
       nptst.assert_array_equal(self.G.transpose().toarray(), self.G.toarray().T)
       nptst.assert_array_equal(self.H.transpose().toarray(), self.H.toarray().T)

    def testOnes(self): 
        self.a.ones()
        nptst.assert_array_equal(self.a.toarray(), numpy.ones(self.a.shape[0]))

        self.A.ones()
        nptst.assert_array_equal(self.A.toarray(), numpy.ones(self.A.shape))
        
        self.G.ones()
        nptst.assert_array_equal(self.G.toarray(), numpy.ones(self.G.shape))        
        
    def testRowInds(self): 
        nptst.assert_array_equal(self.B.rowInds(0), numpy.array([1, 6]))
        nptst.assert_array_equal(self.B.rowInds(1), numpy.array([3]))
        
        nptst.assert_array_equal(self.H.rowInds(0), numpy.array([1, 6]))
        nptst.assert_array_equal(self.H.rowInds(1), numpy.array([3]))
        
        nptst.assert_array_equal(self.C.rowInds(0), numpy.array([1, 62]))
        nptst.assert_array_equal(self.C.rowInds(1), numpy.array([]))
        
    def testColInds(self): 
        nptst.assert_array_equal(self.B.colInds(3), numpy.array([1, 3]))
        nptst.assert_array_equal(self.B.colInds(1), numpy.array([0]))
        
        nptst.assert_array_equal(self.H.colInds(3), numpy.array([1, 3]))
        nptst.assert_array_equal(self.H.colInds(1), numpy.array([0]))
        
        nptst.assert_array_equal(self.D.colInds(0), numpy.array([0, 2, 3]))
        nptst.assert_array_equal(self.D.colInds(2), numpy.array([]))

    def testValues(self): 
        nptst.assert_array_equal(self.A.values(), self.A[self.A.nonzero()])
        nptst.assert_array_equal(self.B.values(), self.B[self.B.nonzero()])
        nptst.assert_array_equal(self.C.values(), self.C[self.C.nonzero()])
        nptst.assert_array_equal(self.D.values(), self.D[self.D.nonzero()])
        nptst.assert_array_equal(self.E.values(), self.E[self.E.nonzero()])
        nptst.assert_array_equal(self.F.values(), self.F[self.F.nonzero()])
        nptst.assert_array_equal(self.G.values(), self.G[self.G.nonzero()])
        nptst.assert_array_equal(self.H.values(), self.H[self.H.nonzero()])

    def testToScipyCsc(self): 
        try: 
            import scipy.sparse
        except ImportError: 
            raise        
        
        A = self.A.toScipyCsc()
        B = self.B.toScipyCsc()
        C = self.C.toScipyCsc()
        D = self.D.toScipyCsc()
        F = self.F.toScipyCsc()
        
        self.assertEquals(A.getnnz(), self.A.getnnz())
        self.assertEquals(B.getnnz(), self.B.getnnz())
        self.assertEquals(C.getnnz(), self.C.getnnz())
        self.assertEquals(D.getnnz(), self.D.getnnz())
        self.assertEquals(F.getnnz(), self.F.getnnz())
        
        #Now check elements are correct 
        (rowInds, colInds) = self.B.nonzero()
        for i in range(rowInds.shape[0]): 
            self.assertEquals(B[rowInds[i], colInds[i]], self.B[rowInds[i], colInds[i]])
            
        (rowInds, colInds) = self.C.nonzero()
        for i in range(rowInds.shape[0]): 
            self.assertEquals(C[rowInds[i], colInds[i]], self.C[rowInds[i], colInds[i]])
            
        (rowInds, colInds) = self.D.nonzero()
        for i in range(rowInds.shape[0]): 
            self.assertEquals(D[rowInds[i], colInds[i]], self.D[rowInds[i], colInds[i]])
            
        (rowInds, colInds) = self.F.nonzero()
        for i in range(rowInds.shape[0]): 
            self.assertEquals(F[rowInds[i], colInds[i]], self.F[rowInds[i], colInds[i]])

    def testToScipyCsr(self): 
        try: 
            import scipy.sparse
        except ImportError: 
            raise        
        
        G = self.G.toScipyCsr()
        H = self.H.toScipyCsr()
        I = self.I.toScipyCsr()

        self.assertEquals(G.getnnz(), self.G.getnnz())
        self.assertEquals(H.getnnz(), self.H.getnnz())
        self.assertEquals(I.getnnz(), self.I.getnnz())

        #Now check elements are correct 
        (rowInds, colInds) = self.G.nonzero()
        for i in range(rowInds.shape[0]): 
            self.assertEquals(G[rowInds[i], colInds[i]], self.G[rowInds[i], colInds[i]])
            
        (rowInds, colInds) = self.H.nonzero()
        for i in range(rowInds.shape[0]): 
            self.assertEquals(H[rowInds[i], colInds[i]], self.H[rowInds[i], colInds[i]])
            
    def testPut(self): 
        A = csarray((10, 10))
        
        rowInds = numpy.array([1, 2, 5, 7], numpy.int)
        colInds = numpy.array([4, 1, 9, 0], numpy.int)
        vals = numpy.random.randn(rowInds.shape[0])
        
        A.put(vals, rowInds, colInds)
        
        for i in range(rowInds.shape[0]): 
            self.assertEquals(A[rowInds[i], colInds[i]], vals[i])
            
        self.assertEquals(A.nnz, rowInds.shape[0])

    def testPutInit(self): 
        A = csarray((10, 10), storageType="colMajor")  
        
        rowInds = numpy.array([1, 2, 5, 7, 8, 1], numpy.int)
        colInds = numpy.array([0, 0, 0, 1, 1, 2], numpy.int)
        vals = numpy.random.randn(rowInds.shape[0])
        
        A.put(vals, rowInds, colInds, True)
        
        for i in range(rowInds.shape[0]): 
            self.assertEquals(A[rowInds[i], colInds[i]], vals[i])
            
        self.assertEquals(A.nnz, rowInds.shape[0])
        
        #Test rowMajor format 
        A = csarray((10, 10), storageType="rowMajor") 
        rowInds = numpy.array([1, 1, 2, 5, 7, 8], numpy.int)
        colInds = numpy.array([0, 2, 0, 0, 1, 1], numpy.int)
        
        A.put(vals, rowInds, colInds, True)
        
        for i in range(rowInds.shape[0]): 
            self.assertEquals(A[rowInds[i], colInds[i]], vals[i])
            
        self.assertEquals(A.nnz, rowInds.shape[0])
        
        #Try a larger matrix 
        numpy.random.seed(21)
        m = 1000000
        n = 1000000      
        numInds = 1000
        
        inds = numpy.random.randint(0, m*n, numInds)
        inds = numpy.unique(inds)
        vals = numpy.random.randn(inds.shape[0])
        
        rowInds, colInds = numpy.unravel_index(inds, (m, n), order="FORTRAN")
        A = csarray((m, n), storageType="colMajor")
        A.put(vals, rowInds, colInds)
        
        for i in range(vals.shape[0]): 
            self.assertEquals(A[rowInds[i], colInds[i]], vals[i])
            
        self.assertEquals(A.nnz, vals.shape[0])
        

if __name__ == "__main__":
    unittest.main()
    
    
