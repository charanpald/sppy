
import unittest
import numpy
import sppy 
import sppy.linalg
import numpy.testing as nptst 
from sppy.lib.Util import Util

class coreTest(unittest.TestCase):
    def setUp(self):
        numpy.random.rand(21)
        numpy.set_printoptions(suppress=True, linewidth=200, precision=3)

    def testRsvd(self): 
        n = 100 
        m = 80
        A = sppy.rand((m, n), 0.5)
        
        ks = [10, 20, 30, 40] 
        q = 2 
        
        lastError = numpy.linalg.norm(A.toarray())        
        
        for k in ks: 
            U, s, V = sppy.linalg.rsvd(A, k, q)
            
            nptst.assert_array_almost_equal(U.T.dot(U), numpy.eye(k))
            nptst.assert_array_almost_equal(V.T.dot(V), numpy.eye(k))
            A2 = (U*s).dot(V.T)
            
            error = numpy.linalg.norm(A.toarray() - A2)
            self.assertTrue(error <= lastError)
            lastError = error 
            
            #Compare versus exact svd 
            U, s, V = numpy.linalg.svd(numpy.array(A.toarray()))
            inds = numpy.flipud(numpy.argsort(s))[0:k*2]
            U, s, V = Util.indSvd(U, s, V, inds)
            
            Ak = (U*s).dot(V.T)
            
            error2 = numpy.linalg.norm(A.toarray() - Ak)
            self.assertTrue(error2 <= error)

    def testRsvd2(self): 
        """
        We test the situation in which one gives an initial omega matrix 
        for the random projections. 
        """
        numRuns = 10 
        
        for i in range(numRuns): 
            m, n = numpy.random.randint(10, 100), numpy.random.randint(10, 100) 
            X = numpy.random.rand(m, n)
            
            k = numpy.random.randint(5, min(m, n)) 
            U, s, V = sppy.linalg.rsvd(X, k)
    
            D = numpy.random.rand(m, n)*0.1
    
            Y = X + D 
            U2, s2, V2 = sppy.linalg.rsvd(Y, k, p=0, q=0)
    
            U3, s3, V3 = sppy.linalg.rsvd(Y, k, p=0, q=0, omega=V)
            
            error1 = numpy.linalg.norm(Y - (U2*s2).dot(V2.T)) 
            error2 = numpy.linalg.norm(Y - (U3*s3).dot(V3.T))
            
            self.assertTrue(error1 >= error2)


    def testNorm(self): 
        n = 100 
        m = 80
        numRuns = 10         
        
        for i in range(numRuns): 
            A = sppy.rand((m, n), 0.5)
            
            self.assertAlmostEquals(numpy.linalg.norm(A.toarray()), sppy.linalg.norm(A))
            
    def testBiCGSTAB(self): 
        #This doesn't always converge 
        numRuns = 10 
        
        for i in range(numRuns): 
            n = numpy.random.randint(5, 20)
            A = numpy.random.rand(n, n)
            x = numpy.random.rand(n)
            
            b = A.dot(x)
            
            A = sppy.csarray(A)
            
            x2, output = sppy.linalg.biCGSTAB(A, b, tol=10**-6, maxIter=n)
            
            if output == 0: 
                nptst.assert_array_almost_equal(x, x2, 3)
                
        #Try with bad input 
        m = 3
        n = 5
        A = numpy.random.rand(n, m)
        A = sppy.csarray(A)
        x = numpy.random.rand(m)
        b = A.dot(x)
        
        self.assertRaises(ValueError, sppy.linalg.biCGSTAB, A, b)
        
        A = numpy.random.rand(n, n)
        A = sppy.csarray(A)
        b = numpy.array(n+1)
        self.assertRaises(ValueError, sppy.linalg.biCGSTAB, A, b)

    def testDiag(self): 
        n = 5
        a = numpy.random.rand(n)
        
        A = sppy.diag(a)
        b = A.diag()

        for i in range(n): 
            self.assertEquals(A[i,i], a[i])
        
        self.assertEquals(A.nnz, n)
        
        nptst.assert_array_almost_equal(a, b)

    def testEye(self): 
        n = 5 
        A = sppy.eye(n)
        
        for i in range(n): 
            self.assertEquals(A[i,i], 1)
        
        self.assertEquals(A.nnz, n)

if __name__ == '__main__':
    unittest.main()

