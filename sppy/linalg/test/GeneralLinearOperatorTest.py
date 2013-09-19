import logging
import sys
import unittest
import numpy
import numpy.testing as nptst 
import math 

from sppy import csarray, rand 
from sppy.linalg import GeneralLinearOperator

class GeneralLinearOperatorTest(unittest.TestCase):
    def setUp(self):
        numpy.set_printoptions(suppress=True, precision=3)
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        
        shape = (15, 15)
        density = 0.5        
        self.A = rand(shape, density)
        self.A = self.A + self.A.T
        
        
    def testAsLinearOperator(self): 
        try: 
            import scipy.sparse.linalg
            #Check it works for the eigenvalue solver 
            L = GeneralLinearOperator.asLinearOperator(self.A)
            k = 10 
            s, V = scipy.sparse.linalg.eigsh(L, k, which="LA")
            
            inds = numpy.flipud(numpy.argsort(s))
            s = s[inds]  
            V = V[:, inds]
            
            B = self.A.toarray()
            s2, V2 = numpy.linalg.eigh(B)
            
            inds = numpy.flipud(numpy.argsort(s2))[0:k]
            s2 = s2[inds]
            V2 = V2[:, inds]
            
            nptst.assert_array_almost_equal(s, s2[0:s.shape[0]])
            
            nptst.assert_array_almost_equal((V*s).dot(V.T), (V2*s2).dot(V2.T)) 
        except:
            print("Couldn't test asLinearOperator") 
        
    def testAsLinearOperatorSum(self): 
        
        n = 10 
        m = 5 
        density = 0.3
        A = rand((n, m), density)    
        B = rand((n, m), density)
        
        L = GeneralLinearOperator.asLinearOperator(A)
        M = GeneralLinearOperator.asLinearOperator(B)

        N = GeneralLinearOperator.asLinearOperatorSum(L, M)
        
        k = 3
        V = numpy.random.rand(m, k)
        W = numpy.random.rand(n, k)
        
        U = N.matmat(V)
        U2 = A.dot(V) + B.dot(V)
        nptst.assert_array_almost_equal(U, U2)
       
        U = N.rmatmat(W)
        U2 = A.T.dot(W) + B.T.dot(W)
        nptst.assert_array_almost_equal(U, U2)     
        
        v = numpy.random.rand(m)
        w = numpy.random.rand(n)        
        
        u = N.matvec(v)
        u2 = A.dot(v) + B.dot(v)
        nptst.assert_array_almost_equal(u, u2)
        
        u = N.rmatvec(w)
        u2 = A.T.dot(w) + B.T.dot(w)
        nptst.assert_array_almost_equal(u, u2)
        
        #See if we get an error if A, B are different shapes 
        B = rand((m, n), 0.1)
        M = GeneralLinearOperator.asLinearOperator(B)
        self.assertRaises(ValueError, GeneralLinearOperator.asLinearOperatorSum, L, M)
        
if __name__ == "__main__":
    unittest.main()
    
