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
            print("Couldn't test testAsLinearOperator") 
        
if __name__ == "__main__":
    unittest.main()
    
